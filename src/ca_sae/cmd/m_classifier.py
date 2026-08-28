import argparse
import json
from pathlib import Path

import torch
from lapsum.topk import soft_topk
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.ca_sae import ClassAlignedSAE


@torch.inference_mode()
def main(
    checkpoint_path: str,
    activations_path: str,
    output_path: str | None = None,
    permute_m: bool = False,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_examples: int | None = None,
):
    """
    Evaluate the zero-shot/free classifier induced by the learned
    feature-class affinity matrix M.

    For an input x, the SAE selects a hard Top-k set S(x).
    We construct

        pi_i(x) = 1 / k(x)   if i in S(x)
                  0           otherwise

    and compute class scores

        s(x) = M^T pi(x).

    The predicted class is

        argmax_c s_c(x).

    No classifier is fitted after training.

    Reports:
      - accuracy
      - top-5 / top-10 accuracy
      - macro precision / recall / F1
      - weighted precision / recall / F1
      - mean correct-class score
      - mean incorrect-class score
      - mean classification margin
      - per-class precision / recall / F1 / support
      - budget statistics
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = ClassAlignedSAE.from_pretrained(
        checkpoint_path,
        device=device,
    )

    model.eval()

    d = model.dict_size
    c = model.num_classes

    # ------------------------------------------------------------------
    # Load held-out/test activations
    # ------------------------------------------------------------------

    dataset = ActivationsDataset(activations_path)

    if max_examples is not None:
        max_examples = min(max_examples, len(dataset))
        dataset = torch.utils.data.Subset(
            dataset,
            range(max_examples),
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # ------------------------------------------------------------------
    # Construct learned M.
    #
    # This mirrors the construction used during training.
    # ------------------------------------------------------------------

    Ktot = model.features_per_class * d

    k = Ktot * torch.softmax(
        model.budget_vector,
        dim=0,
    )

    M = model.class_matrix
    if permute_m:
        M = M[torch.randperm(M.shape[0])]

    M = soft_topk(
        M,
        k.unsqueeze(-1),
        model.alpha,
        dim=1,
    ).to(torch.float32)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------

    total_examples = 0

    correct = 0
    correct_top5 = 0
    correct_top10 = 0

    # Confusion matrix.
    confusion = torch.zeros(
        (c, c),
        dtype=torch.long,
        device=device,
    )

    # Score statistics.
    true_score_sum = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device,
    )

    wrong_score_sum = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device,
    )

    margin_sum = torch.tensor(
        0.0,
        dtype=torch.float64,
        device=device,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    for batch in tqdm(loader):
        # ActivationsDataset is expected to return (activations, labels).
        #
        # If your dataset returns a different structure, adjust this
        # single line.
        x, labels = batch

        x = x.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        ).long()

        batch_size_actual = x.shape[0]

        # --------------------------------------------------------------
        # Hard sparse selection.
        #
        # encoded_acts contains only the selected Top-k activations.
        # --------------------------------------------------------------

        encoded_acts = model.encode(x)
        k_hat = model.estimate_k(x).long()

        # Recover the selection mask.
        #
        # topk_per_row produces zero everywhere except selected
        # coordinates. Since SAE activations are ReLU'd, selected
        # coordinates normally have positive values.
        selected = encoded_acts > 0

        # --------------------------------------------------------------
        # pi(x) = p(x) / k_hat
        #
        # For hard Top-k:
        #
        #   p_i = 1 if selected
        #         0 otherwise
        #
        # Therefore each selected feature contributes exactly 1/k.
        # --------------------------------------------------------------

        pi = selected.to(torch.float32) / k_hat.unsqueeze(1).float()

        # --------------------------------------------------------------
        # Class scores
        #
        # scores[b, c] =
        #     sum_i pi[b, i] * M[i, c]
        #
        # This is exactly M^T pi.
        # --------------------------------------------------------------

        scores = pi @ M

        predictions = scores.argmax(dim=1)

        # --------------------------------------------------------------
        # Top-k predictions
        # --------------------------------------------------------------

        topk_max = min(10, c)

        top_predictions = scores.topk(
            k=topk_max,
            dim=1,
        ).indices

        correct_mask = predictions == labels

        correct += correct_mask.sum().item()

        if c >= 5:
            correct_top5 += (
                (top_predictions[:, :5] == labels.unsqueeze(1)).any(dim=1).sum().item()
            )
        else:
            correct_top5 += correct_mask.sum().item()

        if c >= 10:
            correct_top10 += (
                (top_predictions[:, :10] == labels.unsqueeze(1)).any(dim=1).sum().item()
            )
        else:
            correct_top10 += correct_mask.sum().item()

        # --------------------------------------------------------------
        # Confusion matrix
        # --------------------------------------------------------------

        flat_indices = labels * c + predictions

        batch_confusion = torch.bincount(
            flat_indices,
            minlength=c * c,
        ).reshape(c, c)

        confusion += batch_confusion

        # --------------------------------------------------------------
        # Score / margin statistics
        # --------------------------------------------------------------

        true_scores = scores[
            torch.arange(batch_size_actual, device=device),
            labels,
        ]

        # For correctly classified examples, the predicted score is
        # the true-class score and the second-highest score is the
        # largest incorrect score.
        #
        # For incorrectly classified examples, we still compute the
        # margin between the true class and the predicted class.
        #
        # This gives:
        #
        #   margin = s_true - max_{c != true} s_c
        #
        true_class_mask = torch.zeros_like(scores, dtype=torch.bool)
        true_class_mask[
            torch.arange(batch_size_actual, device=device),
            labels,
        ] = True

        scores_without_true = scores.masked_fill(
            true_class_mask,
            float("-inf"),
        )

        max_wrong_scores = scores_without_true.max(dim=1).values

        margins = true_scores - max_wrong_scores

        true_score_sum += true_scores.double().sum()

        wrong_score_sum += max_wrong_scores.double().sum()

        margin_sum += margins.double().sum()

        total_examples += batch_size_actual

    # ------------------------------------------------------------------
    # Classification metrics from confusion matrix
    # ------------------------------------------------------------------

    tp = confusion.diag().double()

    predicted_count = confusion.sum(dim=0).double()
    true_count = confusion.sum(dim=1).double()

    precision_per_class = torch.where(
        predicted_count > 0,
        tp / predicted_count.clamp_min(1),
        torch.zeros_like(tp),
    )

    recall_per_class = torch.where(
        true_count > 0,
        tp / true_count.clamp_min(1),
        torch.zeros_like(tp),
    )

    f1_per_class = torch.where(
        (precision_per_class + recall_per_class) > 0,
        2
        * precision_per_class
        * recall_per_class
        / (precision_per_class + recall_per_class).clamp_min(1e-12),
        torch.zeros_like(tp),
    )

    valid_classes = true_count > 0

    macro_precision = precision_per_class[valid_classes].mean()
    macro_recall = recall_per_class[valid_classes].mean()
    macro_f1 = f1_per_class[valid_classes].mean()

    weighted_precision = (
        precision_per_class * true_count
    ).sum() / true_count.sum().clamp_min(1)

    weighted_recall = (
        recall_per_class * true_count
    ).sum() / true_count.sum().clamp_min(1)

    weighted_f1 = (f1_per_class * true_count).sum() / true_count.sum().clamp_min(1)

    accuracy = correct / total_examples
    top5_accuracy = correct_top5 / total_examples
    top10_accuracy = correct_top10 / total_examples

    mean_true_score = (true_score_sum / total_examples).item()

    mean_wrong_score = (wrong_score_sum / total_examples).item()

    mean_margin = (margin_sum / total_examples).item()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    results = {
        "num_examples": total_examples,
        "num_features": d,
        "num_classes": c,
        "total_association_budget": float(Ktot),
        "mean_feature_budget": float(k.mean()),
        "min_feature_budget": float(k.min()),
        "max_feature_budget": float(k.max()),
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "macro_precision": macro_precision.item(),
        "macro_recall": macro_recall.item(),
        "macro_f1": macro_f1.item(),
        "weighted_precision": weighted_precision.item(),
        "weighted_recall": weighted_recall.item(),
        "weighted_f1": weighted_f1.item(),
        "mean_true_class_score": mean_true_score,
        "mean_max_wrong_class_score": mean_wrong_score,
        "mean_classification_margin": mean_margin,
    }

    print("\n=== SoftSAE-CA Free Classifier Evaluation ===")
    print(f"Examples:                  {total_examples:,}")
    print(f"Features:                  {d:,}")
    print(f"Classes:                   {c:,}")
    print(f"Total association K:       {Ktot:.1f}")
    print(f"Mean feature budget:       {k.mean().item():.3f}")
    print(f"Min feature budget:        {k.min().item():.3f}")
    print(f"Max feature budget:        {k.max().item():.3f}")
    print()
    print(f"Accuracy:                   {accuracy:.4f}")
    print(f"Top-5 accuracy:             {top5_accuracy:.4f}")
    print(f"Top-10 accuracy:            {top10_accuracy:.4f}")
    print()
    print(f"Macro precision:            {macro_precision.item():.4f}")
    print(f"Macro recall:               {macro_recall.item():.4f}")
    print(f"Macro F1:                   {macro_f1.item():.4f}")
    print()
    print(f"Weighted precision:         {weighted_precision.item():.4f}")
    print(f"Weighted recall:            {weighted_recall.item():.4f}")
    print(f"Weighted F1:                {weighted_f1.item():.4f}")
    print()
    print(f"Mean true-class score:      {mean_true_score:.4f}")
    print(f"Mean max wrong-class score: {mean_wrong_score:.4f}")
    print(f"Mean classification margin: {mean_margin:.4f}")

    # ------------------------------------------------------------------
    # Per-class output
    # ------------------------------------------------------------------

    per_class = []

    precision_cpu = precision_per_class.cpu()
    recall_cpu = recall_per_class.cpu()
    f1_cpu = f1_per_class.cpu()
    support_cpu = true_count.cpu()
    predicted_cpu = predicted_count.cpu()

    for class_idx in range(c):
        per_class.append(
            {
                "class": class_idx,
                "precision": float(precision_cpu[class_idx]),
                "recall": float(recall_cpu[class_idx]),
                "f1": float(f1_cpu[class_idx]),
                "support": int(support_cpu[class_idx]),
                "predicted": int(predicted_cpu[class_idx]),
            }
        )

    results["classes"] = per_class

    # ------------------------------------------------------------------
    # Save confusion matrix
    # ------------------------------------------------------------------

    results["confusion_matrix"] = confusion.cpu().tolist()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(
                results,
                f,
                indent=2,
            )

        print(f"\nSaved results to {output_path}")

    return results


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the free classifier induced by "
            "SoftSAE-CA feature-class affinities."
        )
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained ClassAlignedSAE checkpoint.",
    )

    parser.add_argument(
        "--activations-path",
        type=str,
        required=True,
        help="Path to ActivationsDataset directory.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    parser.add_argument(
        "--permute-m",
        default=False,
        action="store_true",
        help="Use for perturbing class feature matrix",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cuda:0, cpu, etc. Defaults to CUDA if available.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples.",
    )

    args = parser.parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        activations_path=args.activations_path,
        output_path=args.output_path,
        permute_m=args.permute_m,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    cli()
