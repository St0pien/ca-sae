import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.ca_sae import ClassAlignedSAE
from ca_sae.eval.empirical_feature_class_map import compute_empirical_feature_class_map


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
    Evaluate the honesty of the learned feature-class affinity matrix M.

    For each dictionary feature i and class c, computes

        A[i, c] = P(feature i is selected | class = c)

    on held-out data, then compares A[i, :] with the learned M[i, :].

    Reports:
      - mean / median cosine similarity
      - precision / recall / F1 of the top ceil(k_i) claimed classes
      - per-feature metrics
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
    # Load held-out activations
    # ------------------------------------------------------------------

    dataset = ActivationsDataset(activations_path)

    if max_examples is not None:
        max_examples = min(max_examples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(max_examples))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    empirical_A, _, total_examples = compute_empirical_feature_class_map(
        model=model, loader=loader, num_classes=model.num_classes, device=device
    )

    # ------------------------------------------------------------------
    # Construct learned M.
    #
    # This mirrors ClassAlignedSAETrainer.get_agreement_loss().
    # ------------------------------------------------------------------

    Ktot = model.features_per_class * d

    k = Ktot * torch.softmax(
        model.budget_vector,
        dim=0,
    )

    # soft_topk operates on the model's class logits.
    M = model.class_matrix

    if permute_m:
        M = M[torch.randperm(M.shape[0])]

    from lapsum.topk import soft_topk

    M = soft_topk(
        M,
        k.unsqueeze(-1),
        model.alpha,
        dim=1,
    ).to(torch.float64)

    # ------------------------------------------------------------------
    # Cosine similarity between learned M[i,:] and empirical A[i,:].
    # ------------------------------------------------------------------

    M_norm = M.norm(dim=1)
    A_norm = empirical_A.norm(dim=1)

    cosine = (M * empirical_A).sum(dim=1) / (M_norm * A_norm).clamp_min(1e-12)

    valid_features = (M_norm > 0) & (A_norm > 0)

    valid_cosine = cosine[valid_features]

    # ------------------------------------------------------------------
    # Top-ceil(k_i) claimed-class precision / recall / F1.
    #
    # M tells us which classes feature i claims.
    #
    # A tells us which classes feature i is actually selected for.
    #
    # We use the top ceil(k_i) classes from A as the empirical
    # "ground-truth" class set.
    # ------------------------------------------------------------------

    precision = torch.zeros(d, dtype=torch.float64, device=device)
    recall = torch.zeros(d, dtype=torch.float64, device=device)
    f1 = torch.zeros(d, dtype=torch.float64, device=device)

    claimed_k = torch.ceil(k).long().clamp(min=1, max=c)

    for i in range(d):
        ki = int(claimed_k[i])

        predicted = torch.topk(
            M[i],
            k=ki,
            dim=0,
        ).indices

        actual = torch.topk(
            empirical_A[i],
            k=ki,
            dim=0,
        ).indices

        predicted_set = torch.zeros(
            c,
            dtype=torch.bool,
            device=device,
        )
        actual_set = torch.zeros(
            c,
            dtype=torch.bool,
            device=device,
        )

        predicted_set[predicted] = True
        actual_set[actual] = True

        tp = (predicted_set & actual_set).sum().float()

        p = tp / float(ki)
        r = tp / float(ki)

        precision[i] = p
        recall[i] = r

        if p + r > 0:
            f1[i] = 2 * p * r / (p + r)

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
        "mean_cosine": float(valid_cosine.mean()),
        "median_cosine": float(valid_cosine.median()),
        "mean_precision_at_ceil_k": float(precision.mean()),
        "mean_recall_at_ceil_k": float(recall.mean()),
        "mean_f1_at_ceil_k": float(f1.mean()),
    }

    print("\n=== SoftSAE-CA Honesty Evaluation ===")
    print(f"Examples:              {total_examples:,}")
    print(f"Features:              {d:,}")
    print(f"Classes:               {c:,}")
    print(f"Total association K:   {Ktot:.1f}")
    print(f"Mean feature budget:   {k.mean().item():.3f}")
    print(f"Min feature budget:    {k.min().item():.3f}")
    print(f"Max feature budget:    {k.max().item():.3f}")
    print()
    print(f"Mean cosine(M, A):     {valid_cosine.mean().item():.4f}")
    print(f"Median cosine(M, A):   {valid_cosine.median().item():.4f}")
    print()
    print(f"Precision @ ceil(k):   {precision.mean().item():.4f}")
    print(f"Recall @ ceil(k):      {recall.mean().item():.4f}")
    print(f"F1 @ ceil(k):          {f1.mean().item():.4f}")

    # ------------------------------------------------------------------
    # Per-feature output.
    # ------------------------------------------------------------------

    per_feature = []

    M_cpu = M.cpu()
    A_cpu = empirical_A.cpu()
    k_cpu = k.cpu()
    cosine_cpu = cosine.cpu()
    precision_cpu = precision.cpu()
    recall_cpu = recall.cpu()
    f1_cpu = f1.cpu()

    for i in range(d):
        ki = int(torch.ceil(k_cpu[i]).clamp(1, c))

        claimed_classes = torch.topk(
            M_cpu[i],
            k=ki,
        ).indices.tolist()

        empirical_classes = torch.topk(
            A_cpu[i],
            k=ki,
        ).indices.tolist()

        per_feature.append(
            {
                "feature": i,
                "budget": float(k_cpu[i]),
                "cosine": float(cosine_cpu[i]),
                "precision": float(precision_cpu[i]),
                "recall": float(recall_cpu[i]),
                "f1": float(f1_cpu[i]),
                "claimed_classes": claimed_classes,
                "empirical_classes": empirical_classes,
            }
        )

    results["features"] = per_feature

    # ------------------------------------------------------------------
    # Save results.
    # ------------------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to {output_path}")

    return results


def cli():
    parser = argparse.ArgumentParser(
        description="Evaluate honesty of SoftSAE-CA feature-class affinities."
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
