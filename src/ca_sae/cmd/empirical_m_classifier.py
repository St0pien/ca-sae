import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lapsum.topk import soft_topk

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.eval.posthoc_M import (
    compute_empirical_feature_class_map,
)


@torch.inference_mode()
def compute_empirical_matrix(
    activations_path: str,
    model,
    num_classes: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_examples: int | None = None,
):
    """
    Compute the empirical feature-class selection matrix

        A[i, c] = P(feature i is selected | class = c)

    using the same helper as the CA honesty evaluation.
    """

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

    empirical_A, _, total_examples = compute_empirical_feature_class_map(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=device,
    )

    return empirical_A.to(torch.float64), total_examples


def build_posthoc_matrix(
    train_A: torch.Tensor,
    rho: float,
    budget_mode: str,
):
    """
    Construct a post-hoc feature-class annotation from the empirical
    training matrix.

    uniform:
        Every feature gets k_i = round(rho) class associations.

    estimated:
        Allocate exactly Ktot = round(rho * d) associations to the
        largest entries of train_A globally.

    Returns:
        posthoc_M: [d, C]
            Binary feature-class association matrix.

        k: [d]
            Number of class associations assigned to each feature.
    """

    d, c = train_A.shape

    Ktot = int(round(rho * d))
    Ktot = min(Ktot, d * c)

    if Ktot < 1:
        raise ValueError(
            f"rho={rho} gives total association budget K={Ktot}. "
            "rho must be positive."
        )

    if budget_mode == "uniform":
        k_int = int(round(rho))

        if k_int < 1:
            raise ValueError(f"Uniform budget must be at least 1, got rho={rho}.")

        if k_int > c:
            raise ValueError(
                f"Uniform budget k={k_int} exceeds number of classes C={c}."
            )

        k = torch.full(
            (d,),
            float(k_int),
            dtype=torch.float64,
            device=train_A.device,
        )

        posthoc_M = torch.zeros_like(train_A)

        for i in range(d):
            top_classes = torch.topk(
                train_A[i],
                k=k_int,
                dim=0,
            ).indices

            posthoc_M[i, top_classes] = 1.0

        return posthoc_M, k

    elif budget_mode == "estimated":
        # Give exactly Ktot associations to the largest empirical
        # feature-class associations in the training data.
        flat_A = train_A.flatten()

        _, top_indices = torch.topk(
            flat_A,
            k=Ktot,
            dim=0,
        )

        hard_M = torch.zeros_like(train_A)

        hard_M.flatten()[top_indices] = 1.0

        k = hard_M.sum(dim=1)

        posthoc_M = soft_topk(train_A, k.unsqueeze(1), 0.001)

        return posthoc_M, k

    else:
        raise ValueError(
            f"Unknown budget_mode={budget_mode!r}. "
            "Expected 'uniform' or 'estimated'."
        )


@torch.inference_mode()
def evaluate_classifier(
    model,
    activations_path: str,
    posthoc_M: torch.Tensor,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    num_classes: int,
    max_examples: int | None = None,
):
    """
    Evaluate the free classifier induced by an ad-hoc feature-class
    matrix M.

    For each input x:

        selected = SAE Top-k features
        pi_i(x) = 1/k(x) if feature i is selected else 0

        scores(x) = pi(x) @ M

        prediction = argmax_c scores_c(x)

    No classifier is fitted after construction of M.
    """

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

    c = num_classes
    d = model.dict_size

    # --------------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------------

    total_examples = 0

    correct = 0
    correct_top5 = 0
    correct_top10 = 0

    confusion = torch.zeros(
        (c, c),
        dtype=torch.long,
        device=device,
    )

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

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    for batch in tqdm(loader):
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

        # ----------------------------------------------------------
        # SAE hard Top-k selection
        # ----------------------------------------------------------

        encoded_acts = model.encode(x)
        k_hat = (encoded_acts > 0).float().sum(dim=-1)

        selected = encoded_acts > 0

        # ----------------------------------------------------------
        # pi(x)
        #
        # Each selected feature contributes exactly 1/k.
        # ----------------------------------------------------------

        pi = selected.to(torch.float32) / k_hat.unsqueeze(1).float()

        # ----------------------------------------------------------
        # Class scores
        #
        # scores[b, c] = sum_i pi[b, i] * M[i, c]
        # ----------------------------------------------------------

        scores = pi @ posthoc_M.to(torch.float32)

        predictions = scores.argmax(dim=1)

        # ----------------------------------------------------------
        # Top-k predictions
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------------

        flat_indices = labels * c + predictions

        batch_confusion = torch.bincount(
            flat_indices,
            minlength=c * c,
        ).reshape(c, c)

        confusion += batch_confusion

        # ----------------------------------------------------------
        # Score / margin statistics
        # ----------------------------------------------------------

        row_indices = torch.arange(
            batch_size_actual,
            device=device,
        )

        true_scores = scores[
            row_indices,
            labels,
        ]

        true_class_mask = torch.zeros_like(
            scores,
            dtype=torch.bool,
        )

        true_class_mask[
            row_indices,
            labels,
        ] = True

        scores_without_true = scores.masked_fill(
            true_class_mask,
            float("-inf"),
        )

        max_wrong_scores = scores_without_true.max(
            dim=1,
        ).values

        margins = true_scores - max_wrong_scores

        true_score_sum += true_scores.double().sum()
        wrong_score_sum += max_wrong_scores.double().sum()
        margin_sum += margins.double().sum()

        total_examples += batch_size_actual

    # --------------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------------

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

    return {
        "num_examples": total_examples,
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
        "confusion_matrix": confusion.cpu().tolist(),
        "precision_per_class": precision_per_class.cpu().tolist(),
        "recall_per_class": recall_per_class.cpu().tolist(),
        "f1_per_class": f1_per_class.cpu().tolist(),
        "support_per_class": true_count.cpu().long().tolist(),
        "predicted_per_class": predicted_count.cpu().long().tolist(),
    }


def main(
    architecture: str,
    checkpoint_path: str,
    test_activations_path: str,
    train_activations_path: str | None,
    precomputed_train_matrix: str | None,
    output_path: str | None = None,
    rho: float = 5.0,
    num_classes: int = 1000,
    budget_mode: str = "uniform",
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
):
    """
    Evaluate a zero-shot/free classifier induced by an ad-hoc
    feature-class matrix.

    The feature-class matrix is estimated from the training data:

        A_train[i, c] =
            P(feature i is selected | class c)

    A binary post-hoc matrix M is then constructed from A_train.

    The classifier evaluated on the test set is:

        pi_i(x) = 1/k(x) if feature i is selected
                  0 otherwise

        scores(x) = pi(x) @ M

        prediction = argmax_c scores_c(x)

    No classifier is fitted on the test data.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path,
        device=device,
    )

    model.eval()

    d = model.dict_size
    c = num_classes

    # ------------------------------------------------------------------
    # Compute empirical matrix on TRAIN data
    # ------------------------------------------------------------------

    print("\nComputing empirical feature-class matrix " "on training data...")

    if train_activations_path is not None:
        print("\nComputing empirical feature-class matrix on training data...")
        train_A, total_train_examples = compute_empirical_matrix(
            activations_path=train_activations_path,
            model=model,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_examples=max_train_examples,
        )
    else:
        print(
            f"Loading precomputed empirical feature-class matrix from: {precomputed_train_matrix}"
        )
        train_A = torch.load(precomputed_train_matrix).to(device)
        total_train_examples = -1

    # ------------------------------------------------------------------
    # Construct post-hoc matrix
    # ------------------------------------------------------------------

    print("Constructing post-hoc feature-class matrix...")

    posthoc_M, k = build_posthoc_matrix(
        train_A=train_A,
        rho=rho,
        budget_mode=budget_mode,
    )

    Ktot = int(k.sum().item())

    # ------------------------------------------------------------------
    # Evaluate on TEST data
    # ------------------------------------------------------------------

    print("Evaluating free classifier on test data...")

    eval_results = evaluate_classifier(
        model=model,
        activations_path=test_activations_path,
        posthoc_M=posthoc_M,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        num_classes=num_classes,
        max_examples=max_test_examples,
    )

    # ------------------------------------------------------------------
    # Budget statistics
    # ------------------------------------------------------------------

    num_zero_budget = int((k == 0).sum().item())

    num_annotated = d - num_zero_budget

    fraction_zero_budget = num_zero_budget / d

    fraction_annotated = num_annotated / d

    # ------------------------------------------------------------------
    # Per-feature information
    # ------------------------------------------------------------------

    train_cpu = train_A.cpu()
    M_cpu = posthoc_M.cpu()
    k_cpu = k.cpu()

    per_feature = []

    for i in range(d):
        ki = int(k_cpu[i].item())

        if ki > 0:
            claimed_classes = torch.topk(
                M_cpu[i],
                k=min(ki, c),
            ).indices.tolist()
        else:
            claimed_classes = []

        # Useful diagnostic:
        #
        # The empirical probability mass assigned to the claimed
        # classes in the training matrix.
        if ki > 0:
            train_claimed_mass = float(train_cpu[i, claimed_classes].sum())
        else:
            train_claimed_mass = 0.0

        per_feature.append(
            {
                "feature": i,
                "budget": ki,
                "train_claimed_mass": train_claimed_mass,
                "claimed_classes": claimed_classes,
            }
        )

    # ------------------------------------------------------------------
    # Combine results
    # ------------------------------------------------------------------

    results = {
        "architecture": architecture,
        "budget_mode": budget_mode,
        "rho": rho,
        "num_train_examples": total_train_examples,
        "num_test_examples": eval_results["num_examples"],
        "num_features": d,
        "num_classes": c,
        "total_association_budget": Ktot,
        "mean_feature_budget": float(k.mean()),
        "min_feature_budget": float(k.min()),
        "max_feature_budget": float(k.max()),
        "num_zero_budget_features": num_zero_budget,
        "fraction_zero_budget_features": fraction_zero_budget,
        "num_annotated_features": num_annotated,
        "fraction_annotated_features": fraction_annotated,
        # --------------------------------------------------------------
        # Classification metrics
        # --------------------------------------------------------------
        "accuracy": eval_results["accuracy"],
        "top5_accuracy": eval_results["top5_accuracy"],
        "top10_accuracy": eval_results["top10_accuracy"],
        "macro_precision": eval_results["macro_precision"],
        "macro_recall": eval_results["macro_recall"],
        "macro_f1": eval_results["macro_f1"],
        "weighted_precision": eval_results["weighted_precision"],
        "weighted_recall": eval_results["weighted_recall"],
        "weighted_f1": eval_results["weighted_f1"],
        "mean_true_class_score": eval_results["mean_true_class_score"],
        "mean_max_wrong_class_score": eval_results["mean_max_wrong_class_score"],
        "mean_classification_margin": eval_results["mean_classification_margin"],
        # --------------------------------------------------------------
        # Classification diagnostics
        # --------------------------------------------------------------
        "classes": [
            {
                "class": class_idx,
                "precision": eval_results["precision_per_class"][class_idx],
                "recall": eval_results["recall_per_class"][class_idx],
                "f1": eval_results["f1_per_class"][class_idx],
                "support": eval_results["support_per_class"][class_idx],
                "predicted": eval_results["predicted_per_class"][class_idx],
            }
            for class_idx in range(c)
        ],
        "confusion_matrix": eval_results["confusion_matrix"],
        # --------------------------------------------------------------
        # Feature-level post-hoc matrix
        # --------------------------------------------------------------
        "features": per_feature,
    }

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print("\n=== Ad-hoc Matrix Free Classifier Evaluation ===")

    print(f"Architecture:              {architecture}")
    print(f"Budget mode:               {budget_mode}")
    print(f"rho:                       {rho}")
    print(f"Train examples:            " f"{total_train_examples:,}")
    print(f"Test examples:             " f"{eval_results['num_examples']:,}")
    print(f"Features:                  {d:,}")
    print(f"Classes:                   {c:,}")
    print(f"Total association K:       {Ktot:,}")

    print()
    print(f"Mean feature budget:       " f"{k.mean().item():.3f}")
    print(f"Min feature budget:        " f"{k.min().item():.3f}")
    print(f"Max feature budget:        " f"{k.max().item():.3f}")

    print(
        f"Zero-budget features:      "
        f"{num_zero_budget:,} "
        f"({100 * fraction_zero_budget:.2f}%)"
    )

    print(
        f"Annotated features:        "
        f"{num_annotated:,} "
        f"({100 * fraction_annotated:.2f}%)"
    )

    print()

    print(f"Accuracy:                  " f"{results['accuracy']:.4f}")

    print(f"Top-5 accuracy:            " f"{results['top5_accuracy']:.4f}")

    print(f"Top-10 accuracy:           " f"{results['top10_accuracy']:.4f}")

    print()

    print(f"Macro precision:           " f"{results['macro_precision']:.4f}")

    print(f"Macro recall:              " f"{results['macro_recall']:.4f}")

    print(f"Macro F1:                  " f"{results['macro_f1']:.4f}")

    print()

    print(f"Weighted precision:        " f"{results['weighted_precision']:.4f}")

    print(f"Weighted recall:           " f"{results['weighted_recall']:.4f}")

    print(f"Weighted F1:               " f"{results['weighted_f1']:.4f}")

    print()

    print(f"Mean true-class score:     " f"{results['mean_true_class_score']:.4f}")

    print(
        f"Mean max wrong-class score:" f" {results['mean_max_wrong_class_score']:.4f}"
    )

    print(
        f"Mean classification margin:" f" {results['mean_classification_margin']:.4f}"
    )

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
            "Evaluate a zero-shot/free classifier induced "
            "by an ad-hoc feature-class matrix."
        )
    )

    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
        help="Architecture of the trained SAE.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the trained SAE checkpoint.",
    )

    matrix_args = parser.add_mutually_exclusive_group(required=True)

    matrix_args.add_argument(
        "--train-activations-path",
        type=str,
        default=None,
        help="Path to the training ActivationsDataset directory.",
    )

    matrix_args.add_argument(
        "--precomputed-matrix",
        type=str,
        default=None,
        help="Path to the precomputed empirical feature-class training matrix",
    )
    parser.add_argument(
        "--test-activations-path",
        type=str,
        required=True,
        help=("Path to the held-out/test " "ActivationsDataset directory."),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=5.0,
        help=("Average number of class associations " "per feature. Defaults to 5."),
    )

    parser.add_argument(
        "--budget-mode",
        type=str,
        choices=[
            "uniform",
            "estimated",
        ],
        default="uniform",
        help=(
            "How to construct the post-hoc matrix. "
            "'uniform' gives every feature round(rho) "
            "class associations. "
            "'estimated' allocates exactly rho*d "
            "associations to the largest entries of "
            "the empirical training matrix."
        ),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes.",
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
        help=("cuda, cuda:0, cpu, etc. Defaults to " "CUDA if available."),
    )

    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help=(
            "Optionally use only the first N " "training examples when constructing M."
        ),
    )

    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=None,
        help=("Optionally evaluate only the first N " "test examples."),
    )

    args = parser.parse_args()

    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        train_activations_path=args.train_activations_path,
        precomputed_train_matrix=args.precomputed_matrix,
        test_activations_path=args.test_activations_path,
        output_path=args.output_path,
        rho=args.rho,
        num_classes=args.num_classes,
        budget_mode=args.budget_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_train_examples=(args.max_train_examples),
        max_test_examples=(args.max_test_examples),
    )


if __name__ == "__main__":
    cli()
