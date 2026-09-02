import argparse
import json
from pathlib import Path

import torch
from lapsum.topk import soft_topk
from torch.utils.data import DataLoader

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.eval.matrix_honesty import calculate_matrix_honesty
from ca_sae.eval.posthoc_M import build_posthoc_M, compute_empirical_matrix
from ca_sae.sae.core import Dictionary


def _compute_matrix(
    activations_path: str,
    model: Dictionary,
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

    A, _, total_examples = compute_empirical_matrix(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=device,
    )

    return A, total_examples


def main(
    architecture: str,
    checkpoint_path: str,
    test_activations_path: str,
    precomputed_train_matrix: str,
    output_path: str | None = None,
    rho: float = 5.0,
    num_classes: int = 1000,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_test_examples: int | None = None,
):
    """
    Evaluate train/test stability of a post-hoc feature-class map.

    budget_mode="uniform":
        Every feature claims ceil(rho) classes.

    budget_mode="estimated":
        Allocate exactly Ktot = rho * d associations to the largest
        feature-class entries in A_train. This estimates adaptive
        per-feature budgets directly from the training data.
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

    Ktot = int(round(rho * d))

    # ------------------------------------------------------------------
    # Compute empirical matrix on TRAIN data
    # ------------------------------------------------------------------

    print(
        f"Loading precomputed empirical feature-class matrix from: {precomputed_train_matrix}"
    )
    train_A = torch.load(precomputed_train_matrix).to(device)

    # ------------------------------------------------------------------
    # Compute empirical matrix on TEST data
    # ------------------------------------------------------------------

    print("Computing empirical feature-class matrix on test data...")

    test_A, total_examples = _compute_matrix(
        activations_path=test_activations_path,
        model=model,
        num_classes=num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_examples=max_test_examples,
    )

    # ------------------------------------------------------------------
    # Construct post-hoc matrix and per-feature budgets
    # ------------------------------------------------------------------

    posthoc_train_M, posthoc_train_k = build_posthoc_M(
        empirical_A=train_A,
        rho=rho,
    )

    posthoc_test_M, posthoc_test_k = build_posthoc_M(test_A, rho=rho)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    matrix_honesty = calculate_matrix_honesty(
        posthoc_train_M, posthoc_train_k, posthoc_test_M, posthoc_test_k
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    results = {
        "rho": rho,
        "num_features": d,
        "num_classes": c,
        "total_association_budget": Ktot,
        "mean_feature_budget": float(posthoc_train_k.mean()),
        "min_feature_budget": float(posthoc_train_k.min()),
        "max_feature_budget": float(posthoc_train_k.max()),
        "mean_feature_test_budget": float(posthoc_test_k.mean()),
        "min_feature_test_budget": float(posthoc_test_k.min()),
        "max_feature_test_budget": float(posthoc_test_k.max()),
        "mean_cosine": matrix_honesty.cosine.mean(),
        "median_cosine": matrix_honesty.cosine.median(),
        "mean_precision_at_ceil_k": matrix_honesty.precision.mean(),
        "mean_recall_at_ceil_k": matrix_honesty.recall.mean(),
        "mean_f1_at_ceil_k": matrix_honesty.f1.mean(),
    }

    print("\n=== Empirical Honesty Evaluation ===")
    print(f"Examples:              {total_examples:,}")
    print(f"Features:              {d:,}")
    print(f"Classes:               {c:,}")
    print(f"Total association K:   {Ktot:.1f}")
    print(f"Mean feature budget:   {posthoc_train_k.mean().item():.3f}")
    print(f"Min feature budget:    {posthoc_train_k.min().item():.3f}")
    print(f"Max feature budget:    {posthoc_train_k.max().item():.3f}")
    print()
    print(f"Mean cosine:           {matrix_honesty.cosine.mean().item():.4f}")
    print(f"Median cosine:         {matrix_honesty.cosine.median().item():.4f}")
    print()
    print(f"Precision @ ceil(k):   {matrix_honesty.precision.mean().item():.4f}")
    print(f"Recall @ ceil(k):      {matrix_honesty.recall.mean().item():.4f}")
    print(f"F1 @ ceil(k):          {matrix_honesty.f1.mean().item():.4f}")

    # ------------------------------------------------------------------
    # Per-feature output
    # ------------------------------------------------------------------

    per_feature = []

    M_cpu = posthoc_train_M.cpu()
    posthoc_M_cpu = posthoc_test_M.cpu()
    k_cpu = posthoc_train_k.cpu()
    posthoc_k_cpu = posthoc_test_k.cpu()
    cosine_cpu = matrix_honesty.cosine.cpu()
    precision_cpu = matrix_honesty.precision.cpu()
    recall_cpu = matrix_honesty.recall.cpu()
    f1_cpu = matrix_honesty.f1.cpu()

    for i in range(d):
        ki = int(torch.ceil(k_cpu[i]).clamp(1, c))
        ki_posthoc = int(torch.ceil(posthoc_k_cpu[i]).clamp(1, c))

        claimed_classes = torch.topk(
            M_cpu[i],
            k=ki,
        ).indices.tolist()

        empirical_classes = torch.topk(
            posthoc_M_cpu[i],
            k=ki_posthoc,
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
    # Save results
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
        description=(
            "Evaluate train/test stability of a post-hoc "
            "feature-class map for a standard SAE."
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

    parser.add_argument(
        "--precomputed-matrix",
        type=str,
        required=True,
        help="Path to the precomputed empirical feature-class training matrix",
    )

    parser.add_argument(
        "--test-activations-path",
        type=str,
        required=True,
        help="Path to the held-out ActivationsDataset directory.",
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
        help=("Average number of class associations per feature. " "Defaults to 5."),
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
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
        "--max-test-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N test examples.",
    )

    args = parser.parse_args()

    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        precomputed_train_matrix=args.precomputed_matrix,
        test_activations_path=args.test_activations_path,
        output_path=args.output_path,
        rho=args.rho,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_test_examples=args.max_test_examples,
    )


if __name__ == "__main__":
    cli()
