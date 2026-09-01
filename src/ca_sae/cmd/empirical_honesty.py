import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.eval.empirical_feature_class_map import (
    compute_empirical_feature_class_map,
)
from ca_sae.sae.ca_sae import ClassAlignedSAE


def compute_empirical_matrix(
    activations_path: str,
    model: ClassAlignedSAE,
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
        Every feature gets k_i = rho.

    estimated:
        Select the Ktot = rho * d largest entries of train_A globally.
        This produces a binary post-hoc M with adaptive per-feature
        budgets k_i.

    Returns:
        posthoc_M: [d, C]
        k:         [d]
    """

    d, c = train_A.shape

    Ktot = int(round(rho * d))
    Ktot = min(Ktot, d * c)

    if budget_mode == "uniform":
        k = torch.full(
            (d,),
            float(rho),
            dtype=torch.float64,
            device=train_A.device,
        )

        k_int = int(round(rho))

        if k_int < 1:
            raise ValueError("Uniform budget must be at least 1.")

        if k_int > c:
            raise ValueError(
                f"Uniform budget k={k_int} exceeds number of classes C={c}."
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

        posthoc_M = torch.zeros_like(train_A)

        posthoc_M.flatten()[top_indices] = 1.0

        k = posthoc_M.sum(dim=1)

        return posthoc_M, k

    else:
        raise ValueError(
            f"Unknown budget_mode={budget_mode!r}. "
            "Expected 'uniform' or 'estimated'."
        )


def evaluate_train_test_agreement(
    test_A: torch.Tensor,
    posthoc_M: torch.Tensor,
    k: torch.Tensor,
):
    """
    Compare a post-hoc feature-class map estimated from training data
    against the empirical feature-class map measured on held-out data.

    Returns both:
      - "active" metrics, computed only over features with k_i > 0
      - "all" metrics, where zero-budget features contribute zero

    For cosine, the cosine similarity of a zero vector is mathematically
    undefined. We therefore keep the active cosine metric as the standard
    cosine evaluation, while explicitly treating zero-budget features as
    cosine=0 for the coverage-aware "all" metric.
    """

    d, c = posthoc_M.shape

    # --------------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------------

    M_norm = posthoc_M.norm(dim=1)
    A_norm = test_A.norm(dim=1)

    cosine = (posthoc_M * test_A).sum(dim=1) / (M_norm * A_norm).clamp_min(1e-12)

    # Features with a nonzero annotation budget.
    active_features = k > 0

    # Features for which the ordinary cosine is mathematically valid.
    valid_cosine = (M_norm > 0) & (A_norm > 0)

    # Coverage-aware cosine:
    # zero-budget features explicitly contribute zero.
    cosine_all = cosine.clone()

    zero_budget = ~active_features
    cosine_all[zero_budget] = 0.0

    # If a feature has a nonzero budget but no empirical test activation,
    # its cosine is undefined. Keep it NaN rather than silently treating
    # it as zero.
    cosine_all[(active_features) & (A_norm == 0)] = float("nan")

    # --------------------------------------------------------------
    # Top-k precision / recall / F1
    # --------------------------------------------------------------

    precision = torch.full(
        (d,),
        float("nan"),
        dtype=torch.float64,
        device=test_A.device,
    )

    recall = torch.full_like(precision, float("nan"))
    f1 = torch.full_like(precision, float("nan"))

    for i in range(d):
        ki = int(k[i].item())

        # Zero-budget features have no claimed classes.
        # Their active-set metrics remain NaN.
        if ki == 0:
            continue

        ki = min(ki, c)

        claimed = torch.topk(
            posthoc_M[i],
            k=ki,
            dim=0,
        ).indices

        actual = torch.topk(
            test_A[i],
            k=ki,
            dim=0,
        ).indices

        claimed_set = torch.zeros(
            c,
            dtype=torch.bool,
            device=test_A.device,
        )

        actual_set = torch.zeros_like(claimed_set)

        claimed_set[claimed] = True
        actual_set[actual] = True

        tp = (claimed_set & actual_set).sum().float()

        p = tp / float(ki)
        r = tp / float(ki)

        precision[i] = p
        recall[i] = r

        if p + r > 0:
            f1[i] = 2 * p * r / (p + r)

    # --------------------------------------------------------------
    # Coverage-aware versions.
    #
    # Zero-budget features contribute exactly zero.
    # --------------------------------------------------------------

    precision_all = torch.nan_to_num(
        precision,
        nan=0.0,
    )

    recall_all = torch.nan_to_num(
        recall,
        nan=0.0,
    )

    f1_all = torch.nan_to_num(
        f1,
        nan=0.0,
    )

    return {
        "cosine": cosine,
        "cosine_all": cosine_all,
        "valid_cosine": valid_cosine,
        "precision": precision,
        "precision_all": precision_all,
        "recall": recall,
        "recall_all": recall_all,
        "f1": f1,
        "f1_all": f1_all,
        "active_features": active_features,
    }


def nanmean(x: torch.Tensor):
    """torch.nanmean wrapper returning a Python float."""
    return float(torch.nanmean(x))


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
    # Compute empirical matrix on TEST data
    # ------------------------------------------------------------------

    print("Computing empirical feature-class matrix on test data...")

    test_A, total_test_examples = compute_empirical_matrix(
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

    posthoc_M, k = build_posthoc_matrix(
        train_A=train_A,
        rho=rho,
        budget_mode=budget_mode,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    metrics = evaluate_train_test_agreement(
        test_A=test_A,
        posthoc_M=posthoc_M,
        k=k,
    )

    cosine = metrics["cosine"]
    cosine_all = metrics["cosine_all"]
    valid_cosine = metrics["valid_cosine"]

    precision = metrics["precision"]
    precision_all = metrics["precision_all"]

    recall = metrics["recall"]
    recall_all = metrics["recall_all"]

    f1 = metrics["f1"]
    f1_all = metrics["f1_all"]

    active_features = metrics["active_features"]

    # Active-only metrics.
    valid_precision = precision[active_features]
    valid_recall = recall[active_features]
    valid_f1 = f1[active_features]

    # ------------------------------------------------------------------
    # Budget statistics
    # ------------------------------------------------------------------

    num_zero_budget = int((k == 0).sum().item())
    num_annotated = d - num_zero_budget

    fraction_zero_budget = num_zero_budget / d
    fraction_annotated = num_annotated / d

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    results = {
        "budget_mode": budget_mode,
        "rho": rho,
        "num_train_examples": total_train_examples,
        "num_test_examples": total_test_examples,
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
        # ----------------------------------------------------------
        # Cosine
        # ----------------------------------------------------------
        # Existing interpretation: only features with valid
        # nonzero vectors.
        "mean_cosine_active": nanmean(cosine[valid_cosine]),
        "median_cosine_active": float(cosine[valid_cosine].median()),
        # New coverage-aware interpretation:
        # zero-budget features contribute zero.
        "mean_cosine_all": nanmean(cosine_all),
        "median_cosine_all": float(torch.nanmedian(cosine_all)),
        # ----------------------------------------------------------
        # Precision / recall / F1
        # ----------------------------------------------------------
        # Existing interpretation: only annotated features.
        "mean_precision_active": nanmean(valid_precision),
        "mean_recall_active": nanmean(valid_recall),
        "mean_f1_active": nanmean(valid_f1),
        # New interpretation: zero-budget features contribute zero.
        "mean_precision_all": float(precision_all.mean()),
        "mean_recall_all": float(recall_all.mean()),
        "mean_f1_all": float(f1_all.mean()),
    }

    print("\n=== Post-hoc Train → Test Honesty Evaluation ===")
    print(f"Budget mode:              {budget_mode}")
    print(f"rho:                      {rho}")
    print(f"Train examples:           {total_train_examples:,}")
    print(f"Test examples:            {total_test_examples:,}")
    print(f"Features:                 {d:,}")
    print(f"Classes:                  {c:,}")
    print(f"Total association K:      {Ktot:,}")
    print()

    print(f"Mean feature budget:      {k.mean().item():.3f}")
    print(f"Min feature budget:       {k.min().item():.3f}")
    print(f"Max feature budget:       {k.max().item():.3f}")

    print(
        f"Zero-budget features:     "
        f"{num_zero_budget:,} "
        f"({100 * fraction_zero_budget:.2f}%)"
    )

    print(
        f"Annotated features:       "
        f"{num_annotated:,} "
        f"({100 * fraction_annotated:.2f}%)"
    )

    print()
    print("Cosine:")
    print(f"  Active only:            " f"{results['mean_cosine_active']:.4f}")
    print(f"  All features:           " f"{results['mean_cosine_all']:.4f}")
    print(f"  Median active:          " f"{results['median_cosine_active']:.4f}")
    print(f"  Median all:             " f"{results['median_cosine_all']:.4f}")

    print()
    print("Precision @ k_i:")
    print(f"  Active only:            " f"{results['mean_precision_active']:.4f}")
    print(f"  All features:           " f"{results['mean_precision_all']:.4f}")

    print()
    print("Recall @ k_i:")
    print(f"  Active only:            " f"{results['mean_recall_active']:.4f}")
    print(f"  All features:           " f"{results['mean_recall_all']:.4f}")

    print()
    print("F1 @ k_i:")
    print(f"  Active only:            " f"{results['mean_f1_active']:.4f}")
    print(f"  All features:           " f"{results['mean_f1_all']:.4f}")

    # ------------------------------------------------------------------
    # Per-feature output
    # ------------------------------------------------------------------

    per_feature = []

    train_cpu = train_A.cpu()
    test_cpu = test_A.cpu()
    M_cpu = posthoc_M.cpu()
    k_cpu = k.cpu()
    cosine_cpu = cosine.cpu()
    cosine_all_cpu = cosine_all.cpu()
    precision_cpu = precision.cpu()
    recall_cpu = recall.cpu()
    f1_cpu = f1.cpu()

    for i in range(d):
        ki = int(k_cpu[i].item())

        if ki > 0:
            claimed_classes = torch.topk(
                M_cpu[i],
                k=min(ki, c),
            ).indices.tolist()

            empirical_classes = torch.topk(
                test_cpu[i],
                k=min(ki, c),
            ).indices.tolist()
        else:
            claimed_classes = []
            empirical_classes = []

        per_feature.append(
            {
                "feature": i,
                "budget": ki,
                # Ordinary cosine. Undefined for zero-budget features.
                "cosine": (
                    None if not torch.isfinite(cosine_cpu[i]) else float(cosine_cpu[i])
                ),
                # Coverage-aware cosine. Zero-budget features = 0.
                "cosine_all": (
                    None
                    if not torch.isfinite(cosine_all_cpu[i])
                    else float(cosine_all_cpu[i])
                ),
                # Active-only metrics. Undefined for k_i = 0.
                "precision": (
                    None if torch.isnan(precision_cpu[i]) else float(precision_cpu[i])
                ),
                "recall": (
                    None if torch.isnan(recall_cpu[i]) else float(recall_cpu[i])
                ),
                "f1": (None if torch.isnan(f1_cpu[i]) else float(f1_cpu[i])),
                # Explicit coverage-aware versions.
                # Zero-budget features = 0.
                "precision_all": float(
                    torch.nan_to_num(
                        precision_cpu[i],
                        nan=0.0,
                    )
                ),
                "recall_all": float(
                    torch.nan_to_num(
                        recall_cpu[i],
                        nan=0.0,
                    )
                ),
                "f1_all": float(
                    torch.nan_to_num(
                        f1_cpu[i],
                        nan=0.0,
                    )
                ),
                "claimed_classes": claimed_classes,
                "test_classes": empirical_classes,
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
        "--budget-mode",
        type=str,
        choices=["uniform", "estimated"],
        default="uniform",
        help=(
            "How to choose post-hoc per-feature class budgets. "
            "'uniform' gives every feature rho associations; "
            "'estimated' allocates exactly rho*d associations "
            "to the largest entries of A_train."
        ),
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
        "--max-train-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N training examples.",
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
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
    )


if __name__ == "__main__":
    cli()
