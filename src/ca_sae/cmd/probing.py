import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.ca_sae import ClassAlignedSAE


def topk_sparse(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep the original activation values of the top-k coordinates
    in each row and set all other coordinates to zero.

    Args:
        x: [batch, d]
        k: number of activations to retain

    Returns:
        [batch, d] tensor with only top-k values retained.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    if k >= x.shape[1]:
        return x

    indices = torch.topk(x, k=k, dim=1).indices

    sparse = torch.zeros_like(x)
    sparse.scatter_(1, indices, x.gather(1, indices))

    return sparse


def load_examples(
    activations_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_examples: int | None,
    train_fraction: float,
    seed: int,
):
    """
    Load the activation dataset and create train/test splits.

    Returns:
        train_loader, test_loader, num_classes
    """

    dataset = ActivationsDataset(activations_path)

    if max_examples is not None:
        max_examples = min(max_examples, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_examples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two examples.")

    train_size = int(len(dataset) * train_fraction)
    test_size = len(dataset) - train_size

    if train_size == 0 or test_size == 0:
        raise ValueError(
            f"Invalid train fraction {train_fraction}; "
            f"dataset has {len(dataset)} examples."
        )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    return train_loader, test_loader


@torch.inference_mode()
def extract_features(
    model,
    loader,
    device: torch.device,
    probe_k: int | None,
):
    """
    Extract representations.

    If model is None:
        representation = raw activation

    Otherwise:
        representation = model.encode(x)

    If probe_k is provided, retain the original values of only
    the top-k representation coordinates.
    """

    all_features = []
    all_labels = []

    for x, y in tqdm(loader):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        if model is None:
            features = x
        else:
            features = model.encode(x)

        if probe_k is not None:
            if probe_k > features.shape[1]:
                raise ValueError(
                    f"--probe-k={probe_k} exceeds representation dimension "
                    f"{features.shape[1]}"
                )

            features = topk_sparse(features, probe_k)

        all_features.append(features.cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
):
    """
    Train a multinomial linear classifier.

    The representation is frozen; only the probe parameters are trained.
    """

    input_dim = train_features.shape[1]

    probe = torch.nn.Linear(input_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(
        train_features,
        train_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    probe.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0

        for features, labels in tqdm(loader, leave=False):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = probe(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            total_examples += labels.shape[0]

        mean_loss = total_loss / total_examples

        print(f"Epoch {epoch + 1:3d}/{epochs}: " f"loss={mean_loss:.6f}")

    return probe


@torch.inference_mode()
def evaluate_probe(
    probe,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    """
    Evaluate the trained linear probe.

    Reports:
      - top-1 accuracy
      - top-5 accuracy
      - top-10 accuracy
      - macro precision / recall / F1
      - weighted precision / recall / F1
      - mean true-class score
      - mean strongest-wrong-class score
      - mean classification margin
      - per-class precision / recall / F1
    """

    probe.eval()

    dataset = torch.utils.data.TensorDataset(
        test_features,
        test_labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    all_logits = []
    all_labels = []

    for features, labels in tqdm(loader):
        features = features.to(device, non_blocking=True)

        logits = probe(features)

        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    num_examples = labels.shape[0]
    num_classes = logits.shape[1]

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    predictions = logits.argmax(dim=1)

    accuracy = (predictions == labels).float().mean().item()

    top5_k = min(5, num_classes)
    top10_k = min(10, num_classes)

    top5_predictions = logits.topk(top5_k, dim=1).indices
    top10_predictions = logits.topk(top10_k, dim=1).indices

    top5_accuracy = (
        (top5_predictions == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    top10_accuracy = (
        (top10_predictions == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    for true, predicted in zip(labels, predictions):
        confusion[true, predicted] += 1

    support = confusion.sum(dim=1)
    predicted_count = confusion.sum(dim=0)
    true_positive = confusion.diag()

    precision = torch.zeros(num_classes, dtype=torch.float64)
    recall = torch.zeros(num_classes, dtype=torch.float64)
    f1 = torch.zeros(num_classes, dtype=torch.float64)

    valid_precision = predicted_count > 0
    valid_recall = support > 0

    precision[valid_precision] = (
        true_positive[valid_precision].double()
        / predicted_count[valid_precision].double()
    )

    recall[valid_recall] = (
        true_positive[valid_recall].double() / support[valid_recall].double()
    )

    valid_f1 = (precision + recall) > 0

    f1[valid_f1] = (
        2
        * precision[valid_f1]
        * recall[valid_f1]
        / (precision[valid_f1] + recall[valid_f1])
    )

    # ---------------------------------------------------------------
    # Macro / weighted metrics
    # ---------------------------------------------------------------

    macro_precision = precision[valid_precision].mean().item()
    macro_recall = recall[valid_recall].mean().item()
    macro_f1 = f1[valid_recall].mean().item()

    total_support = support.sum().double()

    weighted_precision = ((precision * support.double()).sum() / total_support).item()

    weighted_recall = ((recall * support.double()).sum() / total_support).item()

    weighted_f1 = ((f1 * support.double()).sum() / total_support).item()

    # ---------------------------------------------------------------
    # Classification margins
    # ---------------------------------------------------------------

    true_scores = logits[
        torch.arange(num_examples),
        labels,
    ]

    # Don't allow the true class to be the "wrong" class.
    wrong_logits = logits.clone()
    wrong_logits[
        torch.arange(num_examples),
        labels,
    ] = -torch.inf

    max_wrong_scores = wrong_logits.max(dim=1).values

    margins = true_scores - max_wrong_scores

    mean_true_class_score = true_scores.mean().item()
    mean_max_wrong_class_score = max_wrong_scores.mean().item()
    mean_classification_margin = margins.mean().item()

    # ---------------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------------

    classes = []

    for c in range(num_classes):
        classes.append(
            {
                "class": c,
                "precision": precision[c].item(),
                "recall": recall[c].item(),
                "f1": f1[c].item(),
                "support": int(support[c].item()),
                "predicted": int(predicted_count[c].item()),
            }
        )

    return {
        "num_examples": num_examples,
        "num_features": test_features.shape[1],
        "num_classes": num_classes,
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "mean_true_class_score": mean_true_class_score,
        "mean_max_wrong_class_score": mean_max_wrong_class_score,
        "mean_classification_margin": mean_classification_margin,
        "classes": classes,
    }


def main(
    activations_path: str,
    architecture: str | None = None,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    probe_k: int | None = None,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_examples: int | None = None,
    train_fraction: float = 0.8,
    epochs: int = 20,
    probe_batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
):
    """
    Train and evaluate a linear probe on either:

        1. raw activations, or
        2. SAE encoded features.

    If probe_k is provided, only the top-k activation values are retained
    while preserving their original magnitudes.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    model = None

    if checkpoint_path is not None and checkpoint_path.lower() != "none":
        print(f"Loading SAE from: {checkpoint_path}")

        model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
            checkpoint_path,
            device=device,
        )

        model.eval()

        print(f"SAE representation dimension: {model.dict_size}")

    else:
        print("No SAE checkpoint supplied.")
        print("Using raw activations directly.")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_loader, test_loader = load_examples(
        activations_path=activations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_examples=max_examples,
        train_fraction=train_fraction,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Extract representations
    # ------------------------------------------------------------------

    print("\nExtracting training representations...")

    train_features, train_labels = extract_features(
        model=model,
        loader=train_loader,
        device=device,
        probe_k=probe_k,
    )

    print("Extracting test representations...")

    test_features, test_labels = extract_features(
        model=model,
        loader=test_loader,
        device=device,
        probe_k=probe_k,
    )

    num_classes = (
        max(
            int(train_labels.max()),
            int(test_labels.max()),
        )
        + 1
    )

    print()
    print(f"Training examples: {len(train_labels):,}")
    print(f"Test examples:     {len(test_labels):,}")
    print(f"Representation dim:{train_features.shape[1]:,}")
    print(f"Number of classes:  {num_classes:,}")

    if probe_k is not None:
        print(f"Probe sparsity k:   {probe_k:,}")
    else:
        print("Probe sparsity k:   disabled")

    # ------------------------------------------------------------------
    # Train probe
    # ------------------------------------------------------------------

    print("\n=== Training linear probe ===")

    probe = train_linear_probe(
        train_features=train_features,
        train_labels=train_labels,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=probe_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    print("\n=== Evaluating linear probe ===")

    results = evaluate_probe(
        probe=probe,
        test_features=test_features,
        test_labels=test_labels,
        batch_size=probe_batch_size,
        device=device,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    results["representation"] = "raw_activations" if model is None else "sae_features"

    results["checkpoint_path"] = checkpoint_path

    results["probe_k"] = probe_k

    results["train_fraction"] = train_fraction
    results["train_examples"] = len(train_labels)
    results["test_examples"] = len(test_labels)

    results["epochs"] = epochs
    results["learning_rate"] = learning_rate
    results["weight_decay"] = weight_decay
    results["seed"] = seed

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print("\n=== Linear Probe Evaluation ===")

    if model is None:
        print("Representation:       Raw activations")
    else:
        print("Representation:       SAE features")

    if probe_k is None:
        print("Probe sparsity:        None")
    else:
        print(f"Probe sparsity:        Top-{probe_k}")

    print(f"Train examples:        {len(train_labels):,}")
    print(f"Test examples:         {len(test_labels):,}")
    print(f"Features:              {train_features.shape[1]:,}")
    print(f"Classes:               {num_classes:,}")
    print()

    print(f"Accuracy:              {results['accuracy']:.4f}")
    print(f"Top-5 accuracy:        {results['top5_accuracy']:.4f}")
    print(f"Top-10 accuracy:       {results['top10_accuracy']:.4f}")
    print()

    print(f"Macro precision:       {results['macro_precision']:.4f}")
    print(f"Macro recall:          {results['macro_recall']:.4f}")
    print(f"Macro F1:              {results['macro_f1']:.4f}")
    print()

    print(f"Weighted precision:    {results['weighted_precision']:.4f}")
    print(f"Weighted recall:       {results['weighted_recall']:.4f}")
    print(f"Weighted F1:           {results['weighted_f1']:.4f}")
    print()

    print(f"Mean true score:       {results['mean_true_class_score']:.4f}")
    print(f"Mean strongest wrong:  " f"{results['mean_max_wrong_class_score']:.4f}")
    print(f"Mean classification:   " f"{results['mean_classification_margin']:.4f}")

    # ------------------------------------------------------------------
    # Save
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
        description=("Train a linear probe on raw activations or SAE features.")
    )

    parser.add_argument(
        "--activations-path",
        type=str,
        required=True,
        help="Path to ActivationsDataset directory.",
    )

    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        default=None,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
        help="Architecture of the trained SAE.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help=(
            "Path to SAE checkpoint. If omitted or set to 'none', "
            "probe raw activations directly."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    parser.add_argument(
        "--probe-k",
        type=int,
        default=None,
        help=(
            "Keep only the top-k activated SAE features for probing, "
            "while preserving their original activation values. "
            "Only applies when using an SAE checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for feature extraction.",
    )

    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=4096,
        help="Batch size for probe training/evaluation.",
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
        help="cuda, cuda:0, cpu, etc.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally use only the first N examples after shuffling.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of examples used for probe training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of probe training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.probe_k is not None and args.probe_k <= 0:
        parser.error("--probe-k must be > 0")

    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")

    main(
        activations_path=args.activations_path,
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        probe_k=args.probe_k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        probe_batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
