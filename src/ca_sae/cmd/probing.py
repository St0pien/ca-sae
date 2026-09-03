import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset


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
    seed: int,
):
    """
    Load the activation dataset and create a DataLoader.

    Returns:
        loader
    """

    dataset = ActivationsDataset(activations_path)

    if max_examples is not None:
        max_examples = min(max_examples, len(dataset))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_examples]
        dataset = torch.utils.data.Subset(dataset, indices.tolist())

    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two examples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    return loader


@torch.inference_mode()
def extract_features(
    model,
    loader,
    device: torch.device,
    probe_k: int | None,
):
    """
    Extract representations, fully materialized in memory.

    If model is None:
        representation = raw activation

    Otherwise:
        representation = model.encode(x)

    If probe_k is provided, retain the original values of only
    the top-k representation coordinates.

    NOTE: intended for the (smaller) test/held-out split only.
    For large training sets, use the streamed path in
    train_linear_probe instead, to avoid materializing the full
    encoded representation in RAM.
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


def infer_input_dim(model, loader: DataLoader) -> int:
    """
    Determine the probe's input dimension without materializing
    the full dataset. Uses the model's dict_size when available,
    otherwise peeks at a single batch's raw activation width.

    Note: when model is None, this spins up a separate iterator
    over loader to peek one batch. This does not consume or skip
    batches used by the actual training loop, since DataLoader
    iterators are independent, but it does briefly spawn a worker
    pool if num_workers > 0.
    """
    if model is not None:
        return model.dict_size

    sample_x, _ = next(iter(loader))
    return sample_x.shape[1]


def train_linear_probe(
    model,
    train_loader: DataLoader,
    num_classes: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    probe_k: int | None,
    input_dim: int,
):
    """
    Train a multinomial linear classifier.

    The representation is frozen; only the probe parameters are trained.

    Streams batches directly from train_loader through the (frozen)
    model encoder, rather than pre-extracting and holding the full
    training representation in memory. This keeps peak memory usage
    bounded by a single batch, which matters at ImageNet scale where
    materializing the full encoded train set (e.g. ~1.28M x 4096
    floats) can exceed available RAM.
    """

    probe = torch.nn.Linear(input_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss()

    probe.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_examples = 0

        for x, y in tqdm(train_loader, leave=False):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            # Encoding is frozen -- no gradient needed through the SAE
            # itself. Use no_grad (not inference_mode) so the resulting
            # tensor remains a normal leaf that the probe's autograd
            # graph can build on top of.
            with torch.no_grad():
                if model is None:
                    features = x
                else:
                    features = model.encode(x)

                if probe_k is not None:
                    if probe_k > features.shape[1]:
                        raise ValueError(
                            f"--probe-k={probe_k} exceeds representation "
                            f"dimension {features.shape[1]}"
                        )
                    features = topk_sparse(features, probe_k)

            optimizer.zero_grad(set_to_none=True)

            logits = probe(features)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.shape[0]
            total_examples += y.shape[0]

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
    train_activations_path: str,
    test_activations_path: str,
    architecture: str | None = None,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    probe_k: int | None = None,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_examples: int | None = None,
    epochs: int = 20,
    probe_batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    num_classes: int | None = None,
):
    """
    Train and evaluate a linear probe on either:

        1. raw activations, or
        2. SAE encoded features.

    If probe_k is provided, only the top-k activation values are retained
    while preserving their original magnitudes.

    The training split is streamed batch-by-batch directly into the
    probe's training loop (never materialized in full), while the
    test split is still fully extracted up front since it is small
    enough to fit comfortably in memory.
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
    #
    # Train loader is consumed batch-by-batch directly by the probe
    # training loop, so it uses probe_batch_size (the size the probe
    # actually trains on) rather than the feature-extraction batch_size.
    # ------------------------------------------------------------------

    train_loader = load_examples(
        activations_path=train_activations_path,
        batch_size=probe_batch_size,
        num_workers=num_workers,
        device=device,
        max_examples=max_examples,
        seed=seed,
    )

    test_loader = load_examples(
        activations_path=test_activations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        max_examples=max_examples,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Extract test representations (kept in memory, confirmed to fit)
    # ------------------------------------------------------------------

    print("Extracting test representations...")

    test_features, test_labels = extract_features(
        model=model,
        loader=test_loader,
        device=device,
        probe_k=probe_k,
    )

    # num_classes can no longer be inferred from train_labels, since
    # the training set is never materialized. Infer from the test
    # split, or take an explicit override if the train split might
    # contain classes absent from test.
    inferred_num_classes = int(test_labels.max()) + 1
    resolved_num_classes = (
        num_classes if num_classes is not None else inferred_num_classes
    )

    input_dim = infer_input_dim(model, train_loader)

    print()
    print(f"Test examples:      {len(test_labels):,}")
    print(f"Representation dim: {input_dim:,}")
    print(f"Number of classes:  {resolved_num_classes:,}")

    if probe_k is not None:
        print(f"Probe sparsity k:   {probe_k:,}")
    else:
        print("Probe sparsity k:   disabled")

    # ------------------------------------------------------------------
    # Train probe (streamed -- no full train_features/train_labels
    # tensor is ever materialized)
    # ------------------------------------------------------------------

    print("\n=== Training linear probe ===")

    probe = train_linear_probe(
        model=model,
        train_loader=train_loader,
        num_classes=resolved_num_classes,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        probe_k=probe_k,
        input_dim=input_dim,
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

    print(f"Test examples:         {len(test_labels):,}")
    print(f"Features:              {input_dim:,}")
    print(f"Classes:               {resolved_num_classes:,}")
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
        "--train-activations-path",
        type=str,
        required=True,
        help="Path to ActivationsDataset directory.",
    )

    parser.add_argument(
        "--test-activations-path",
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
        help="Batch size for test-set feature extraction.",
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

    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Optional explicit number of classes. If omitted, inferred "
            "from the test split's max label + 1. Set explicitly if the "
            "training split may contain classes absent from the test split."
        ),
    )

    args = parser.parse_args()

    if args.probe_k is not None and args.probe_k <= 0:
        parser.error("--probe-k must be > 0")

    main(
        train_activations_path=args.train_activations_path,
        test_activations_path=args.test_activations_path,
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        probe_k=args.probe_k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
        epochs=args.epochs,
        probe_batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_classes=args.num_classes,
    )


if __name__ == "__main__":
    cli()
