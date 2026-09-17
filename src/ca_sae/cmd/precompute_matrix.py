import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.eval.posthoc_M import compute_empirical_matrix
from ca_sae.sae.core import Dictionary


def load_model(
    checkpoint_path: str,
    architecture: str,
    device: torch.device,
) -> Dictionary:
    """
    Load an SAE checkpoint.

    Currently supports ClassAlignedSAE. Add other SAE classes here as
    needed when comparing architectures.
    """
    return SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path,
        device=device,
    )


def main(
    architecture: str,
    checkpoint_path: str,
    activations_path: str,
    output_path: str,
    num_classes: int = 1000,
    batch_size: int = 4096,
    num_workers: int = 4,
    device: str | None = None,
    max_examples: int | None = None,
):
    """
    Compute and save the empirical feature-class alignment matrix A.

    The saved tensor has shape:

        [num_features, num_classes]

    where

        A[i, c] = P(feature i is selected | class = c).
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    # ------------------------------------------------------------------
    # Load SAE
    # ------------------------------------------------------------------

    model = load_model(
        architecture=architecture,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Features: {model.dict_size:,}")

    # ------------------------------------------------------------------
    # Load activations
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

    print(f"Examples: {len(dataset):,}")

    # ------------------------------------------------------------------
    # Compute empirical A
    # ------------------------------------------------------------------

    empirical_A, class_counts, total_examples = compute_empirical_matrix(
        model=model,
        loader=loader,
        num_classes=num_classes,
        device=device,
    )

    # ------------------------------------------------------------------
    # Save A
    # ------------------------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save CPU tensor to avoid tying the output to CUDA.
    torch.save(
        empirical_A.cpu(),
        output_path,
    )

    # ------------------------------------------------------------------
    # Basic diagnostics
    # ------------------------------------------------------------------

    classes_with_examples = class_counts > 0

    print()
    print("=== Empirical Feature-Class Alignment ===")
    print(f"Examples processed:    {total_examples:,}")
    print(f"Features:              {empirical_A.shape[0]:,}")
    print(f"Classes:               {empirical_A.shape[1]:,}")
    print(
        f"Classes observed:      "
        f"{classes_with_examples.sum().item():,} / {len(class_counts):,}"
    )
    print(f"A dtype:               {empirical_A.dtype}")
    print(f"A shape:               {tuple(empirical_A.shape)}")
    print(f"A min:                 {empirical_A.min().item():.6f}")
    print(f"A max:                 {empirical_A.max().item():.6f}")
    print(f"A mean:                {empirical_A.mean().item():.6f}")
    print()
    print(f"Saved A to:            {output_path}")

    return empirical_A


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the empirical feature-class alignment matrix " "A for an SAE."
        )
    )

    parser.add_argument(
        "--architecture",
        "-a",
        type=str,
        required=True,
        choices=SUPPORTED_ARCHITECTURES.keys(),
        help="SAE architecture to use",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the SAE checkpoint.",
    )

    parser.add_argument(
        "--activations-path",
        type=str,
        required=True,
        help="Path to the ActivationsDataset directory.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where the A tensor will be saved.",
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
        "--max-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples.",
    )

    args = parser.parse_args()

    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        activations_path=args.activations_path,
        output_path=args.output_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    cli()
