import argparse
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.core import Dictionary


@torch.no_grad()
def main(
    dictionary: Dictionary,
    dataset,
    batch_size=4096,
    num_workers=4,
    device="cuda",
    num_classes=1000,
):
    dictionary.to(device)
    dictionary.eval()

    # feature_class_mass[i, c] is the total activation mass
    # of feature i over examples belonging to class c.
    feature_class_mass = torch.zeros(
        dictionary.dict_size,
        num_classes,
        dtype=torch.float32,
        device=device,
    )

    # Total activation mass for each feature.
    feature_mass = torch.zeros(
        dictionary.dict_size,
        dtype=torch.float32,
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    for x, labels in tqdm(dataloader):
        x = x.to(device)
        labels = labels.to(device)

        # f: [batch, dict_size]
        f = dictionary.encode(x)

        # We use non-negative SAE activation magnitude as the
        # measure of feature usage.
        #
        # If f is guaranteed non-negative, this is equivalent
        # to f itself.
        feature_mass += f.sum(dim=0)

        # Convert labels to one-hot:
        # [batch] -> [batch, num_classes]
        labels_one_hot = torch.nn.functional.one_hot(
            labels,
            num_classes=num_classes,
        ).float()

        # Accumulate activation mass for each feature/class pair:
        #
        # f.T:             [dict_size, batch]
        # labels_one_hot:  [batch, num_classes]
        #
        # result:          [dict_size, num_classes]
        feature_class_mass += f.T @ labels_one_hot

    # --------------------------------------------------------------
    # Empirical class distribution for every feature
    # --------------------------------------------------------------

    eps = 1e-12

    class_distribution = feature_class_mass / feature_mass.unsqueeze(1).clamp_min(eps)

    # --------------------------------------------------------------
    # Feature class entropy
    # --------------------------------------------------------------

    # H_i = -sum_c q_ic log(q_ic)
    entropy = -(class_distribution * torch.log(class_distribution.clamp_min(eps))).sum(
        dim=1
    )

    # Normalized entropy lies in [0, 1].
    #
    # 0 = feature is associated with one class
    # 1 = feature is uniformly distributed across all classes
    normalized_entropy = entropy / torch.log(
        torch.tensor(
            float(num_classes),
            device=device,
        )
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    return {
        "entropy": entropy.cpu(),
        "normalized_entropy": normalized_entropy.cpu(),
        "class_distribution": class_distribution.cpu(),
        "feature_mass": feature_mass.cpu(),
    }


def cli():
    parser = argparse.ArgumentParser(
        description="Compute feature class entropy for a trained SAE."
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
        required=True,
        help="Path to the trained SAE checkpoint directory.",
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
        help="Path to save the entropy tensor (.pt).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda, cuda:0, or cpu.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes in the dataset.",
    )

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    dataset = ActivationsDataset(args.activations_path)

    if args.max_examples is not None:
        max_examples = min(
            args.max_examples,
            len(dataset),
        )

        dataset = torch.utils.data.Subset(
            dataset,
            range(max_examples),
        )

    # --------------------------------------------------------------
    # SAE
    # --------------------------------------------------------------

    sae = SUPPORTED_ARCHITECTURES[args.architecture].from_pretrained(
        args.checkpoint_path,
        device=device,
    )

    sae.eval()

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    results = main(
        dictionary=sae,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        num_classes=args.num_classes,
    )

    entropy = results["entropy"]
    normalized_entropy = results["normalized_entropy"]

    print("\n=== Feature Class Entropy ===")
    print(f"Number of features: {entropy.shape[0]}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Mean entropy: {entropy.mean().item():.6f}")
    print(f"Median entropy: {entropy.median().item():.6f}")
    print(f"Mean normalized entropy: " f"{normalized_entropy.mean().item():.6f}")
    print(f"Median normalized entropy: " f"{normalized_entropy.median().item():.6f}")

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    output_path = Path(args.output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        results,
        output_path,
    )

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    cli()
