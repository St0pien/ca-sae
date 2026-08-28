import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.batch_top_k import BatchTopKSAE
from ca_sae.sae.ca_sae import ClassAlignedSAE
from ca_sae.sae.core import Dictionary
from ca_sae.sae.softsae import SoftSAE


@torch.no_grad()
def main(
    dictionary: Dictionary,
    dataset,
    batch_size=4096,
    num_workers=4,
    device="cuda",
):
    dictionary.to(device)
    out = defaultdict(float)
    active_features = torch.zeros(
        dictionary.dict_size, dtype=torch.float32, device=device
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    for x, _ in tqdm(dataloader):
        x = x.to(device)
        x_hat, f = dictionary(x, output_features=True)
        l2_loss = torch.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()

        active_features += f.sum(dim=0)

        # cosine similarity between x and x_hat
        x_normed = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / torch.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (
            torch.linalg.norm(x_hat, dim=-1) / torch.linalg.norm(x, dim=-1)
        ).mean()

        # compute variance explained
        total_variance = torch.var(x, dim=0).sum()
        residual_variance = torch.var(x - x_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = torch.linalg.norm(x_hat, dim=-1, ord=2) ** 2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        out["l2_loss"] += l2_loss.item()
        out["l1_loss"] += l1_loss.item()
        out["l0"] += l0.item()
        out["frac_variance_explained"] += frac_variance_explained.item()
        out["cossim"] += cossim.item()
        out["l2_ratio"] += l2_ratio.item()
        out["relative_reconstruction_bias"] += relative_reconstruction_bias.item()

    out = {key: value / len(dataloader) for key, value in out.items()}
    frac_alive = (active_features != 0).float().sum() / dictionary.dict_size
    out["frac_alive"] = frac_alive.item()

    return out


def cli():
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction statistics of a ClassAlignedSAE."
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
        help="Path to the ClassAlignedSAE checkpoint directory.",
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
        default=None,
        help="Optional path to save results as JSON.",
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
        help="Device, e.g. cuda, cuda:0, or cpu. Defaults to CUDA if available.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N examples.",
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
        max_examples = min(args.max_examples, len(dataset))
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
    )

    print("\n=== SAE Evaluation ===")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    cli()
