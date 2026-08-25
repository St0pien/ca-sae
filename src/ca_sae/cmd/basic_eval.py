from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

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
    dataset = ActivationsDataset("activations/imagenet_test_hf")
    # sae = BatchTopKSAE.from_pretrained("checkpoints/test/BatchTopKSAESecond/ae.pt")
    # sae = BatchTopKSAE.from_pretrained("checkpoints/test/batch_topk_92/ae.pt")
    # sae = SoftSAE.from_pretrained("checkpoints/test/rework_v1_basic_l0/ae.pt")
    # sae = SoftSAE.from_pretrained("checkpoints/test/rework_v1_gradient_trick/ae.pt")
    # sae = ClassAlignedSAE.from_pretrained("checkpoints/test/ca_sae_v1/ae.pt", features_per_class = 5)
    sae = ClassAlignedSAE.from_pretrained("checkpoints/test/ca_sae_v2")
    # sae = BatchTopKSAE.from_pretrained("checkpoints/test/batchtopk_63")
    results = main(sae, dataset)

    print(results)


if __name__ == "__main__":
    cli()
