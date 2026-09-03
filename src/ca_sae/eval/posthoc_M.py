from lapsum.topk import soft_topk
import torch
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
from tqdm import tqdm

from ca_sae.sae.core import Dictionary


def compute_empirical_matrix(
    model: Dictionary,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    d = model.dict_size
    c = num_classes

    feature_class_counts = torch.zeros(
        (d, c),
        dtype=torch.float64,
        device=device,
    )

    class_counts = torch.zeros(
        c,
        dtype=torch.float64,
        device=device,
    )

    total_examples = 0

    with torch.inference_mode():
        for x, labels in tqdm(loader):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Actual inference-time selection.
            features = model.encode(x)

            # Binary selected-feature mask.
            selected = features > 0

            # Count examples per class.
            class_counts += torch.bincount(
                labels,
                minlength=c,
            ).to(torch.float64)

            # Count feature selections for each class.
            for class_id in labels.unique():
                class_mask = labels == class_id
                class_id = int(class_id)

                feature_class_counts[:, class_id] += (
                    selected[class_mask].sum(dim=0).to(torch.float64)
                )

            total_examples += x.shape[0]

    # A[i, c] = P(feature i selected | class c)
    empirical_A = feature_class_counts / class_counts.clamp_min(1.0).unsqueeze(0)

    return empirical_A, class_counts, total_examples


def build_posthoc_M(
    empirical_A: torch.Tensor, rho: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    dict_size = empirical_A.shape[0]

    empirical_k = empirical_A.sum(dim=1)

    tau = empirical_k.mean() / dict_size

    empirical_k = (empirical_A * (empirical_A > tau)).sum(dim=1) / empirical_A.sum()
    # empirical_k = empirical_k / empirical_A.sum()

    # empirical_A = soft_topk(
    #     empirical_A, (empirical_k * dict_size * rho).unsqueeze(1), 0.001, dim=1
    # )

    return empirical_A, empirical_k * dict_size * rho
