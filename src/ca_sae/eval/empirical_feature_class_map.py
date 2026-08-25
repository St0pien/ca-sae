import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from ca_sae.sae.core import Dictionary


def compute_empirical_feature_class_map(
    model: Dictionary,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Compute the empirical feature-class selection map.

    For each dictionary feature i and class c:

        A[i, c] = P(feature i is selected | class = c)

    using the model's actual hard Top-k inference-time selection.

    Returns:
        empirical_A:
            Tensor of shape [dict_size, num_classes].
        class_counts:
            Tensor of shape [num_classes], containing the number of
            examples observed for each class.
        total_examples:
            Total number of examples processed.
    """
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
