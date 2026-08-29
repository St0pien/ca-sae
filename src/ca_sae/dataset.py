import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class ActivationsDataset(Dataset):
    def __init__(self, activations_path: str):
        super().__init__()
        base_path = Path(activations_path)
        config_path = base_path / "config.json"
        embeddings_path = base_path / "embeddings.npy"
        labels_path = base_path / "labels.npy"
        with open(config_path) as f:
            config = json.load(f)

        self.embeddings = np.memmap(
            embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(config["dataset_size"], config["embedding_size"]),
        )
        self.labels = np.memmap(
            labels_path,
            dtype=np.int64,
            mode="r",
            shape=(config["dataset_size"],),
        )

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.embeddings[index].copy()),
            int(self.labels[index]),
        )

    def __len__(self):
        return len(self.embeddings)
