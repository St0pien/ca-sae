import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset


class ActivationsDataset(TensorDataset):
    def __init__(self, activations_path: str):
        base_path = Path(activations_path)
        config_path = base_path / "config.json"
        embeddings_path = base_path / "embeddings.npy"
        labels_path = base_path / "labels.npy"
        with open(config_path) as f:
            config = json.load(f)

        embeddings = np.memmap(
            embeddings_path,
            dtype=np.float16,
            mode="r",
            shape=(config["dataset_size"], config["clip_embedding_size"]),
        )
        labels = np.memmap(
            labels_path,
            dtype=np.int64,
            mode="r",
            shape=(config["dataset_size"],),
        )

        super().__init__(
            torch.from_numpy(embeddings), torch.from_numpy(labels)
        )
