from dataclasses import dataclass, field
from typing import Optional

import torch

AUTOCAST_DTYPE = torch.float32


@dataclass
class WandbConfig:
    entity: str
    project: str
    name: str


@dataclass
class SAEConfig:
    activation_dim: int
    dict_size: int
    k: int
    lr: Optional[float] = None
    auxk_alpha: float = 1 / 32
    warmup_steps: int = 1000
    decay_start: Optional[int] = None
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    k_anneal_steps: Optional[int] = None
    dead_feature_threshold: int = 2_000_000


@dataclass
class DataLoaderConfig:
    batch_size: int = 4096
    num_workers: int = 4
    prefetch_factor: int = 2
    shuffle: bool = True


@dataclass
class TrainConfig:
    activations_path: str
    sae: SAEConfig
    epochs: int
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    device: str = "cuda"
    wandb: Optional[WandbConfig] = None
    save_dir: Optional[str] = None
    normalize_activations: bool = True
