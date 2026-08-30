import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import wandb
from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.batch_top_k import BatchTopKSAEConfig, BatchTopKTrainer
from ca_sae.sae.ca_sae import ClassAlignedSAEConfig, ClassAlignedSAETrainer
from ca_sae.sae.config import (
    AUTOCAST_DTYPE,
    SAEConfig,
    TrainConfig,
    WandbConfig,
    DataLoaderConfig,
)
from ca_sae.sae.core import SAETrainer
from ca_sae.sae.softsae import SoftSAEConfig, SoftSAETrainer


from torch.utils.data import Sampler


class ChunkBatchSampler(Sampler):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        chunk_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if chunk_size % batch_size != 0:
            raise ValueError("chunk_size must be divisible by batch_size")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.seed = seed
        self.drop_last = drop_last

        self.num_chunks = (dataset_size + chunk_size - 1) // chunk_size

        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        chunks = np.arange(self.num_chunks)
        rng.shuffle(chunks)

        for chunk in chunks:
            start = chunk * self.chunk_size
            end = min(
                start + self.chunk_size,
                self.dataset_size,
            )

            # Generate batches sequentially.
            for batch_start in range(
                start,
                end,
                self.batch_size,
            ):
                batch_end = min(
                    batch_start + self.batch_size,
                    end,
                )

                if self.drop_last and batch_end - batch_start < self.batch_size:
                    continue

                yield list(range(batch_start, batch_end))

    def __len__(self):
        full_batches = self.dataset_size // self.batch_size

        if self.drop_last:
            return full_batches

        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def make_sae_trainer(steps: int, cfg: SAEConfig) -> SAETrainer:
    if isinstance(cfg, BatchTopKSAEConfig):
        return BatchTopKTrainer(steps, cfg)
    elif isinstance(cfg, SoftSAEConfig):
        return SoftSAETrainer(steps, cfg)
    elif isinstance(cfg, ClassAlignedSAEConfig):
        return ClassAlignedSAETrainer(steps, cfg)
    else:
        raise ValueError(f"Unkown sae config: {cfg.__class__.__name__}")


def get_norm_factor(data, num_batches: int = 100) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147

    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for batch in tqdm(
        data, total=min(len(data), num_batches), desc="Calculating norm factor"
    ):
        act_BD = batch[0]

        count += 1
        mean_squared_norm = torch.mean(torch.sum(act_BD**2, dim=1))
        total_mean_squared_norm += mean_squared_norm

        if count >= num_batches:
            break

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = torch.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")

    return norm_factor


def get_stats(trainer: SAETrainer, step: int, act: torch.Tensor, labels: torch.Tensor):
    with torch.no_grad():
        x = act.clone()
        y = labels.clone()
        log = {}
        if isinstance(trainer, ClassAlignedSAETrainer):
            x, x_hat, f, losslog = trainer.loss(x, y, step=step, logging=True)
        else:
            x, x_hat, f, losslog = trainer.loss(x, step=step, logging=True)

        l0 = (f != 0).float().sum(dim=-1).mean().item()
        total_variance = torch.var(x, dim=0).sum()
        residual_variance = torch.var(x - x_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        log["frac_variance_explained"] = frac_variance_explained.item()

        log.update(
            {
                f"{k}": v.cpu().item() if isinstance(v, torch.Tensor) else v
                for k, v in losslog.items()
            }
        )

        log[f"l0"] = l0
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            log[f"{name}"] = value
    return log


def main(cfg: TrainConfig):
    autocast_context = (
        nullcontext()
        if cfg.device == "cpu"
        else torch.autocast(device_type=cfg.device, dtype=AUTOCAST_DTYPE)
    )

    if cfg.save_dir is not None:
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True)
        with open(save_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    wandb_ctx = wandb.init(
        entity=cfg.wandb.entity if cfg.wandb is not None else "",
        project=cfg.wandb.project if cfg.wandb is not None else "",
        name=cfg.wandb.name if cfg.wandb is not None else "",
        config=asdict(cfg),
        mode="disabled" if cfg.wandb is None else "online",
    )

    dataset = ActivationsDataset(cfg.activations_path)
    batch_sampler = ChunkBatchSampler(
        dataset_size=len(dataset),
        batch_size=cfg.dataloader.batch_size,
        chunk_size=262_144,
        seed=42,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )

    trainer = make_sae_trainer(cfg.epochs * len(dataloader), cfg.sae)
    trainer.to(cfg.device)

    with wandb_ctx as wandb_run:
        if cfg.normalize_activations:
            norm_factor = get_norm_factor(dataloader)
            wandb_run.config.update({"norm_factor": norm_factor})

        step = 0
        for epoch in tqdm(range(cfg.epochs), position=0):
            for activations, labels in tqdm(dataloader, position=1, leave=False):
                activations = activations.to(device=cfg.device)
                labels = labels.to(device=cfg.device)
                if cfg.normalize_activations:
                    activations /= norm_factor

                log_obj = get_stats(trainer, step, activations, labels)
                wandb_run.log(log_obj)

                with autocast_context:
                    trainer.update(step, activations, labels)

                step += 1

    if cfg.normalize_activations:
        trainer.ae.scale_biases(norm_factor)
    if cfg.save_dir is not None:
        final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
        torch.save(final, save_dir / "ae.pt")


if __name__ == "__main__":
    main(
        TrainConfig(
            activations_path="activations/imagenet_val_dino",
            # sae=ClassAlignedSAEConfig(
            #     512,
            #     4096,
            #     64,
            #     lr=6e-4,
            #     soft_topk_alpha=0.001,
            #     # hard_topk_steps=2000,
            #     # alpha_anneal_steps=1000,
            #     decay_start=5000,
            #     agreement_loss_weight=0.15,
            # ),
            sae=BatchTopKSAEConfig(
                768,
                4096,
                63,
                lr=6e-4,
                decay_start=5000,
            ),
            epochs=30,
            save_dir="checkpoints/test/asdf",
            # wandb=WandbConfig(
            #     entity="st0pien-default-team",
            #     project="CASAE",
            #     name="ca_sae_v2",
            # ),
            # dataloader=DataLoaderConfig(num_workers=4, prefetch_factor=3),
        )
    )
