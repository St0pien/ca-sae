from collections import namedtuple
from dataclasses import dataclass, field
from math import isclose
from typing import Optional

import torch
import torch.nn as nn

from classae.sae.config import SAEConfig
from classae.sae.core import geometric_median

from .core import (
    Dictionary,
    SAETrainer,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)


class MatryoshkaBatchTopKSAE(Dictionary, nn.Module):
    def __init__(
        self, activation_dim: int, dict_size: int, k: int, group_sizes: list[int]
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        assert sum(group_sizes) == dict_size, "group sizes must sum to dict_size"
        assert all(s > 0 for s in group_sizes), "all group sizes must be positive"

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))

        self.active_groups = len(group_sizes)
        group_indices = [0] + list(torch.cumsum(torch.tensor(group_sizes), dim=0))
        self.group_indices = group_indices

        self.register_buffer("group_sizes", torch.tensor(group_sizes))

        self.W_enc = nn.Parameter(torch.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(dict_size, activation_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.W_dec.data = set_decoder_norm_to_unit_norm(
            self.W_dec.data.T, activation_dim, dict_size
        ).T
        self.W_enc.data = self.W_dec.data.clone().T

    def encode(
        self, x: torch.Tensor, return_active: bool = False, use_threshold: bool = True
    ):
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if use_threshold:
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
        else:
            # Flatten and perform batch top-k
            flattened_acts = post_relu_feat_acts_BF.flatten()
            post_topk = flattened_acts.topk(self.k * x.size(0), sorted=False, dim=-1)

            encoded_acts_BF = (
                torch.zeros_like(post_relu_feat_acts_BF.flatten())
                .scatter_(-1, post_topk.indices, post_topk.values)
                .reshape(post_relu_feat_acts_BF.shape)
            )

        max_act_index = self.group_indices[self.active_groups]
        encoded_acts_BF[:, max_act_index:] = 0

        if return_active:
            return encoded_acts_BF, encoded_acts_BF.sum(0) > 0, post_relu_feat_acts_BF
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.b_enc.data *= scale
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(
        cls, path, k=None, device=None, **kwargs
    ) -> "MatryoshkaBatchTopKSAE":
        state_dict = torch.load(f"{path}/ae.pt")
        activation_dim, dict_size = state_dict["W_enc"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        group_sizes = state_dict["group_sizes"].tolist()

        autoencoder = cls(activation_dim, dict_size, k=k, group_sizes=group_sizes)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


@dataclass
class MatryoshkaBatchTopKSAEConfig(SAEConfig):
    k: int = 64
    group_fractions: list[float] = field(default_factory=list)
    group_weights: Optional[list[float]] = None


class MatryoshkaBatchTopKTrainer(SAETrainer):
    def __init__(self, steps: int, cfg: MatryoshkaBatchTopKSAEConfig):
        super().__init__(steps, cfg)
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.k = cfg.k
        self.threshold_beta = cfg.threshold_beta
        self.threshold_start_step = cfg.threshold_start_step
        self.k_anneal_steps = cfg.k_anneal_steps

        assert isclose(sum(cfg.group_fractions), 1.0), "group_fractions must sum to 1.0"
        # Calculate all groups except the last one
        group_sizes = [int(f * cfg.dict_size) for f in cfg.group_fractions[:-1]]
        # Put remainder in the last group
        group_sizes.append(cfg.dict_size - sum(group_sizes))

        group_weights = cfg.group_weights
        if group_weights is None:
            group_weights = [1.0 / len(group_sizes)] * len(group_sizes)

        assert len(group_sizes) == len(
            group_weights
        ), "group_sizes and group_weights must have the same length"

        self.group_fractions = cfg.group_fractions
        self.group_sizes = group_sizes
        self.group_weights = group_weights

        self.ae = MatryoshkaBatchTopKSAE(
            cfg.activation_dim, cfg.dict_size, cfg.k, group_sizes
        )

        if cfg.lr is not None:
            self.lr = cfg.lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = cfg.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = cfg.auxk_alpha
        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.topk_aux = cfg.activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = torch.zeros(
            cfg.dict_size,
            dtype=torch.long,
        )
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, cfg.warmup_steps, decay_start=cfg.decay_start)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )

    def to(self, *args, **kwargs):
        self.ae.to(*args, **kwargs)
        self.num_tokens_since_fired = self.num_tokens_since_fired.to(*args, **kwargs)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None:
            return

        assert (
            0 <= k_anneal_steps < self.steps
        ), "k_anneal_steps must be >= 0 and < steps."
        # self.k is the target k set for the trainer, not the dictionary's current k
        assert activation_dim > self.k, "activation_dim must be greater than k"

        step = min(step, k_anneal_steps)
        ratio = step / k_anneal_steps
        annealed_value = activation_dim * (1 - ratio) + self.k * ratio

        # Update in-place
        self.ae.k.fill_(int(annealed_value))

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.topk_aux, dead_features.sum())

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # We don't want to apply the bias
            x_reconstruct_aux = auxk_acts_BF @ self.ae.W_dec
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f: torch.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
            active = f[f > 0]

            if active.size(0) == 0:
                min_activation = torch.tensor([0.0], dtype=torch.float)
            else:
                min_activation = active.min().detach().to(dtype=torch.float32)

            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=False
        )
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step:
            self.update_threshold(f)

        x_reconstruct = torch.zeros_like(x) + self.ae.b_dec
        total_l2_loss = 0.0
        l2_losses = torch.tensor([], device=x.device)

        W_dec_chunks = torch.split(self.ae.W_dec, self.ae.group_sizes.tolist(), dim=0)
        f_chunks = torch.split(f, self.ae.group_sizes.tolist(), dim=1)

        for i in range(self.ae.active_groups):
            W_dec_slice = W_dec_chunks[i]
            acts_slice = f_chunks[i]
            x_reconstruct = x_reconstruct + acts_slice @ W_dec_slice

            l2_loss = (x - x_reconstruct).pow(2).sum(
                dim=-1
            ).mean() * self.group_weights[i]
            total_l2_loss += l2_loss
            l2_losses = torch.cat([l2_losses, l2_loss.unsqueeze(0)])

        min_l2_loss = l2_losses.min().item()
        max_l2_loss = l2_losses.max().item()
        mean_l2_loss = l2_losses.mean()

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        auxk_loss = self.get_auxiliary_loss(
            (x - x_reconstruct).detach(), post_relu_acts_BF
        )
        loss = mean_l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_reconstruct,
                f,
                {
                    "l2_loss": mean_l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "min_l2_loss": min_l2_loss,
                    "max_l2_loss": max_l2_loss,
                },
            )

    def update(self, step, x, _):
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T,
            self.ae.W_dec.grad.T,
            self.ae.activation_dim,
            self.ae.dict_size,
        ).T
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()
