from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lapsum.topk import soft_topk

from ca_sae.sae.config import SAEConfig
from ca_sae.sae.core import (
    Dictionary,
    SAETrainer,
    geometric_median,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
    topk_per_row,
)


class SoftSAE(Dictionary, nn.Module):
    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        k: int,
        alpha: float,
        k_max: Optional[int] = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        if k_max is None:
            k_max = k * 2

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("k_max", torch.tensor(k_max, dtype=torch.int))
        self.register_buffer("norm_factor", torch.tensor(1.0))
        self.register_buffer(
            "shift_factor", torch.zeros(activation_dim, dtype=torch.float32)
        )

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        k_estimator_encoder = nn.Linear(activation_dim, dict_size)
        k_estimator_encoder.weight.data = self.encoder.weight.data.clone()
        k_estimator_encoder.bias.data.zero_()
        self.k_estimator = nn.Sequential(
            k_estimator_encoder, nn.ReLU(), nn.Linear(dict_size, 1), nn.ReLU()
        )

    def estimate_k(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.k_estimator((x - self.b_dec) / self.norm_factor).squeeze(-1)
        k_hat = logit * (self.dict_size)
        return torch.clamp(k_hat, min=1, max=self.dict_size)

    def encode(self, x: torch.Tensor, return_active: bool = False, use_hard_topk=True):
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))

        if use_hard_topk:
            with torch.no_grad():
                k_estimate = self.estimate_k(x).long()
                encoded_acts = topk_per_row(post_relu_feat_acts, k_estimate)
        else:
            k_estimate = self.estimate_k(x)
            weights = soft_topk(
                post_relu_feat_acts,
                k_estimate.view(k_estimate.shape[0], 1),
                self.alpha.clone(),
            )
            encoded_acts = post_relu_feat_acts * weights

        if return_active:
            return (
                encoded_acts,
                encoded_acts.sum(0) > 0,
                post_relu_feat_acts,
                k_estimate,
            )
        else:
            return encoded_acts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts = self.encode(x)
        x_hat = self.decode(encoded_acts)

        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale
        self.norm_factor.fill_(scale)

    @classmethod
    def from_pretrained(
        cls, path, k=None, alpha=None, device=None, **kwargs
    ) -> "SoftSAE":
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if k is None:
            k = state_dict["k"].item()
        elif "k" in state_dict and k != state_dict["k"].item():
            raise ValueError(f"k={k} != {state_dict['k'].item()}=state_dict['k']")

        if alpha is None:
            alpha = state_dict["alpha"].item()
        elif "alpha" in state_dict and alpha != state_dict["alpha"].item():
            raise ValueError(
                f"alpha={k} != {state_dict['alpha'].item()}=state_dict['alpha']"
            )

        autoencoder = cls(activation_dim, dict_size, k, alpha)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


@dataclass
class SoftSAEConfig(SAEConfig):
    k_loss_weight: float = 1.0
    k_loss_beta: float = 5.0
    soft_topk_alpha: float = 0.0001
    alpha_anneal_steps: Optional[int] = None
    hard_topk_steps: Optional[int] = None
    k_max: Optional[int] = None
    softplus_beta: float = 5.0


class SoftSAETrainer(SAETrainer):
    ae: SoftSAE

    def __init__(self, steps: int, cfg: SoftSAEConfig):
        super().__init__(steps, cfg)
        self.steps = steps
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.k = cfg.k
        self.k_max = cfg.k_max
        self.k_anneal_steps = cfg.k_anneal_steps
        self.k_loss_weight = cfg.k_loss_weight
        self.k_loss_beta = cfg.k_loss_beta
        self.soft_topk_alpha = cfg.soft_topk_alpha
        self.alpha_anneal_steps = cfg.alpha_anneal_steps
        self.hard_topk_steps = cfg.hard_topk_steps

        self.ae = SoftSAE(
            cfg.activation_dim,
            cfg.dict_size,
            cfg.k,
            1 if cfg.alpha_anneal_steps is not None else cfg.soft_topk_alpha,
            cfg.k_max,
        )

        if cfg.lr is not None:
            self.lr = cfg.lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = cfg.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = cfg.auxk_alpha
        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.top_k_aux = cfg.activation_dim // 2  # Heuristic from B.1 of the paper
        self.softplus_beta = cfg.softplus_beta
        self.num_tokens_since_fired = torch.zeros(cfg.dict_size, dtype=torch.long)
        self.logging_parameters = [
            "effective_l0",
            "dead_features",
            "pre_norm_auxk_loss",
            "avg_k",
            "min_k",
            "max_k",
            "k_loss",
            "ae_soft_topk_alpha",
            "use_hard_topk",
            "lr_log",
            "avg_enc_grad",
            "avg_mlp_grad",
        ]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.avg_k = -1
        self.min_k = -1
        self.max_k = -1
        self.k_loss = -1
        self.ae_soft_topk_alpha = 1
        self.use_hard_topk = 0
        self.avg_enc_grad = 0
        self.avg_mlp_grad = 0

        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, cfg.warmup_steps, decay_start=cfg.decay_start)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_fn
        )
        self.lr_log = self.scheduler.get_last_lr()[0]

    def to(self, *args, **kwargs):
        self.ae.to(*args, **kwargs)
        self.num_tokens_since_fired = self.num_tokens_since_fired.to(*args, **kwargs)

    def update_annealed_k(
        self, step: int, activation_dim: int, k_anneal_steps: Optional[int] = None
    ) -> None:
        """Update k buffer in-place with annealed value"""
        if k_anneal_steps is None or k_anneal_steps == 0:
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

    def update_annealed_alpha(
        self, step: int, alpha_anneal_steps: Optional[int] = None
    ):
        if alpha_anneal_steps is None or alpha_anneal_steps == 0:
            return

        assert (
            0 <= alpha_anneal_steps < self.steps
        ), "alpha_anneal_steps must be >= 0 and < steps."

        step = min(step, alpha_anneal_steps)
        ratio = step / alpha_anneal_steps
        annealed_value = (1 - ratio) + self.soft_topk_alpha * ratio
        self.ae.alpha.fill_(annealed_value)

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
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

    # def get_k_loss(self, estimated_k: torch.Tensor):
    #     return F.softplus(estimated_k.mean() - self.ae.k, beta=self.softplus_beta)

    def get_k_loss(self, estimated_k: torch.Tensor):
        return estimated_k.mean() / self.ae.dict_size

    def loss(self, x, step=None, logging=False):
        use_hard_topk = self.hard_topk_steps is not None and step > (
            self.steps - self.hard_topk_steps
        )

        f, active_indices_F, post_relu_acts, estimated_k = self.ae.encode(
            x, return_active=True, use_hard_topk=use_hard_topk
        )

        with torch.no_grad():
            f_hard = self.ae.encode(x, use_hard_topk=True)

        f_combined = f_hard + (f - f.detach())

        x_hat = self.ae.decode(f_combined)

        with torch.no_grad():
            x_hat_soft = self.ae.decode(f)
            print(torch.nn.functional.mse_loss(x_hat, x_hat_soft))

        e = x - x_hat

        self.effective_l0 = self.ae.k.item()

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        self.avg_k = estimated_k.mean(dtype=torch.float32)
        self.min_k = estimated_k.min()
        self.max_k = estimated_k.max()
        self.ae_soft_topk_alpha = self.ae.alpha.item()
        self.use_hard_topk = 1 if use_hard_topk else 0
        self.lr_log = self.scheduler.get_last_lr()[0]

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts)
        k_loss = self.get_k_loss(estimated_k) if not use_hard_topk else 0.0
        self.k_loss = k_loss
        loss = l2_loss + self.k_loss_weight * k_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f_hard,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x, _):
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        loss = self.loss(x, step=step)
        loss.backward()

        self.avg_enc_grad = (
            self.ae.encoder.weight.grad.mean().item()
            if self.ae.encoder.weight.grad is not None
            else 0
        )
        self.avg_mlp_grad = (
            self.ae.k_estimator[0].weight.grad.mean().item()
            if self.ae.k_estimator[0].weight.grad is not None
            else 0
        )

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.update_annealed_k(step, self.ae.activation_dim, self.k_anneal_steps)
        self.update_annealed_alpha(step, self.alpha_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
