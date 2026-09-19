from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from classae.sae.config import SAEConfig
from classae.sae.core import geometric_median

from .core import (
    Dictionary,
    SAETrainer,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)


class TopAFASAE(Dictionary, nn.Module):
    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor, return_active: bool = False):
        """
        Top-AFA activation (Algorithm 1 of the AFA paper): rather than a fixed k,
        adaptively activates the minimum number of features (sorted by their
        decoder-norm-scaled magnitude) such that the resulting sparse feature
        norm matches the AFA target ||x - b_dec||_2. No threshold buffer is
        needed (unlike BatchTopKSAE) since this is already input-adaptive at
        both train and inference time.
        """
        x_cent = x - self.b_dec
        post_relu_feat_acts_BF = F.relu(self.encoder(x_cent))

        afa_target = torch.norm(x_cent, p=2, dim=1, keepdim=True).pow(2)

        dec_norms_F = torch.norm(self.decoder.weight, p=2, dim=0)
        dec_scaled_acts_BF = (post_relu_feat_acts_BF * dec_norms_F).pow(2)

        sorted_indices_BF = torch.argsort(dec_scaled_acts_BF, dim=-1, descending=True)
        cumulative_fa_BF = torch.cumsum(
            torch.gather(dec_scaled_acts_BF, -1, sorted_indices_BF), dim=-1
        )
        cumulative_fa_BF[..., -1] = 1e8  # ensure final cumulative sum dominates

        afa_target = torch.sqrt(afa_target)
        cumulative_fa_BF = torch.sqrt(cumulative_fa_BF)

        k_B = torch.abs(cumulative_fa_BF - afa_target).argmin(dim=-1) + 1

        rank_BF = torch.arange(
            post_relu_feat_acts_BF.shape[1], device=x.device
        ).unsqueeze(0)
        mask_BF = torch.zeros_like(post_relu_feat_acts_BF, dtype=torch.bool).scatter_(
            dim=1,
            index=sorted_indices_BF,
            src=rank_BF < k_B.unsqueeze(1),
        )
        encoded_acts_BF = post_relu_feat_acts_BF * mask_BF

        if return_active:
            return (
                encoded_acts_BF,
                encoded_acts_BF.sum(0) > 0,
                post_relu_feat_acts_BF,
            )
        else:
            return encoded_acts_BF

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)

        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "TopAFASAE":
        state_dict = torch.load(f"{path}/ae.pt")
        dict_size, activation_dim = state_dict["encoder.weight"].shape

        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


@dataclass
class TopAFASAEConfig(SAEConfig):
    l1_coeff: float = 0.0
    afa_coeff: float = 1 / 16  # stable value reported across all layers in the paper
    aux_penalty: float = 1 / 32
    topk_aux: Optional[int] = None  # defaults to activation_dim // 2 if unset


class TopAFATrainer(SAETrainer):
    def __init__(self, steps: int, cfg: TopAFASAEConfig):
        super().__init__(steps, cfg)
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps
        self.l1_coeff = cfg.l1_coeff
        self.afa_coeff = cfg.afa_coeff
        self.aux_penalty = cfg.aux_penalty

        self.ae = TopAFASAE(cfg.activation_dim, cfg.dict_size)

        if cfg.lr is not None:
            self.lr = cfg.lr
        else:
            scale = cfg.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.topk_aux = cfg.topk_aux or cfg.activation_dim // 2
        # Tracked per-token (not per-batch), matching BatchTopKTrainer's convention.
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

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        """
        Note: unlike BatchTopKTrainer's aux loss, this is NOT normalized by
        residual variance — it's a direct penalized MSE, faithful to the
        original TopAFASAE.get_auxiliary_loss. self.aux_penalty is applied
        here (inside), not multiplied in externally.
        """
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if self.dead_features > 0:
            k_aux = min(self.topk_aux, self.dead_features)

            acts_topk_aux = torch.topk(
                post_relu_acts_BF[:, dead_features],
                k_aux,
                dim=-1,
            )
            acts_aux_BF = torch.zeros_like(post_relu_acts_BF[:, dead_features]).scatter(
                -1, acts_topk_aux.indices, acts_topk_aux.values
            )

            x_reconstruct_aux = acts_aux_BF @ self.ae.decoder.weight[:, dead_features].T
            l2_loss_aux = self.aux_penalty * (
                (x_reconstruct_aux.float() - residual_BD.float()).pow(2).mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux
            return l2_loss_aux
        else:
            self.pre_norm_auxk_loss = -1
            return torch.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def loss(self, x, step=None, logging=False):
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(x, return_active=True)
        x_hat = self.ae.decode(f)

        e = x - x_hat

        l0_norm = (f > 0).float().sum(dim=-1).mean()
        self.effective_l0 = l0_norm.item()

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).mean()
        l1_norm = f.float().abs().sum(-1).mean()
        l1_loss = self.l1_coeff * l1_norm
        afa_loss = self.afa_coeff * torch.mean(
            (torch.norm(f, p=2, dim=-1) - torch.norm(x, p=2, dim=-1)).pow(2)
        )
        aux_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)

        loss = l2_loss + l1_loss + afa_loss + aux_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "l1_norm": l1_norm.item(),
                    "afa_loss": afa_loss.item(),
                    "aux_loss": (
                        aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
                    ),
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

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
