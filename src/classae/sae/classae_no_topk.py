"""
Ablation variant of ClassAlignedSAE (CA-SAE).

This drops the adaptive top-k feature-selection machinery (the k-estimator
network, hard/soft top-k over features) and replaces it with a standard
ReLU SAE trained with an L1 sparsity penalty -- the usual differentiable
relaxation of L0 used to train "vanilla" SAEs.

The class-feature affinity mechanism (class_matrix, budget_vector,
calculate_M(), and the contrastive agreement loss built on top of it) is
kept intact and unchanged in spirit, so this isolates the effect of
top-k-style hard/soft feature selection vs. plain L1-regularized ReLU
activations, while both variants still learn the same kind of
class <-> feature association structure.

Everything else (decoder unit-norm constraint, geometric-median init,
dead-feature auxiliary loss, gradient projection, LR schedule) mirrors
CA-SAE so the two are comparable apples-to-apples in an ablation.
"""

import json
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lapsum.topk import soft_topk

from classae.sae.config import SAEConfig
from classae.sae.core import (
    Dictionary,
    SAETrainer,
    geometric_median,
    get_lr_schedule,
    remove_gradient_parallel_to_decoder_directions,
    set_decoder_norm_to_unit_norm,
)


class ClasSAE_NO_TOPK(Dictionary, nn.Module):
    """Standard (ReLU + L1) SAE with a class-feature affinity matrix.

    Architecturally identical to ClassAlignedSAE minus the k-estimator and
    minus any topk / soft-topk selection in `encode`. `calculate_M` is kept
    verbatim: it still produces a [dict_size, num_classes] soft assignment
    matrix used to guide features towards classes via the agreement loss
    computed in the trainer.
    """

    def __init__(
        self,
        activation_dim: int,
        dict_size: int,
        num_classes: int,
        rho: int,
        alpha: float,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_classes = num_classes
        self.rho = rho

        # alpha here only controls the temperature of the soft_topk used
        # inside calculate_M() (class <-> feature affinity), not any
        # feature-selection step -- there is no feature-level topk in this
        # variant at all.
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.T.clone()
        self.encoder.bias.data.zero_()
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

        # Same affinity mechanism as CA-SAE, unchanged.
        self.class_matrix = nn.Parameter(torch.randn(dict_size, num_classes) * 0.01)
        self.budget_vector = nn.Parameter(
            torch.zeros((dict_size,), dtype=torch.float32)
        )

    def calculate_M(self):
        """Unchanged from ClassAlignedSAE: per-feature class-affinity budget
        via soft_topk over the class dimension."""
        Ktot = float(self.rho * self.dict_size)

        k = Ktot * torch.softmax(self.budget_vector, dim=0)
        k = torch.clamp(k, 1.0, float(self.num_classes))

        M = soft_topk(self.class_matrix, k.unsqueeze(-1), self.alpha.clone(), dim=1)

        return M

    def encode(self, x: torch.Tensor, return_active: bool = False):
        """Plain ReLU encoding -- no top-k / soft-top-k feature selection.
        Sparsity is enforced purely via the L1 penalty in the trainer."""
        post_relu_feat_acts = F.relu(self.encoder(x - self.b_dec))
        encoded_acts = post_relu_feat_acts

        if return_active:
            return (
                encoded_acts,
                encoded_acts.sum(0) > 0,
                post_relu_feat_acts,
            )
        else:
            return encoded_acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f) + self.b_dec

    def forward(self, x: torch.Tensor, output_features: bool = True):
        encoded_acts = self.encode(x)
        x_hat = self.decode(encoded_acts)

        if not output_features:
            return x_hat
        else:
            return x_hat, encoded_acts

    def scale_biases(self, scale: float):
        self.encoder.bias.data *= scale
        self.b_dec.data *= scale

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "ClasSAE_NO_TOPK":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(
            f"{path}/ae.pt",
            map_location=device,
            weights_only=True,
        )

        # Handle checkpoints saved from DataParallel / DDP.
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :]: value for key, value in state_dict.items()
            }

        # Infer dimensions from the state dict.
        decoder_weight = state_dict["decoder.weight"]
        activation_dim, dict_size = decoder_weight.shape

        class_matrix = state_dict["class_matrix"]
        class_matrix_dict_size, num_classes = class_matrix.shape

        if class_matrix_dict_size != dict_size:
            raise ValueError(
                f"Inconsistent dict_size: decoder.weight has {dict_size}, "
                f"but class_matrix has {class_matrix_dict_size}"
            )

        alpha = state_dict["alpha"].item()

        with open(f"{path}/config.json") as f_config:
            json_config = json.load(f_config)
            rho = json_config["sae"]["rho"]

        model = cls(
            activation_dim=activation_dim,
            dict_size=dict_size,
            num_classes=num_classes,
            rho=rho,
            alpha=alpha,
        )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model


@dataclass
class ClasSAE_NO_TOPK_Config(SAEConfig):
    l1_penalty: float = 1e-3
    l1_anneal_steps: Optional[int] = None
    soft_topk_alpha: float = 0.001
    alpha_anneal_steps: Optional[int] = None
    num_classes: int = 1000
    rho: int = 5
    agreement_loss_weight: float = 1.0
    agreement_tau: float = 1.0
    tau_anneal_start: float = 50.0
    tau_anneal_steps: Optional[int] = None


class ClasSAE_NO_TOPK_Trainer(SAETrainer):
    ae: ClasSAE_NO_TOPK

    def __init__(self, steps, cfg: ClasSAE_NO_TOPK_Config):
        super().__init__(steps, cfg)
        self.steps = steps
        self.decay_start = cfg.decay_start
        self.warmup_steps = cfg.warmup_steps

        self.l1_penalty = cfg.l1_penalty
        self.l1_anneal_steps = cfg.l1_anneal_steps

        self.agreement_loss_weight = cfg.agreement_loss_weight
        self.soft_topk_alpha = cfg.soft_topk_alpha
        self.alpha_anneal_steps = cfg.alpha_anneal_steps

        self.agreement_tau = cfg.agreement_tau
        self.tau_anneal_start = cfg.tau_anneal_start
        self.tau_anneal_steps = cfg.tau_anneal_steps

        self.active_tau = cfg.agreement_tau
        self.active_l1_penalty = 0.0 if cfg.l1_anneal_steps else cfg.l1_penalty

        self.ae = ClasSAE_NO_TOPK(
            cfg.activation_dim,
            cfg.dict_size,
            cfg.num_classes,
            cfg.rho,
            cfg.soft_topk_alpha,
        )

        if cfg.lr is not None:
            self.lr = cfg.lr
        else:
            # Same 1 / sqrt(d) scaling law used for CA-SAE.
            scale = cfg.dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = cfg.auxk_alpha
        self.dead_feature_threshold = cfg.dead_feature_threshold
        self.topk_aux = cfg.activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = torch.zeros(cfg.dict_size, dtype=torch.long)

        ### LOGGING SETUP
        self.logging_parameters = [
            "dead_features",
            "pre_norm_auxk_loss",
            "avg_l0",
            "min_l0",
            "max_l0",
            "l1_loss",
            "active_l1_penalty",
            "agreement_loss",
            "ae_soft_topk_alpha",
            "active_tau",
            "lr_log",
            "avg_enc_grad",
        ]
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1
        self.avg_l0 = -1
        self.min_l0 = -1
        self.max_l0 = -1
        self.l1_loss = -1
        self.agreement_loss = -1
        self.ae_soft_topk_alpha = 1
        self.avg_enc_grad = 0

        ### LOGGING SETUP

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

    def update_annealed_l1(self, step: int, l1_anneal_steps: Optional[int] = None):
        """Warm the L1 coefficient up from 0 to its target value, mirroring
        CA-SAE's k/alpha annealing. Standard trick to avoid instantly
        collapsing most features to zero early in training."""
        if l1_anneal_steps is None or l1_anneal_steps == 0:
            self.active_l1_penalty = self.l1_penalty
            return

        assert (
            0 <= l1_anneal_steps < self.steps
        ), "l1_anneal_steps must be >= 0 and < steps."

        step = min(step, l1_anneal_steps)
        ratio = step / l1_anneal_steps
        self.active_l1_penalty = self.l1_penalty * ratio

    def update_annealed_alpha(
        self, step: int, alpha_anneal_steps: Optional[int] = None
    ):
        """Unchanged from CA-SAE: anneals the temperature used inside
        calculate_M() (class <-> feature affinity), independent of L1."""
        if alpha_anneal_steps is None or alpha_anneal_steps == 0:
            return

        assert (
            0 <= alpha_anneal_steps < self.steps
        ), "alpha_anneal_steps must be >= 0 and < steps."

        step = min(step, alpha_anneal_steps)
        ratio = step / alpha_anneal_steps
        annealed_value = (1 - ratio) + self.soft_topk_alpha * ratio
        self.ae.alpha.fill_(annealed_value)

    def update_annealed_tau(self, step: int, tau_anneal_steps: Optional[int] = None):
        if tau_anneal_steps is None or tau_anneal_steps == 0:
            return

        assert (
            0 <= tau_anneal_steps < self.steps
        ), "tau_anneal_steps must be >= 0 and < steps."

        step = min(step, tau_anneal_steps)
        ratio = step / tau_anneal_steps
        annealed_value = (
            self.tau_anneal_start * (1 - ratio) + self.agreement_tau * ratio
        )
        self.active_tau = annealed_value

    def get_auxiliary_loss(
        self, residual_BD: torch.Tensor, post_relu_acts_BF: torch.Tensor
    ):
        """Unchanged from CA-SAE. Dead-feature revival via reconstructing
        the residual from the top activations among dead features -- this
        is independent of whether the main encode path uses topk or not."""
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.topk_aux, dead_features.sum())

            auxk_latents = torch.where(
                dead_features[None], post_relu_acts_BF, -torch.inf
            )

            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            auxk_buffer_BF = torch.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=auxk_acts
            )

            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

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

    def get_l1_loss(self, f: torch.Tensor):
        """L1 penalty over feature activations -- the differentiable
        relaxation of L0 sparsity, replacing CA-SAE's k_loss entirely."""
        return f.norm(p=1, dim=-1).mean()

    def get_agreement_loss(self, acts: torch.Tensor, labels: torch.Tensor):
        """Same contrastive form as CA-SAE's agreement loss, but the
        normalizer is now the realized L0 norm (count of active features
        per example, detached) rather than a predicted k_hat, since there
        is no k-estimator in this variant."""
        l0 = (acts > 0).sum(dim=-1).clamp(min=1).to(acts.dtype)  # [B]
        p = soft_topk(acts, l0.unsqueeze(1), self.ae.alpha.clone())
        pi = p / l0.unsqueeze(-1).detach()  # [B, d]

        M = self.ae.calculate_M()  # [d, C]

        s = pi @ M  # [B, C]

        tau = self.active_tau

        s_true = s.gather(1, labels.unsqueeze(-1)).squeeze(-1)  # [B]
        log_denom = torch.logsumexp(s / tau, dim=1)  # [B]

        loss = -s_true / tau + log_denom

        return loss.mean()

    def loss(self, x, y, step=None, logging=False):
        f, active_indices_F, post_relu_acts = self.ae.encode(x, return_active=True)

        x_hat = self.ae.decode(f)

        e = x - x_hat

        num_tokens_in_step = x.size(0)
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l0_per_example = (f > 0).sum(dim=-1)
        self.avg_l0 = l0_per_example.float().mean()
        self.min_l0 = l0_per_example.min()
        self.max_l0 = l0_per_example.max()
        self.ae_soft_topk_alpha = self.ae.alpha.clone()
        self.lr_log = self.scheduler.get_last_lr()[0]

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts)
        l1_loss = self.get_l1_loss(f)
        self.l1_loss = l1_loss

        agreement_loss = self.get_agreement_loss(post_relu_acts, y)
        self.agreement_loss = agreement_loss

        loss = (
            l2_loss
            + self.active_l1_penalty * l1_loss
            + self.auxk_alpha * auxk_loss
            + self.agreement_loss_weight * agreement_loss
        )

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x, y):
        if step == 0:
            median = geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        loss = self.loss(x, y, step=step)
        loss.backward()

        self.avg_enc_grad = (
            self.ae.encoder.weight.grad.mean().item()
            if self.ae.encoder.weight.grad is not None
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
        self.update_annealed_l1(step, self.l1_anneal_steps)
        self.update_annealed_alpha(step, self.alpha_anneal_steps)
        self.update_annealed_tau(step, self.tau_anneal_steps)

        # Make sure the decoder is still unit-norm
        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()
