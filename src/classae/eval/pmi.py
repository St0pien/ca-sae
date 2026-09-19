"""
Shared feature-class pointwise mutual information (PMI) machinery.

Motivation
----------
A feature-class table entry like `M[i, c]` or an empirical conditional
`A[i, c] = P(fire_i | c)` only tells you how often feature i fires when
class c is present. It cannot distinguish:

  - a feature that fires on 90% of class-c images AND on 90% of every
    other class's images (uninformative -- it's just "usually on"), from
  - a feature that fires on 90% of class-c images and on 5% of every
    other class (highly informative about c specifically).

Pointwise mutual information fixes this by comparing the conditional
firing rate against the feature's *marginal* firing rate:

    PMI(i, c) = log( P(fire_i | c) / P(fire_i) )

  PMI = 0   -> knowing the class tells you nothing about whether i fires
  PMI > 0   -> i fires more than its baseline rate when c is present
               (evidence FOR c)
  PMI < 0   -> i fires less than its baseline rate when c is present
               (evidence AGAINST c)

This module estimates P(fire_i | c) empirically from activations (no
model-internal assumptions beyond `encode`) and derives PMI(i, c) for
every feature/class pair. It's used both by the standalone PMI eval
(`ca_sae.cmd.mutual_information`, which reduces PMI to per-feature and
per-architecture informativeness summaries) and by the graded probe
perturbation eval (`ca_sae.cmd.tpp`, which uses PMI as a per-feature,
per-class informativeness score to weight edits).

Model API assumed (same as the other eval scripts in this codebase):
  - `model.encode(x)`  -> sparse code z, [B, d], already top-k-gated
  - `model.dict_size`  -> d
"""

import torch
from tqdm import tqdm


@torch.inference_mode()
def compute_conditional_and_priors(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    num_classes: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streams over `x_all` in chunks, encoding each batch and accumulating
    fire_count[d, num_classes] and class_count[num_classes], rather than
    materializing a dense [N, d] firing indicator (and a [N, d] float
    copy of it) before ever reducing anything -- for a training-sized
    activation set (N ~ 1M+, d ~ 4096) that would be tens of GB held at
    once for no reason, since the only thing ever needed downstream is
    a [d, num_classes] sum. Peak memory here is
    O(d * num_classes + chunk_size * d) instead of O(N * d).

    We use `z > 0` rather than a magnitude threshold because the docs
    for this codebase's SAE API state the code is already top-k-gated,
    so nonzero entries ARE the selection -- consistent with the "use
    selection mass, not activation magnitude" convention used
    elsewhere in this codebase (see SoftSAE-CA's use of p over z).

    Returns:
      p_fire_given_c: [d, num_classes], P(fire_i | c)
      class_priors:   [num_classes], P(c) (empirical class frequency)
    """
    d = model.dict_size
    fire_count = torch.zeros(d, num_classes)
    class_count = torch.zeros(num_classes)

    for start in tqdm(range(0, len(x_all), chunk_size), desc="Streaming PMI stats"):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        z = model.encode(x_batch)
        fired = (z > 0).float()  # [B, d], one chunk at a time -- never the full N

        onehot = torch.zeros(len(labels_batch), num_classes, device=device)
        onehot.scatter_(1, labels_batch.unsqueeze(1), 1.0)

        fire_count += (fired.T @ onehot).cpu()
        class_count += onehot.sum(dim=0).cpu()

    p_fire_given_c = fire_count / class_count.clamp(min=1).unsqueeze(0)
    class_priors = class_count / class_count.sum().clamp(min=1)
    return p_fire_given_c, class_priors


def compute_marginal_firing_rate(
    p_fire_given_c: torch.Tensor, class_priors: torch.Tensor
) -> torch.Tensor:
    """
    P(fire_i) = sum_c P(fire_i | c) * P(c)

    This is the class-prior-weighted average of each feature's row in
    p_fire_given_c -- i.e. literally "how often does this feature fire,
    ignoring class." Note this can also be estimated directly by
    fired.float().mean(dim=0) on the unlabelled data; both should agree
    up to sampling noise, and computing it this way keeps everything
    downstream expressible in terms of the same two tensors.
    """
    return p_fire_given_c @ class_priors


def compute_pmi(
    p_fire_given_c: torch.Tensor,
    marginal_rate: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PMI(i, c) = log( P(fire_i | c) / P(fire_i) )

    Returns [d, num_classes]. Features/classes with essentially zero
    conditional or marginal firing rate get clamped via eps rather than
    producing -inf/nan; these entries carry no real evidence either way
    and are best excluded from summaries via firing-rate filters
    rather than trusted as extreme PMI values.
    """
    p_cond = p_fire_given_c.clamp(min=eps)
    p_marg = marginal_rate.clamp(min=eps).unsqueeze(1)
    return torch.log(p_cond / p_marg)


def normalized_pmi(
    pmi: torch.Tensor,
    p_fire_given_c: torch.Tensor,
    class_priors: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NPMI(i, c) = PMI(i, c) / -log(P(fire_i, c))

    Plain PMI is unbounded and biased toward rare events (a feature
    that fires on a single sample of a single class gets a huge PMI
    from noise alone). Normalizing into [-1, 1] makes magnitudes
    comparable across features with very different firing rates, which
    matters when comparing summaries *across architectures* that may
    have very different overall sparsity levels.

    P(fire_i, c) = P(fire_i | c) * P(c) -- the class prior multiplication
    is required here; omitting it (as an earlier version of this function
    did) silently returns values far outside [-1, 1], since P(fire_i | c)
    alone is much larger than the true joint whenever num_classes is large.
    """
    p_joint = (p_fire_given_c * class_priors.unsqueeze(0)).clamp(min=eps)
    return pmi / (-torch.log(p_joint))


def normalize_pmi_to_unit_interval(
    pmi: torch.Tensor, upper_percentile: float = 99.0
) -> torch.Tensor:
    """
    ReLU(PMI), then scaled into [0, 1] using a single GLOBAL upper
    percentile as the reference (not a per-class or per-feature
    min-max). Using a global reference means one exceptionally
    class-specific feature can't compress every other feature's
    informativeness toward zero on a per-column normalization, and it
    keeps informativeness values comparable across classes and across
    architectures at different sparsity levels.

    Only positive PMI counts as "evidence for c" -- negative or zero
    PMI (the feature fires no more than its own baseline rate for c,
    or actively less) is mapped to exactly 0. Ablating harder because a
    feature is anti-correlated with c would work against the stated
    goal, not for it.
    """
    pos = pmi.clamp(min=0)
    ref = torch.quantile(pos.flatten(), upper_percentile / 100.0).clamp(min=1e-6)
    return (pos / ref).clamp(max=1.0)
