"""
Feature-class pointwise mutual information (PMI) evaluation.

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

This script estimates P(fire_i | c) empirically from activations (no
model-internal assumptions beyond `encode`), derives PMI(i, c) for every
feature/class pair, and reduces this to per-feature and per-architecture
informativeness summaries so different SAE architectures (or checkpoints
of the same architecture) can be compared on a common scale.

Optionally, if the model exposes a trained feature-class table (e.g.
ClassAlignedSAE.calculate_M()), this script also computes a PMI *proxy*
directly from the trained parameters -- no data pass over classes beyond
a single label-free marginal-rate estimate -- and reports how well it
tracks the fully empirical PMI. This is a calibration check on the
parametric table itself, not just a feature-informativeness score.

Model API assumed (same as the other eval scripts in this codebase):
  - `model.encode(x)`  -> sparse code z, [B, d], already top-k-gated
  - `model.dict_size`  -> d
  - (optional) `model.calculate_M()` -> [d, C] trained claim matrix,
    for architectures that expose one (e.g. ClassAlignedSAE)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.sae.ca_sae import ClassAlignedSAE

try:
    from scipy.stats import pearsonr, spearmanr

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ======================================================================
# Data loading
# ======================================================================


def load_all_activations(activations_path: str, batch_size: int, num_workers: int):
    dataset = ActivationsDataset(activations_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    xs, ys = [], []
    for x, y in tqdm(loader, desc=f"Loading activations from {activations_path}"):
        xs.append(x.float())
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).long()


# ======================================================================
# Empirical firing statistics
# ======================================================================


@torch.inference_mode()
def compute_firing_indicator(
    model, x_all: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    """
    Returns a binary [N, d] tensor: whether each feature fired (was
    part of the top-k selection / nonzero code) for each sample.

    We use `z > 0` rather than a magnitude threshold because the docs
    for this codebase's SAE API state the code is already top-k-gated,
    so nonzero entries ARE the selection -- consistent with the "use
    selection mass, not activation magnitude" convention used
    elsewhere in this codebase (see SoftSAE-CA's use of p over z).
    """
    d = model.dict_size
    fired = torch.zeros(len(x_all), d, dtype=torch.bool)
    for start in tqdm(
        range(0, len(x_all), chunk_size), desc="Encoding for firing statistics"
    ):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        z = model.encode(x_batch)
        fired[start:end] = (z > 0).cpu()
    return fired


def compute_conditional_and_priors(
    fired: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    fired: [N, d] bool
    labels: [N]

    Returns:
      p_fire_given_c: [d, num_classes], P(fire_i | c)
      class_priors:   [num_classes], P(c) (empirical class frequency)
    """
    d = fired.shape[1]
    p_fire_given_c = torch.zeros(d, num_classes)
    class_priors = torch.zeros(num_classes)

    fired_f = fired.float()
    for c in range(num_classes):
        mask = labels == c
        n_c = mask.sum().item()
        class_priors[c] = n_c / len(labels)
        if n_c > 0:
            p_fire_given_c[:, c] = fired_f[mask].mean(dim=0)
        # if a class has zero samples in this split, its column stays 0;
        # it will simply never be selected as anyone's most-informative
        # class and contributes 0 prior mass to the marginal below.
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


# ======================================================================
# PMI
# ======================================================================


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
    and are best excluded from summaries via the firing-rate filters
    below rather than trusted as extreme PMI values.
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


# ======================================================================
# Per-feature and per-architecture summaries
# ======================================================================


def summarize_features(
    pmi: torch.Tensor,
    p_fire_given_c: torch.Tensor,
    marginal_rate: torch.Tensor,
    min_firing_rate: float,
    informative_pmi_threshold: float,
) -> dict:
    """
    Reduces the [d, C] PMI matrix to per-feature and aggregate
    informativeness numbers.

    A feature that essentially never fires (marginal_rate below
    min_firing_rate) is excluded from aggregates: its PMI values are
    dominated by sampling noise from a handful of activations rather
    than reflecting a real class relationship, and letting a few such
    features skew the mean would misrepresent architecture-level
    informativeness.
    """
    d, num_classes = pmi.shape
    active = marginal_rate >= min_firing_rate
    n_active = int(active.sum().item())

    max_pmi, argmax_class = pmi.max(dim=1)
    # "effective number of classes" a feature meaningfully fires for:
    # perplexity of its normalized conditional-firing distribution.
    # A specialist (fires ~only for one class) -> ~1. A generalist
    # (fires uniformly across all classes) -> ~num_classes.
    row_sum = p_fire_given_c.sum(dim=1, keepdim=True).clamp(min=1e-8)
    row_dist = p_fire_given_c / row_sum
    row_entropy = -(row_dist * torch.log(row_dist.clamp(min=1e-8))).sum(dim=1)
    effective_num_classes = torch.exp(row_entropy)

    frac_informative = (
        float((max_pmi[active] > informative_pmi_threshold).float().mean().item())
        if n_active > 0
        else None
    )

    per_feature = {
        "marginal_firing_rate": marginal_rate.tolist(),
        "max_pmi": max_pmi.tolist(),
        "argmax_class": argmax_class.tolist(),
        "effective_num_classes": effective_num_classes.tolist(),
        "is_active": active.tolist(),
    }

    aggregate = {
        "dict_size": d,
        "num_classes": num_classes,
        "num_active_features": n_active,
        "frac_active_features": n_active / d,
        "mean_max_pmi_active": (
            float(max_pmi[active].mean().item()) if n_active > 0 else None
        ),
        "std_max_pmi_active": (
            float(max_pmi[active].std().item()) if n_active > 1 else None
        ),
        # weight by firing rate so features that almost never fire
        # (but happen to clear the min_firing_rate bar) don't get
        # equal say to features that fire constantly and reliably
        "firing_weighted_mean_max_pmi": (
            float(
                (max_pmi[active] * marginal_rate[active]).sum().item()
                / marginal_rate[active].sum().item()
            )
            if n_active > 0 and marginal_rate[active].sum().item() > 0
            else None
        ),
        "frac_informative_features": frac_informative,
        "mean_effective_num_classes_active": (
            float(effective_num_classes[active].mean().item()) if n_active > 0 else None
        ),
        "informative_pmi_threshold": informative_pmi_threshold,
        "min_firing_rate": min_firing_rate,
    }
    return {"per_feature": per_feature, "aggregate": aggregate}


# ======================================================================
# Optional: calibration check against a trained parametric M
# ======================================================================


def effective_num_classes_from_M(M: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Same perplexity-of-normalized-distribution idea as
    summarize_features' effective_num_classes, but computed from the
    trained *claim* distribution M_hat = M / k rather than from raw
    empirical firing. Comparing this against the raw-firing version
    directly tests whether class-aligned training tightens claims
    (this number should track the target budget rho) without
    necessarily tightening raw activation breadth (the firing-based
    number, which is free to stay broad).
    """
    M_hat = M / k.clamp(min=1e-8).unsqueeze(1)
    entropy = -(M_hat * torch.log(M_hat.clamp(min=1e-8))).sum(dim=1)
    return torch.exp(entropy)


def pmi_proxy_from_trained_M(
    M: torch.Tensor,
    k: torch.Tensor,
    class_priors: torch.Tensor,
    marginal_rate: torch.Tensor,
) -> torch.Tensor:
    """
    Approximates PMI(i, c) from a trained claim matrix M (e.g.
    ClassAlignedSAE's M, before hardening if possible -- see caveat
    below) plus a single label-free marginal firing-rate estimate,
    without re-estimating the full empirical conditional P(fire_i | c).

    M_hat = M / k  is the row-normalized claim distribution, which by
    the water-filling result for the base agreement objective
    approximates a thresholded, renormalized empirical P(c | fire_i).
    We Bayes-flip it back into PMI's native P(fire_i | c) / P(fire_i)
    form using the (cheap, label-free) marginal_rate:

        P(c | fire_i) ~= M_hat[i, c]
        P(fire_i | c) = P(c | fire_i) * P(fire_i) / P(c)     (Bayes)
        PMI(i, c)     = log( P(fire_i | c) / P(fire_i) )
                      = log( M_hat[i, c] / P(c) )

    Caveat: this is only as good as (a) how close training got to the
    Proposition-1 vertex solution, which is itself only exact for the
    base (non-contrastive) agreement loss with a fixed encoder, and
    (b) whether M was captured before full hardening -- once alpha_M
    -> 0, every row collapses to a uniform 1/k_i over its claimed
    classes and this proxy loses all relative weighting, degrading to
    a same-or-different-class indicator rather than a graded score.
    """
    M_hat = M / k.clamp(min=1e-8).unsqueeze(1)
    p_c_given_fire_i = M_hat.clamp(min=1e-8)
    pmi_proxy = torch.log(p_c_given_fire_i) - torch.log(
        class_priors.clamp(min=1e-8)
    ).unsqueeze(0)
    return pmi_proxy


def compare_pmi_to_proxy(pmi_empirical: torch.Tensor, pmi_proxy: torch.Tensor) -> dict:
    """
    Flattens both PMI matrices and reports correlation -- the direct
    check of whether the trained table's implied selectivity tracks
    what actually happens on data, rather than assuming Proposition 1's
    theoretical vertex solution is closely realized in practice.
    """
    a = pmi_empirical.flatten().numpy()
    b = pmi_proxy.flatten().numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]

    result = {"num_pairs_compared": int(finite.sum())}
    if _SCIPY_AVAILABLE and len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        pear_r, pear_p = pearsonr(a, b)
        spear_r, spear_p = spearmanr(a, b)
        result.update(
            {
                "pearson_r": float(pear_r),
                "pearson_p": float(pear_p),
                "spearman_r": float(spear_r),
                "spearman_p": float(spear_p),
            }
        )
    elif len(a) >= 2 and np.std(a) > 0 and np.std(b) > 0:
        result["pearson_r"] = float(np.corrcoef(a, b)[0, 1])
        result["note"] = "install scipy for p-values and Spearman correlation"
    else:
        result["note"] = "insufficient variance or overlap to compute correlation"
    return result


# ======================================================================
# Main driver
# ======================================================================


@torch.inference_mode
def main(
    architecture: str,
    checkpoint_path: str,
    activations_path: str,
    num_classes: int | None,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    min_firing_rate: float,
    informative_pmi_threshold: float,
    informative_npmi_threshold: float,
    check_trained_M: bool,
    output_path: str | None,
    device: str | None,
    max_examples: int | None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path, device=device
    )
    model.eval()

    print("Loading activations...")
    x_all, labels_all = load_all_activations(activations_path, batch_size, num_workers)
    if max_examples is not None:
        x_all, labels_all = x_all[:max_examples], labels_all[:max_examples]

    if num_classes is None:
        num_classes = int(labels_all.max().item()) + 1
    print(
        f"num_classes={num_classes}, num_examples={len(x_all)}, dict_size={model.dict_size}"
    )

    fired = compute_firing_indicator(model, x_all, chunk_size, device)
    p_fire_given_c, class_priors = compute_conditional_and_priors(
        fired, labels_all, num_classes
    )
    marginal_rate = compute_marginal_firing_rate(p_fire_given_c, class_priors)
    pmi = compute_pmi(p_fire_given_c, marginal_rate)
    npmi = normalized_pmi(pmi, p_fire_given_c, class_priors)

    summary = summarize_features(
        pmi, p_fire_given_c, marginal_rate, min_firing_rate, informative_pmi_threshold
    )
    npmi_summary = summarize_features(
        npmi, p_fire_given_c, marginal_rate, min_firing_rate, informative_npmi_threshold
    )

    print("\n=== PMI summary (raw) ===")
    for key, val in summary["aggregate"].items():
        print(f"  {key}: {val}")
    print("\n=== NPMI summary (bounded [-1, 1], comparable across architectures) ===")
    for key, val in npmi_summary["aggregate"].items():
        print(f"  {key}: {val}")

    calibration = None
    eff_classes_claimed = None
    if check_trained_M:
        if not isinstance(model, ClassAlignedSAE):
            print(
                "\n[note] --check-trained-M was set but model is not a ClassAlignedSAE "
                "(no calculate_M() available); skipping calibration check."
            )
        else:
            print("\nComparing empirical PMI against trained-M PMI proxy...")
            M = model.calculate_M().to(device)
            k = M.sum(dim=1)  # per-feature claim mass, from the trained table itself
            pmi_proxy = pmi_proxy_from_trained_M(
                M.cpu(), k.cpu(), class_priors, marginal_rate
            )
            calibration = compare_pmi_to_proxy(pmi, pmi_proxy)
            print("\n=== Calibration: does trained M track empirical PMI? ===")
            for key, val in calibration.items():
                print(f"  {key}: {val}")

            eff_classes_claimed = effective_num_classes_from_M(M.cpu(), k.cpu())
            print(
                "\n=== Claimed vs. raw-firing breadth (does CA tighten claims without tightening firing?) ==="
            )
            print(
                f"  mean effective_num_classes, M claims (target ~ rho):   {eff_classes_claimed.mean().item():.2f}"
            )
            print(
                f"  mean effective_num_classes, raw empirical firing:      {summary['aggregate']['mean_effective_num_classes_active']:.2f}"
            )

    result = {
        "architecture": architecture,
        "checkpoint_path": checkpoint_path,
        "num_examples": len(x_all),
        "pmi_summary": summary,
        "npmi_summary": npmi_summary,
        "trained_M_calibration": calibration,
        "eff_classes_claimed_M": (
            eff_classes_claimed.tolist() if eff_classes_claimed is not None else None
        ),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved results to {output_path}")

    if not _SCIPY_AVAILABLE:
        print(
            "\n[note] scipy not found -- calibration p-values and Spearman correlation "
            "were skipped. Install with `pip install scipy` for the full stats."
        )

    return result


def cli():
    parser = argparse.ArgumentParser(
        description=(
            "Compute feature-class pointwise mutual information (PMI) to measure "
            "per-feature class-informativeness, comparable across SAE architectures."
        )
    )
    parser.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--activations-path", required=True)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Defaults to max(labels) + 1 if not given.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--min-firing-rate",
        type=float,
        default=1e-4,
        help="Features with marginal firing rate below this are excluded from "
        "aggregate summaries (their PMI is dominated by sampling noise).",
    )
    parser.add_argument(
        "--informative-pmi-threshold",
        type=float,
        default=2.0,
        help="A feature's max-over-classes raw PMI above this counts as 'informative' "
        "for the raw-PMI frac_informative_features summary. PMI=2.0 means the feature "
        "fires ~e^2 ~= 7.4x more often for its top class than its baseline rate.",
    )
    parser.add_argument(
        "--informative-npmi-threshold",
        type=float,
        default=0.3,
        help="Same idea as --informative-pmi-threshold but on the NPMI ([-1,1]) scale "
        "used for the NPMI summary. Must be well under 1.0 -- unlike raw PMI, NPMI "
        "cannot exceed 1.0, so reusing the raw-PMI threshold here silently zeroes "
        "this metric.",
    )
    parser.add_argument(
        "--check-trained-M",
        action="store_true",
        help="If the model exposes calculate_M() (e.g. ClassAlignedSAE), also "
        "compute a parameter-only PMI proxy and correlate it against the "
        "empirical PMI, as a calibration check on the trained table.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-examples", type=int, default=None)

    args = parser.parse_args()
    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        activations_path=args.activations_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        min_firing_rate=args.min_firing_rate,
        informative_pmi_threshold=args.informative_pmi_threshold,
        informative_npmi_threshold=args.informative_npmi_threshold,
        check_trained_M=args.check_trained_M,
        output_path=args.output_path,
        device=args.device,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    cli()
