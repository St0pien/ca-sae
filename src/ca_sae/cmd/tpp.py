"""
Graded, PMI-weighted probe perturbation evaluation for SoftSAE-CA.

Replaces the hard "ablate every feature with M[i,c] > 0" rule with a
continuous, per-sample edit:

    strength_i(x, c) = activation_strength_i(x) * informativeness_i(c)
    z'(x)             = z(x) * (1 - strength(x, c))

A feature is only edited hard when BOTH conditions hold: it fired
strongly for this specific sample (relative to its own typical firing
magnitude), AND it carries strong empirical evidence for class c
(positive PMI(i, c), the same quantity computed in the PMI eval
script). A feature that's uninformative for c stays untouched no
matter how hard it fired; a feature that's highly informative for c
but didn't fire on this particular sample is untouched too, since
there's nothing there to remove. This replaces two arbitrary choices
in the original script -- a hard M[:,c]>0 threshold, and all-or-nothing
zeroing -- with a single graded, class- and sample-aware quantity.

Why activation magnitude is normalized per-feature, not used raw:
the SoftSAE-CA paper's own Remark 1 notes that the magnitude of z_i is
"fixed by reconstruction and decoder-column norms and is largely
arbitrary as a measure of relevance" -- i.e. raw z_i is not comparable
ACROSS features. We therefore normalize each feature's activation by a
reference scale (a percentile of that feature's own nonzero firing
magnitudes, estimated once from held-out data) before combining it
with informativeness. This is a within-feature normalization, so
"strongly activated" means "unusually strong for this feature", not
"has a numerically larger raw z_i than some other feature".

Control condition: rather than a size-matched random feature *set*
(the natural control for a binary mask, but with no obvious analog for
a continuous edit), we shuffle the informativeness vector across
feature indices while keeping the activation-based gating identical.
Because this only permutes the SAME set of informativeness values, the
total edit mass is identical to the real condition by construction --
no separate size-matching is needed. This isolates the question the
eval is actually about: does content-aware ALIGNMENT between
informativeness and feature identity matter, or would any
activation-shaped partial edit of the same magnitude do just as well?

Model API assumed (same as the other eval scripts in this codebase):
  - `model.encode(x)`  -> sparse code z, [B, d], already top-k-gated
  - `model.decode(z)`  -> [B, d] -> [B, n]
  - `model.dict_size`  -> d
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset

try:
    from scipy.stats import pearsonr, wilcoxon

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
# Empirical PMI(i, c) -- same construction as the standalone PMI eval
# script. Duplicated here (rather than imported) only for
# self-containedness; if this lives in the same package as that
# script, prefer importing compute_firing_indicator /
# compute_conditional_and_priors / compute_marginal_firing_rate /
# compute_pmi from there instead of maintaining two copies.
# ======================================================================


@torch.inference_mode()
def compute_pmi_stats_streaming(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    num_classes: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming replacement for compute_firing_indicator + compute_conditional_and_priors.

    The original two-step version materializes a dense [N, d] bool
    tensor (fired) and then a [N, d] float copy of it (fired_f) before
    ever reducing anything -- for a training-sized activation set
    (N ~ 1M+, d ~ 4096) that's tens of GB held at once for no reason,
    since the only thing ever needed downstream is a [d, num_classes]
    sum. This version accumulates fire_count[d, num_classes] and
    class_count[num_classes] batch by batch, so peak memory is
    O(d * num_classes + chunk_size * d) instead of O(N * d).
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
    return p_fire_given_c @ class_priors


def compute_pmi(
    p_fire_given_c: torch.Tensor, marginal_rate: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p_cond = p_fire_given_c.clamp(min=eps)
    p_marg = marginal_rate.clamp(min=eps).unsqueeze(1)
    return torch.log(p_cond / p_marg)


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


def compute_activation_reference_scale(
    model,
    x_ref: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    percentile: float = 90.0,
    max_examples_for_estimate: int = 50_000,
) -> torch.Tensor:
    """
    Per-feature reference activation magnitude: a percentile of each
    feature's own NONZERO firing values, estimated on a bounded-size
    random subsample rather than the full reference set.

    A percentile estimate doesn't need every example -- a random
    subsample of ~50k is statistically sufficient, and it caps peak
    memory (previously this concatenated a dense [N, d] float32 tensor
    for the FULL reference set, e.g. ~21GB at N=1.28M, d=4096; capping
    N here bounds that to a fixed, predictable size regardless of how
    large x_ref is).
    """
    if len(x_ref) > max_examples_for_estimate:
        idx = torch.randperm(len(x_ref))[:max_examples_for_estimate]
        x_ref = x_ref[idx]

    d = model.dict_size
    z_chunks = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(x_ref), chunk_size),
            desc="Estimating per-feature activation scale",
        ):
            z_chunks.append(
                model.encode(x_ref[start : start + chunk_size].to(device)).cpu()
            )
    z_all = torch.cat(z_chunks, dim=0)  # [min(N, max_examples_for_estimate), d]

    ref_scale = torch.ones(d)
    for i in range(d):
        nz = z_all[:, i]
        nz = nz[nz > 0]
        if len(nz) > 0:
            ref_scale[i] = torch.quantile(nz, percentile / 100.0)
    return ref_scale.clamp(min=1e-6)


# ======================================================================
# Building and applying the graded edit
# ======================================================================


def build_strength_vector(
    z_batch: torch.Tensor,
    ref_scale: torch.Tensor,
    info_c: torch.Tensor,
    edit_strength_scale: float,
) -> torch.Tensor:
    """
    z_batch:   [B, d] raw SAE codes for this batch
    ref_scale: [d]    per-feature reference activation magnitude
    info_c:    [d]    per-class informativeness in [0, 1] for class c

    Returns strength in [0, 1]^[B, d]: the fraction of each feature's
    activation to remove for a sample being edited toward class c.
    Elementwise product of "how strongly active, relative to this
    feature's own scale" and "how informative for this class" -- a
    feature that is high on only one of the two stays close to
    untouched, matching the requested behavior directly.
    """
    act_norm = (z_batch / ref_scale.unsqueeze(0)).clamp(max=1.0)
    strength = act_norm * info_c.unsqueeze(0) * edit_strength_scale
    return strength.clamp(min=0.0, max=1.0)


@torch.inference_mode()
def graded_edit_and_probe(
    model,
    probe: nn.Linear,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    ref_scale: torch.Tensor,
    info_c: torch.Tensor,
    target_class: int,
    edit_strength_scale: float,
    chunk_size: int,
    device: torch.device,
) -> dict:
    """
    Same measurement contract as the original ablate_and_probe:
    forget_accuracy on target-class samples, retain_accuracy on
    everything else, mean post-edit logit for target_class. The edit
    itself is now z' = z * (1 - strength) instead of a hard zero-mask.
    """
    ref_scale = ref_scale.to(device)
    info_c = info_c.to(device)

    target_mask = labels_all == target_class
    retain_mask = ~target_mask

    correct = torch.empty(len(x_all), dtype=torch.bool)
    target_logit_after = torch.empty(len(x_all))
    mean_strength_applied = torch.empty(len(x_all))
    effective_features_edited = torch.empty(len(x_all))

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        z = model.encode(x_batch)
        strength = build_strength_vector(z, ref_scale, info_c, edit_strength_scale)
        z_edited = z * (1.0 - strength)
        x_hat = model.decode(z_edited)

        logits = probe(x_hat)
        preds = logits.argmax(dim=1)

        correct[start:end] = (preds == labels_batch).cpu()
        target_logit_after[start:end] = logits[:, target_class].cpu()
        mean_strength_applied[start:end] = strength.mean(dim=1).cpu()
        # sum(strength) per sample = "effective number of fully-ablated-equivalent
        # features" -- e.g. two features edited at strength 0.5 each contribute the
        # same 1.0 as one feature fully zeroed. This is the direct, interpretable
        # analog of the old hard-threshold script's |F_c| (features actually
        # ablated), letting the two approaches be compared apples-to-apples on
        # "how much of the representation was touched" rather than only on
        # mean_strength_applied, which is diluted by the (correctly) untouched
        # majority of the dictionary and is hard to eyeball on its own.
        effective_features_edited[start:end] = strength.sum(dim=1).cpu()

    return {
        "forget_accuracy": correct[target_mask].float().mean().item(),
        "retain_accuracy": correct[retain_mask].float().mean().item(),
        "mean_target_logit_after": target_logit_after[target_mask].mean().item(),
        "mean_strength_applied": mean_strength_applied.mean().item(),
        "mean_effective_features_edited": effective_features_edited.mean().item(),
    }


@torch.inference_mode()
def probe_baseline(
    probe: nn.Linear,
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    target_class: int,
    chunk_size: int,
    device: torch.device,
    round_trip_through_sae: bool,
) -> dict:
    target_mask = labels_all == target_class
    retain_mask = ~target_mask
    correct = torch.empty(len(x_all), dtype=torch.bool)
    target_logit = torch.empty(len(x_all))

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        if round_trip_through_sae:
            z = model.encode(x_batch)
            x_used = model.decode(z)
        else:
            x_used = x_batch

        logits = probe(x_used)
        preds = logits.argmax(dim=1)
        correct[start:end] = (preds == labels_batch).cpu()
        target_logit[start:end] = logits[:, target_class].cpu()

    return {
        "forget_accuracy": correct[target_mask].float().mean().item(),
        "retain_accuracy": correct[retain_mask].float().mean().item(),
        "mean_target_logit_after": target_logit[target_mask].mean().item(),
    }


# ======================================================================
# Independent linear probe (unchanged from the original script)
# ======================================================================


def train_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> nn.Linear:
    n, dim = x_train.shape
    probe = nn.Linear(dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)

    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss, total_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch, y_batch = x_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = probe(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        print(
            f"  probe epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / n:.4f} acc={total_correct / n:.4f}"
        )

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


# ======================================================================
# Aggregate summary
# ======================================================================


def compute_aggregate_metrics(results: dict) -> dict:
    classes = sorted(results.keys())
    baseline_acc = np.array(
        [results[c]["baseline_sae_roundtrip"]["forget_accuracy"] for c in classes]
    )
    baseline_retain = np.array(
        [results[c]["baseline_sae_roundtrip"]["retain_accuracy"] for c in classes]
    )
    graded_forget = np.array(
        [results[c]["graded_edit"]["forget_accuracy"] for c in classes]
    )
    graded_retain = np.array(
        [results[c]["graded_edit"]["retain_accuracy"] for c in classes]
    )
    shuffled_forget = np.array(
        [results[c]["shuffled_info_control"]["forget_accuracy"] for c in classes]
    )
    shuffled_retain = np.array(
        [results[c]["shuffled_info_control"]["retain_accuracy"] for c in classes]
    )
    mean_strength = np.array(
        [results[c]["graded_edit"]["mean_strength_applied"] for c in classes]
    )
    mean_eff_features_graded = np.array(
        [results[c]["graded_edit"]["mean_effective_features_edited"] for c in classes]
    )
    mean_eff_features_shuffled = np.array(
        [
            results[c]["shuffled_info_control"]["mean_effective_features_edited"]
            for c in classes
        ]
    )

    graded_drop = baseline_acc - graded_forget
    shuffled_drop = baseline_acc - shuffled_forget
    paired_diff = graded_drop - shuffled_drop

    win_rate = float(np.mean(paired_diff > 0)) if len(paired_diff) > 0 else None

    wilcoxon_stat, wilcoxon_p = None, None
    corr_r, corr_p = None, None
    if _SCIPY_AVAILABLE:
        if len(paired_diff) >= 2 and not np.allclose(graded_drop, shuffled_drop):
            wilcoxon_stat, wilcoxon_p = wilcoxon(graded_drop, shuffled_drop)
            wilcoxon_stat, wilcoxon_p = float(wilcoxon_stat), float(wilcoxon_p)
        if (
            len(baseline_acc) >= 2
            and np.std(baseline_acc) > 0
            and np.std(graded_drop) > 0
        ):
            corr_r, corr_p = pearsonr(baseline_acc, graded_drop)
            corr_r, corr_p = float(corr_r), float(corr_p)
    elif (
        len(baseline_acc) >= 2 and np.std(baseline_acc) > 0 and np.std(graded_drop) > 0
    ):
        corr_r = float(np.corrcoef(baseline_acc, graded_drop)[0, 1])

    return {
        "num_classes": len(classes),
        "mean_edit_strength_applied": (
            float(mean_strength.mean()) if len(mean_strength) else None
        ),
        "mean_effective_features_edited_graded": (
            float(mean_eff_features_graded.mean())
            if len(mean_eff_features_graded)
            else None
        ),
        "mean_effective_features_edited_shuffled": (
            float(mean_eff_features_shuffled.mean())
            if len(mean_eff_features_shuffled)
            else None
        ),
        # forget-accuracy drop per "unit" of representation touched -- the
        # direct efficiency comparison against the old hard-threshold script's
        # |F_c|-features-ablated approach: how much forgetting do you get per
        # effectively-fully-ablated feature, on average.
        "forget_drop_per_effective_feature_graded": (
            float(np.mean(graded_drop / np.clip(mean_eff_features_graded, 1e-6, None)))
            if len(mean_eff_features_graded)
            else None
        ),
        "mean_forget_accuracy_drop_graded_edit": (
            float(graded_drop.mean()) if len(graded_drop) else None
        ),
        "std_forget_accuracy_drop_graded_edit": (
            float(graded_drop.std()) if len(graded_drop) else None
        ),
        "mean_forget_accuracy_drop_shuffled_control": (
            float(shuffled_drop.mean()) if len(shuffled_drop) else None
        ),
        "std_forget_accuracy_drop_shuffled_control": (
            float(shuffled_drop.std()) if len(shuffled_drop) else None
        ),
        "mean_paired_difference_graded_minus_shuffled": (
            float(paired_diff.mean()) if len(paired_diff) else None
        ),
        "fraction_of_classes_graded_beats_shuffled": win_rate,
        "wilcoxon_statistic_graded_vs_shuffled": wilcoxon_stat,
        "wilcoxon_p_value_graded_vs_shuffled": wilcoxon_p,
        "mean_retain_accuracy_baseline": (
            float(baseline_retain.mean()) if len(baseline_retain) else None
        ),
        "mean_retain_accuracy_graded_edit": (
            float(graded_retain.mean()) if len(graded_retain) else None
        ),
        "mean_retain_accuracy_shuffled_control": (
            float(shuffled_retain.mean()) if len(shuffled_retain) else None
        ),
        "mean_retain_accuracy_collateral_graded": (
            float((graded_retain - baseline_retain).mean())
            if len(graded_retain)
            else None
        ),
        "mean_retain_accuracy_collateral_shuffled": (
            float((shuffled_retain - baseline_retain).mean())
            if len(shuffled_retain)
            else None
        ),
        "pearson_corr_baseline_accuracy_vs_graded_drop": corr_r,
        "pearson_corr_p_value": corr_p,
        "scipy_available": _SCIPY_AVAILABLE,
    }


def print_aggregate_summary(agg: dict) -> None:
    print("\n=== Aggregate summary (graded PMI-weighted edit) ===")
    print(f"Classes evaluated:                                {agg['num_classes']}")
    print(
        f"Mean edit strength applied (both conditions):     {agg['mean_edit_strength_applied']:.4f}"
    )
    print(
        f"Mean effective features edited, graded / shuffled: "
        f"{agg['mean_effective_features_edited_graded']:.2f} / "
        f"{agg['mean_effective_features_edited_shuffled']:.2f}"
    )
    print(
        f"Forget-drop per effective feature edited (graded): {agg['forget_drop_per_effective_feature_graded']:.4f}"
    )
    print(
        f"Mean forget-acc drop, graded edit:                "
        f"{agg['mean_forget_accuracy_drop_graded_edit']:.3f} "
        f"(std {agg['std_forget_accuracy_drop_graded_edit']:.3f})"
    )
    print(
        f"Mean forget-acc drop, shuffled-info control:      "
        f"{agg['mean_forget_accuracy_drop_shuffled_control']:.3f} "
        f"(std {agg['std_forget_accuracy_drop_shuffled_control']:.3f})"
    )
    print(
        f"Mean paired difference (graded - shuffled):       {agg['mean_paired_difference_graded_minus_shuffled']:.3f}"
    )
    print(
        f"Fraction of classes where graded beats shuffled:  {agg['fraction_of_classes_graded_beats_shuffled']:.2f}"
    )
    if agg["wilcoxon_p_value_graded_vs_shuffled"] is not None:
        print(
            f"Wilcoxon signed-rank (graded vs shuffled):         "
            f"stat={agg['wilcoxon_statistic_graded_vs_shuffled']:.3f} "
            f"p={agg['wilcoxon_p_value_graded_vs_shuffled']:.4f}"
        )
    else:
        print(
            "Wilcoxon signed-rank (graded vs shuffled):         (install scipy, or drops were identical)"
        )
    print(
        f"Mean retain-accuracy collateral, graded / shuffled: "
        f"{agg['mean_retain_accuracy_collateral_graded']:+.4f} / "
        f"{agg['mean_retain_accuracy_collateral_shuffled']:+.4f}"
    )
    if agg["pearson_corr_baseline_accuracy_vs_graded_drop"] is not None:
        p_str = (
            f"p={agg['pearson_corr_p_value']:.4f}"
            if agg["pearson_corr_p_value"] is not None
            else "(p n/a)"
        )
        print(
            f"Corr(baseline accuracy, graded drop):             r={agg['pearson_corr_baseline_accuracy_vs_graded_drop']:.3f} {p_str}"
        )


# ======================================================================
# Main driver
# ======================================================================


def main(
    architecture: str,
    checkpoint_path: str,
    train_activations_path: str,
    test_activations_path: str,
    pmi_activations_path: str | None,
    target_classes: list[int] | None,
    num_random_target_classes: int,
    edit_strength_scale: float,
    pmi_upper_percentile: float,
    activation_scale_percentile: float,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
    probe_batch_size: int,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    seed: int,
    output_path: str | None,
    device: str | None,
    max_train_examples: int | None,
    max_test_examples: int | None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    rng = random.Random(seed)
    torch.manual_seed(seed)

    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path, device=device
    )
    model.eval()

    print(
        "Loading probe-training activations (clean, never touched by the SAE again)..."
    )
    x_train, y_train = load_all_activations(
        train_activations_path, batch_size, num_workers
    )
    if max_train_examples is not None:
        x_train, y_train = x_train[:max_train_examples], y_train[:max_train_examples]

    print("Loading evaluation activations...")
    x_test, y_test = load_all_activations(
        test_activations_path, batch_size, num_workers
    )
    if max_test_examples is not None:
        x_test, y_test = x_test[:max_test_examples], y_test[:max_test_examples]

    num_classes = int(y_train.max().item()) + 1

    # ------------------------------------------------------------
    # PMI(i, c) and per-feature activation reference scale, both
    # estimated on held-out reference data (defaults to the probe's
    # training split if a separate one isn't given -- see caveat in
    # the printed note below about what this does and doesn't buy you).
    # ------------------------------------------------------------
    if pmi_activations_path is not None:
        print("Loading separate activations for PMI / activation-scale estimation...")
        x_pmi, y_pmi = load_all_activations(
            pmi_activations_path, batch_size, num_workers
        )
    else:
        print(
            "[note] --pmi-activations-path not given; reusing the probe-training split "
            "for PMI and activation-scale estimation. This does not leak test labels, "
            "but if you want PMI estimated fully independently of anything the probe "
            "saw, pass a third, disjoint split explicitly."
        )
        x_pmi, y_pmi = x_train, y_train

    p_fire_given_c, class_priors = compute_pmi_stats_streaming(
        model, x_pmi, y_pmi, num_classes, chunk_size, device
    )
    marginal_rate = compute_marginal_firing_rate(p_fire_given_c, class_priors)
    pmi = compute_pmi(p_fire_given_c, marginal_rate)
    info = normalize_pmi_to_unit_interval(
        pmi, upper_percentile=pmi_upper_percentile
    )  # [d, C]

    ref_scale = compute_activation_reference_scale(
        model, x_pmi, chunk_size, device, percentile=activation_scale_percentile
    )  # [d]

    # ------------------------------------------------------------
    # Train and freeze the independent probe
    # ------------------------------------------------------------
    print("Training independent linear probe on clean embeddings...")
    probe = train_linear_probe(
        x_train,
        y_train,
        num_classes,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        batch_size=probe_batch_size,
        device=device,
    )

    # ------------------------------------------------------------
    # Target classes
    # ------------------------------------------------------------
    targets = list(target_classes) if target_classes is not None else []
    if num_random_target_classes > 0:
        pool = [c for c in range(num_classes) if c not in targets]
        targets = targets + rng.sample(pool, min(num_random_target_classes, len(pool)))
    print(f"Target classes: {targets}")

    # ------------------------------------------------------------
    # Per-class graded-edit sweep
    # ------------------------------------------------------------
    results = {}
    d = model.dict_size
    for c in tqdm(targets, desc="Graded probe perturbation"):
        info_c = info[:, c]
        info_c_shuffled = info_c[
            torch.randperm(d, generator=torch.Generator().manual_seed(seed + c))
        ]

        baseline_raw = probe_baseline(
            probe,
            model,
            x_test,
            y_test,
            c,
            chunk_size,
            device,
            round_trip_through_sae=False,
        )
        baseline_roundtrip = probe_baseline(
            probe,
            model,
            x_test,
            y_test,
            c,
            chunk_size,
            device,
            round_trip_through_sae=True,
        )

        graded_result = graded_edit_and_probe(
            model,
            probe,
            x_test,
            y_test,
            ref_scale,
            info_c,
            c,
            edit_strength_scale,
            chunk_size,
            device,
        )
        shuffled_result = graded_edit_and_probe(
            model,
            probe,
            x_test,
            y_test,
            ref_scale,
            info_c_shuffled,
            c,
            edit_strength_scale,
            chunk_size,
            device,
        )

        results[c] = {
            "baseline_raw_embeddings": baseline_raw,
            "baseline_sae_roundtrip": baseline_roundtrip,
            "graded_edit": graded_result,
            "shuffled_info_control": shuffled_result,
        }

        print(
            f"  class {c}: baseline(raw/roundtrip) forget_acc="
            f"{baseline_raw['forget_accuracy']:.3f}/{baseline_roundtrip['forget_accuracy']:.3f} | "
            f"graded_edit forget_acc={graded_result['forget_accuracy']:.3f} "
            f"retain_acc={graded_result['retain_accuracy']:.3f} "
            f"eff_features_edited={graded_result['mean_effective_features_edited']:.2f} | "
            f"shuffled_control forget_acc={shuffled_result['forget_accuracy']:.3f} "
            f"retain_acc={shuffled_result['retain_accuracy']:.3f} "
            f"eff_features_edited={shuffled_result['mean_effective_features_edited']:.2f}"
        )

    aggregate = compute_aggregate_metrics(results)
    print_aggregate_summary(aggregate)
    if not _SCIPY_AVAILABLE:
        print(
            "\n[note] scipy not found -- Wilcoxon test skipped. `pip install scipy` for the full stats."
        )

    summary = {
        "architecture": architecture,
        "num_target_classes": len(results),
        "edit_strength_scale": edit_strength_scale,
        "pmi_upper_percentile": pmi_upper_percentile,
        "activation_scale_percentile": activation_scale_percentile,
        "aggregate": aggregate,
        "results_by_class": results,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved results to {output_path}")

    return summary


def cli():
    parser = argparse.ArgumentParser(
        description="Graded, PMI-weighted probe perturbation evaluation for SoftSAE-CA."
    )
    parser.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-activations-path", required=True)
    parser.add_argument("--test-activations-path", required=True)
    parser.add_argument(
        "--pmi-activations-path",
        default=None,
        help="Optional separate split for estimating PMI(i,c) and per-feature activation "
        "reference scale. Defaults to reusing --train-activations-path if not given.",
    )
    parser.add_argument("--target-classes", type=int, nargs="+", default=None)
    parser.add_argument("--num-random-target-classes", type=int, default=20)
    parser.add_argument(
        "--edit-strength-scale",
        type=float,
        default=1.0,
        help="Global multiplier on the strength vector before clamping to [0,1]. "
        "Values > 1 push more (activation, informativeness) pairs toward full ablation.",
    )
    parser.add_argument("--pmi-upper-percentile", type=float, default=99.0)
    parser.add_argument("--activation-scale-percentile", type=float, default=90.0)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)

    args = parser.parse_args()
    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        train_activations_path=args.train_activations_path,
        test_activations_path=args.test_activations_path,
        pmi_activations_path=args.pmi_activations_path,
        target_classes=args.target_classes,
        num_random_target_classes=args.num_random_target_classes,
        edit_strength_scale=args.edit_strength_scale,
        pmi_upper_percentile=args.pmi_upper_percentile,
        activation_scale_percentile=args.activation_scale_percentile,
        probe_epochs=args.probe_epochs,
        probe_lr=args.probe_lr,
        probe_weight_decay=args.probe_weight_decay,
        probe_batch_size=args.probe_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output_path=args.output_path,
        device=args.device,
        max_train_examples=args.max_train_examples,
        max_test_examples=args.max_test_examples,
    )


if __name__ == "__main__":
    cli()
