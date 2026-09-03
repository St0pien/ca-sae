"""
Class-editing evaluation for SoftSAE-CA: structural alignment, specialist
vs. generalist ablation, and cross-class steering.

Three independent checks, each cheaper/more informative than the last:

  1. Structural alignment check (no forward passes, no classifier).
     Read F_c = {i : M[i, c] > 0} straight off the trained matrix M and
     ask: do sibling classes (small WordNet distance) share more claimed
     features than distant classes do? This is computed once from M and
     the WordNet distance matrix alone, so it's worth running BEFORE
     investing in the sweeps below -- if the correlation is weak here,
     the causal experiments downstream are unlikely to look good either.

  2. Specialist vs. generalist ablation grid. For each target class c,
     split F_c into "specialist" (low per-feature budget k_i) and
     "generalist" (high k_i) subsets using global quantiles of k, then
     hard-ablate each subset (plus the full F_c, plus a size-matched
     random control) and report forget accuracy on c together with
     retain accuracy broken out by WordNet-distance tier to c
     (sibling / close / distant / unrelated). The prediction under test:
     specialist ablation -> large forget effect, ~no collateral outside
     class c; generalist ablation -> collateral that fades with
     WordNet distance; random ablation -> small effect on everything.

  3. Cross-class steering. Instead of just erasing c, edit a class-c
     sample's code toward a *different* class c': remove the evidence
     for c that the sample actually used (its own selected coords
     intersected with F_c), then inject F_c' at a magnitude matched to
     the class-conditional mean activation of those features on real
     c' samples, scaled by a dose parameter alpha. Reports steering
     success rate (does zero-shot prediction flip to c'?), a manifold-
     validity check against a random-vector-of-matched-norm control
     (so "steered" isn't confounded with "broken"), a dose-response
     sweep, and success rate as a function of WordNet distance between
     c and c' -- with specialist-only vs. specialist+generalist
     injection run separately, since the prediction is that reaching a
     sibling class needs only the specialist delta while reaching a
     distant class needs the shared/generalist features too.

Model API assumptions (matching the codebase's existing eval scripts):
  - `model.encode(x)`  returns the sparse code z, [B, d], already
    top-k-gated (non-selected coords are exactly 0).
  - `model.decode(z)`  maps a code back to embedding space, [B, d] -> [B, n].
  - `model.dict_size`  is d.
  - `ClassAlignedSAE.calculate_M()` / `.budget_vector` expose the
    trained feature-class table directly; non-CA architectures fall
    back to a precomputed empirical matrix via `build_posthoc_M`.

No forget-set gradients are used anywhere. F_c / F_c' are lookups into
M, and steering-target magnitudes come from a single pass computing
class-conditional mean activations -- both are cheap, non-optimization
operations, consistent with the rest of this codebase's eval scripts.
"""

import argparse
import itertools
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ca_sae.const import SUPPORTED_ARCHITECTURES
from ca_sae.dataset import ActivationsDataset
from ca_sae.eval.posthoc_M import build_posthoc_M
from ca_sae.labels import IMAGENET2012_CLASSES
from ca_sae.sae.ca_sae import ClassAlignedSAE
from ca_sae.sae.core import topk_per_row

try:
    from nltk.corpus import wordnet as wn
except ImportError as e:
    raise ImportError(
        "This script needs nltk's WordNet corpus:\n"
        "    pip install nltk\n"
        "    python -c \"import nltk; nltk.download('wordnet')\""
    ) from e


# ======================================================================
# WordNet: ImageNet class <-> synset, pairwise taxonomic distance
# ======================================================================


def load_imagenet_classes(imagenet_labels_dict: OrderedDict):
    wnids = list(imagenet_labels_dict.keys())
    synonyms = [
        [s.strip() for s in label.split(",")] for label in imagenet_labels_dict.values()
    ]
    return wnids, synonyms


def wnid_to_synset(wnid: str):
    return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))


def build_wordnet_distance_matrix(wnids, cache_path: str | None = None):
    """dist[i, j] = shortest-path distance (# edges) between class i and
    j's synsets. dist[i, i] = 0. Cached to disk since it never depends
    on the SAE checkpoint."""
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)

    c = len(wnids)
    synsets = [wnid_to_synset(w) for w in wnids]
    dist = np.zeros((c, c), dtype=np.int32)

    for i in tqdm(range(c), desc="Building WordNet distance matrix"):
        for j in range(i + 1, c):
            d = synsets[i].shortest_path_distance(synsets[j])
            if d is None:
                d = 999
            dist[i, j] = d
            dist[j, i] = d

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, dist)

    return dist


def distance_to_bin(distance: int, bin_edges: list[int]) -> str:
    """bin_edges=[2,4,6] -> "sibling" (d<=2), "close" (2<d<=4),
    "distant" (4<d<=6), "unrelated" (d>6)."""
    default_names = ["sibling", "close", "distant", "unrelated"]
    names = (
        default_names
        if len(bin_edges) + 1 <= len(default_names)
        else [f"bin_{k}" for k in range(len(bin_edges) + 1)]
    )
    for edge, name in zip(bin_edges, names):
        if distance <= edge:
            return name
    return names[len(bin_edges)]


def classes_under_hypernym(wnids, hypernym_wnid: str) -> list[int]:
    """Class indices whose synset has `hypernym_wnid` as an ancestor."""
    target = wnid_to_synset(hypernym_wnid)
    out = []
    for idx, w in enumerate(wnids):
        syn = wnid_to_synset(w)
        if any(target in path for path in syn.hypernym_paths()):
            out.append(idx)
    return out


# ======================================================================
# Rank correlation (no scipy dependency)
# ======================================================================


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank ranking, ties get the mean of their tied ranks."""
    a = np.asarray(a, dtype=np.float64)
    unique_vals, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank = (start + cum - 1) / 2.0
    return avg_rank[inverse]


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


# ======================================================================
# Zero-shot CLIP classifier weights (external measuring stick)
# ======================================================================

IMAGENET_ZEROSHOT_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


@torch.inference_mode()
def build_zeroshot_weights(
    synonyms: list[list[str]],
    clip_model_name: str = "ViT-B/32",
    templates: list[str] | None = None,
    device: torch.device | str = "cuda",
    batch_size: int = 256,
) -> torch.Tensor:
    """Standard CLIP zero-shot classifier construction (Radford et al.
    2021, Sec 3.1.4). Returns [C, D] L2-normalized text embeddings."""
    import clip

    templates = templates or IMAGENET_ZEROSHOT_TEMPLATES
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    weights = []
    for class_synonyms in tqdm(synonyms, desc="Building zero-shot weights"):
        prompts = [t.format(name) for name in class_synonyms for t in templates]
        embeds_chunks = []
        for i in range(0, len(prompts), batch_size):
            tokens = clip.tokenize(prompts[i : i + batch_size]).to(device)
            e = model.encode_text(tokens)
            e = e / e.norm(dim=-1, keepdim=True)
            embeds_chunks.append(e)
        class_embed = torch.cat(embeds_chunks, dim=0).mean(dim=0)
        class_embed = class_embed / class_embed.norm()
        weights.append(class_embed.float())

    return torch.stack(weights, dim=0)


@torch.inference_mode()
def zeroshot_classify(x: torch.Tensor, zeroshot_weights: torch.Tensor) -> torch.Tensor:
    """x: [B, D] (unnormalized ok). Returns cosine-similarity scores [B, C]."""
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return x @ zeroshot_weights.T.to(x.dtype)


def topk_correctness(
    scores: torch.Tensor, labels: torch.Tensor, k: int
) -> torch.Tensor:
    k = min(k, scores.shape[1])
    top_idx = scores.topk(k=k, dim=1).indices
    return (top_idx == labels.unsqueeze(1)).any(dim=1)


# ======================================================================
# Data loading
# ======================================================================


def load_all_activations(activations_path: str, batch_size: int, num_workers: int):
    dataset = ActivationsDataset(activations_path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    xs, ys = [], []
    for x, y in tqdm(loader, desc="Loading activations"):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0).long()


# ======================================================================
# Feature-class matrix M, per-feature budgets k
# ======================================================================


def load_feature_class_matrix(
    model, precomputed_matrix: str | None, rho: float, device: torch.device
):
    if isinstance(model, ClassAlignedSAE) and precomputed_matrix is None:
        print("Using built-in M matrix from ClassAlignedSAE model")
        M = model.calculate_M()
        k = (
            torch.softmax(model.budget_vector, dim=0)
            * model.features_per_class
            * model.dict_size
        )
    else:
        if precomputed_matrix is None:
            raise ValueError(
                "Model is not a ClassAlignedSAE, so --precomputed-matrix is required."
            )
        print(
            f"Loading precomputed empirical feature-class matrix from: {precomputed_matrix}"
        )
        train_A = torch.load(precomputed_matrix).to(device)
        print("Constructing post-hoc feature-class matrix M...")
        M, k = build_posthoc_M(train_A, rho=rho)

    M = topk_per_row(M, k)
    return M.to(device, dtype=torch.float32), k.to(device, dtype=torch.float32)


def compute_row_entropy(M: torch.Tensor) -> torch.Tensor:
    """Normalized entropy of each feature's class-claim row, in [0, 1].
    0 = pure specialist, 1 = budget spread maximally broadly."""
    eps = 1e-12
    row_sum = M.sum(dim=1, keepdim=True).clamp_min(eps)
    p = M / row_sum
    h = -(p * (p + eps).log()).sum(dim=1)
    k = M.sum(dim=1)
    max_h = k.clamp_min(1.0 + eps).log()
    return torch.where(k > 1, h / max_h.clamp_min(eps), torch.zeros_like(h)).clamp(
        0.0, 1.0
    )


# ======================================================================
# Check 1: structural alignment -- does M's own geometry track WordNet?
# ======================================================================


def structural_alignment_check(
    M: torch.Tensor, dist_matrix: np.ndarray, bin_edges: list[int]
) -> dict:
    """
    Computes Jaccard(F_c, F_c') for every class pair directly from M --
    no forward passes, no classifier, no editing. Vectorized: with
    binary indicator B = (M > 0) of shape [d, C],

        intersection = B.T @ B          [C, C]
        size_c       = B.sum(0)         [C]
        union        = size_c[:,None] + size_c[None,:] - intersection
        jaccard      = intersection / union

    Then correlates the upper-triangle of `jaccard` against the
    corresponding WordNet distances (Spearman, since the relationship
    is expected to be monotone, not linear), and reports mean jaccard
    per WordNet distance bin as a more interpretable summary.

    If the correlation here is weak or the wrong sign, the causal
    ablation/steering experiments below are unlikely to show the
    hierarchy-shaped collateral the method predicts -- worth checking
    first, since this costs O(C^2) dense matmuls and nothing else.
    """
    B = (M > 0).float()
    intersection = (B.T @ B).cpu().numpy()
    size_c = B.sum(dim=0).cpu().numpy()
    union = size_c[:, None] + size_c[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, intersection / union, 0.0)

    c = jaccard.shape[0]
    iu = np.triu_indices(c, k=1)
    jacc_pairs = jaccard[iu]
    dist_pairs = dist_matrix[iu]

    corr = spearman_corr(dist_pairs, jacc_pairs)

    bin_names = [distance_to_bin(int(d), bin_edges) for d in dist_pairs]
    bins = sorted(set(bin_names))
    mean_jaccard_by_bin = {
        b: float(jacc_pairs[np.array(bin_names) == b].mean()) for b in bins
    }
    count_by_bin = {b: int((np.array(bin_names) == b).sum()) for b in bins}

    return {
        "spearman_corr_distance_vs_jaccard": corr,
        "mean_jaccard_by_wordnet_bin": mean_jaccard_by_bin,
        "pair_count_by_wordnet_bin": count_by_bin,
        "note": "expect a negative correlation: closer classes should share more claimed features",
    }


# ======================================================================
# Check 2: specialist vs. generalist ablation, stratified by WordNet tier
# ======================================================================


def split_specialist_generalist(
    k: torch.Tensor,
    feature_indices: torch.Tensor,
    spec_quantile: float,
    gen_quantile: float,
) -> tuple[list[int], list[int]]:
    """Thresholds are computed over the *global* distribution of k (all
    d features), not just F_c, so "specialist"/"generalist" mean the
    same thing across every target class."""
    lo = torch.quantile(k, spec_quantile).item()
    hi = torch.quantile(k, gen_quantile).item()
    idx = feature_indices.tolist()
    specialists = [i for i in idx if k[i].item() <= lo]
    generalists = [i for i in idx if k[i].item() >= hi]
    return specialists, generalists


def build_ablation_conditions(
    f_c: torch.Tensor,
    k: torch.Tensor,
    dict_size: int,
    rng: random.Random,
    spec_quantile: float,
    gen_quantile: float,
) -> dict[str, list[int]]:
    specialists, generalists = split_specialist_generalist(
        k, f_c, spec_quantile, gen_quantile
    )
    full = f_c.tolist()

    def size_matched_random(n: int, exclude: set[int]) -> list[int]:
        pool = [i for i in range(dict_size) if i not in exclude]
        return rng.sample(pool, min(n, len(pool)))

    conditions = {"full": full}
    if specialists:
        conditions["specialists"] = specialists
        conditions["random_matched_specialists"] = size_matched_random(
            len(specialists), set(full)
        )
    if generalists:
        conditions["generalists"] = generalists
        conditions["random_matched_generalists"] = size_matched_random(
            len(generalists), set(full)
        )
    conditions["random_matched_full"] = size_matched_random(len(full), set(full))
    return conditions


@torch.inference_mode()
def ablate_and_evaluate(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    distance_row: np.ndarray,
    bin_edges: list[int],
    feature_indices: list[int],
    target_class: int,
    chunk_size: int,
    device: torch.device,
) -> dict:
    """Single hard ablation of `feature_indices` (zeroed for every
    sample, source class or not -- editing is a global, deployable
    surgery, not a per-sample intervention). Reports forget accuracy on
    `target_class`, retain accuracy overall, and retain accuracy broken
    out by WordNet-distance tier to `target_class`."""
    d = model.dict_size
    mask = torch.zeros(d, device=device, dtype=torch.bool)
    if feature_indices:
        mask[torch.tensor(feature_indices, device=device)] = True

    target_mask = labels_all == target_class
    retain_mask = ~target_mask
    sample_distance = distance_row[labels_all.numpy()]
    sample_bin = np.array(
        [distance_to_bin(int(dd), bin_edges) for dd in sample_distance]
    )

    correct_top1 = torch.empty(len(x_all), dtype=torch.bool)
    correct_top5 = torch.empty(len(x_all), dtype=torch.bool)
    cos_sims = torch.empty(len(x_all))

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        z = model.encode(x_batch)
        z_edited = z.masked_fill(mask.unsqueeze(0), 0.0)
        x_hat = model.decode(z_edited)

        scores = zeroshot_classify(x_hat, zeroshot_weights)
        preds = scores.argmax(dim=1)
        correct_top1[start:end] = (preds == labels_batch).cpu()
        correct_top5[start:end] = topk_correctness(scores, labels_batch, k=5).cpu()
        cos_sims[start:end] = torch.nn.functional.cosine_similarity(
            x_hat, x_batch, dim=-1
        ).cpu()

    retain_by_bin, retain_by_bin_top5 = {}, {}
    for bin_name in sorted(set(sample_bin[retain_mask.numpy()])):
        bm = torch.from_numpy((sample_bin == bin_name) & retain_mask.numpy())
        retain_by_bin[bin_name] = correct_top1[bm].float().mean().item()
        retain_by_bin_top5[bin_name] = correct_top5[bm].float().mean().item()

    return {
        "num_features_ablated": len(feature_indices),
        "forget_class_accuracy_top1": correct_top1[target_mask].float().mean().item(),
        "forget_class_accuracy_top5": correct_top5[target_mask].float().mean().item(),
        "retain_accuracy_overall_top1": correct_top1[retain_mask].float().mean().item(),
        "retain_accuracy_overall_top5": correct_top5[retain_mask].float().mean().item(),
        "retain_accuracy_by_wordnet_bin_top1": retain_by_bin,
        "retain_accuracy_by_wordnet_bin_top5": retain_by_bin_top5,
        "fidelity_cosine_mean_on_retain": cos_sims[retain_mask].mean().item(),
    }


def run_specialist_generalist_grid(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    dist_matrix: np.ndarray,
    bin_edges: list[int],
    M: torch.Tensor,
    k: torch.Tensor,
    target_classes: list[int],
    spec_quantile: float,
    gen_quantile: float,
    rng: random.Random,
    chunk_size: int,
    device: torch.device,
) -> dict:
    """The 2x2(x3) grid from the design doc: for each target class,
    {specialists, generalists, full, random-matched controls} crossed
    with {forget effect, retain-by-WordNet-tier collateral}."""
    grid = {}
    for c in tqdm(target_classes, desc="Specialist/generalist grid"):
        f_c = (M[:, c] > 0).nonzero(as_tuple=True)[0]
        if len(f_c) == 0:
            continue
        conditions = build_ablation_conditions(
            f_c, k, model.dict_size, rng, spec_quantile, gen_quantile
        )
        grid[c] = {}
        for cond_name, feats in conditions.items():
            grid[c][cond_name] = ablate_and_evaluate(
                model,
                x_all,
                labels_all,
                zeroshot_weights,
                dist_matrix[c],
                bin_edges,
                feats,
                c,
                chunk_size,
                device,
            )
    return grid


# ======================================================================
# Check 3: cross-class steering
# ======================================================================


@torch.inference_mode()
def compute_class_conditional_mean_activation(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    num_classes: int,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    """mean_activation[c, i] = mean over samples of class c of z_i (raw
    top-k-gated code, zeros included). This is the "typical activation
    when this feature is doing its normal job for class c" magnitude
    used to set injection strength during steering -- matching real
    activity, not an arbitrary constant."""
    d = model.dict_size
    sums = torch.zeros(num_classes, d, device=device)
    counts = torch.zeros(num_classes, device=device)

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)
        z = model.encode(x_batch)
        sums.index_add_(0, labels_batch, z)
        counts.index_add_(
            0, labels_batch, torch.ones_like(labels_batch, dtype=torch.float)
        )

    return sums / counts.clamp_min(1.0).unsqueeze(1)


@torch.inference_mode()
def steer_batch(
    model,
    x_batch: torch.Tensor,
    f_source_mask: torch.Tensor,
    f_target_indices: torch.Tensor,
    mean_activation_target_row: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Remove the sample's own evidence for the source class (its
    selected coords intersected with F_source), then set the target
    class's claimed features to `alpha` times their typical magnitude
    on real target-class samples. Returns the decoded, steered
    embedding x_hat."""
    z = model.encode(x_batch)
    selected = z != 0
    remove_mask = selected & f_source_mask.unsqueeze(0)
    z_steered = z.masked_fill(remove_mask, 0.0)
    if len(f_target_indices) > 0:
        z_steered[:, f_target_indices] = (
            alpha * mean_activation_target_row[f_target_indices]
        )
    return model.decode(z_steered)


@torch.inference_mode()
def random_vector_control_batch(
    model,
    x_batch: torch.Tensor,
    f_source_mask: torch.Tensor,
    injected_dim_count: int,
    injected_norm_ref: torch.Tensor,
    device: torch.device,
    rng_generator: torch.Generator,
) -> torch.Tensor:
    """Null baseline for steering: remove the same source evidence, but
    inject noise (random coords, matched in count and norm to the real
    injection) instead of the real class-c' features. If this produces
    a similar prediction-flip rate to the real steering, "success"
    above is really just "breaking the embedding", not steering it."""
    d = model.dict_size
    z = model.encode(x_batch)
    selected = z != 0
    remove_mask = selected & f_source_mask.unsqueeze(0)
    z_steered = z.masked_fill(remove_mask, 0.0)
    if injected_dim_count > 0:
        rand_idx = torch.randperm(d, generator=rng_generator, device="cpu")[
            :injected_dim_count
        ].to(device)
        noise = torch.randn(
            injected_dim_count, generator=rng_generator, device="cpu"
        ).to(device)
        noise = noise / noise.norm().clamp_min(1e-12) * injected_norm_ref
        z_steered[:, rand_idx] = noise
    return model.decode(z_steered)


@torch.inference_mode()
def evaluate_steering_pair(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    M: torch.Tensor,
    k: torch.Tensor,
    mean_activation: torch.Tensor,
    source_class: int,
    target_class: int,
    alphas: list[float],
    spec_quantile: float,
    gen_quantile: float,
    samples_per_pair: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Steer up to `samples_per_pair` source-class samples toward
    target_class, at each alpha, under two injection modes:
      "specialists_only": inject only the low-k_i subset of F_target
      "full":              inject all of F_target (specialists + generalists)
    Prediction under test: "specialists_only" should already succeed
    for nearby classes but fall off faster with WordNet distance than
    "full" does.
    """
    torch_gen = torch.Generator().manual_seed(seed)

    idx = (labels_all == source_class).nonzero(as_tuple=True)[0]
    if len(idx) > samples_per_pair:
        perm = torch.randperm(len(idx), generator=torch_gen)[:samples_per_pair]
        idx = idx[perm]
    if len(idx) == 0:
        return {"num_samples": 0}

    f_source = (M[:, source_class] > 0).nonzero(as_tuple=True)[0]
    f_source_mask = torch.zeros(model.dict_size, device=device, dtype=torch.bool)
    f_source_mask[f_source.to(device)] = True

    f_target = (M[:, target_class] > 0).nonzero(as_tuple=True)[0]
    target_specialists, _ = split_specialist_generalist(
        k, f_target, spec_quantile, gen_quantile
    )
    injection_sets = {
        "specialists_only": torch.tensor(
            target_specialists, device=device, dtype=torch.long
        ),
        "full": f_target.to(device),
    }

    mean_row = mean_activation[target_class].to(device)
    x_src = x_all[idx].to(device)
    real_norm_ref = x_src.norm(dim=-1).mean()

    result: dict = {"num_samples": len(idx), "by_mode": {}}

    for mode_name, target_idx in injection_sets.items():
        if len(target_idx) == 0:
            continue
        mode_result = {"dose_response": []}
        for alpha in alphas:
            x_hat_chunks, control_chunks = [], []
            for start in range(0, len(x_src), chunk_size):
                x_batch = x_src[start : start + chunk_size]
                x_hat_chunks.append(
                    steer_batch(
                        model, x_batch, f_source_mask, target_idx, mean_row, alpha
                    )
                )
                injected_norm_ref = (alpha * mean_row[target_idx]).norm()
                control_chunks.append(
                    random_vector_control_batch(
                        model,
                        x_batch,
                        f_source_mask,
                        len(target_idx),
                        injected_norm_ref,
                        device,
                        torch_gen,
                    )
                )
            x_hat = torch.cat(x_hat_chunks, dim=0)
            x_control = torch.cat(control_chunks, dim=0)

            scores = zeroshot_classify(x_hat, zeroshot_weights)
            preds = scores.argmax(dim=1)
            success_rate = (preds == target_class).float().mean().item()
            target_cos = scores[:, target_class].mean().item()

            control_scores = zeroshot_classify(x_control, zeroshot_weights)
            control_preds = control_scores.argmax(dim=1)
            control_success_rate = (control_preds == target_class).float().mean().item()

            mode_result["dose_response"].append(
                {
                    "alpha": alpha,
                    "steering_success_rate": success_rate,
                    "random_control_success_rate": control_success_rate,
                    "mean_cosine_to_target_text": target_cos,
                    "mean_embedding_norm": x_hat.norm(dim=-1).mean().item(),
                    "real_embedding_norm_ref": real_norm_ref.item(),
                }
            )
        result["by_mode"][mode_name] = mode_result

    return result


def evaluate_steering_grid(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    M: torch.Tensor,
    k: torch.Tensor,
    mean_activation: torch.Tensor,
    dist_matrix: np.ndarray,
    bin_edges: list[int],
    class_pairs: list[tuple[int, int]],
    alphas: list[float],
    spec_quantile: float,
    gen_quantile: float,
    samples_per_pair: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    per_pair = {}
    for c_src, c_tgt in tqdm(class_pairs, desc="Steering pairs"):
        per_pair[(c_src, c_tgt)] = evaluate_steering_pair(
            model,
            x_all,
            labels_all,
            zeroshot_weights,
            M,
            k,
            mean_activation,
            c_src,
            c_tgt,
            alphas,
            spec_quantile,
            gen_quantile,
            samples_per_pair,
            chunk_size,
            device,
            seed,
        )

    # Aggregate success rate (at the largest alpha, "full" mode) by
    # WordNet distance bin between source and target.
    by_bin: dict[str, list[float]] = {}
    for (c_src, c_tgt), res in per_pair.items():
        if "full" not in res.get("by_mode", {}):
            continue
        dose = res["by_mode"]["full"]["dose_response"]
        if not dose:
            continue
        last = dose[-1]
        bin_name = distance_to_bin(int(dist_matrix[c_src, c_tgt]), bin_edges)
        by_bin.setdefault(bin_name, []).append(last["steering_success_rate"])

    success_rate_by_wordnet_bin = {b: float(np.mean(v)) for b, v in by_bin.items()}

    return {
        "per_pair": {
            f"{c_src}->{c_tgt}": res for (c_src, c_tgt), res in per_pair.items()
        },
        "steering_success_rate_by_wordnet_bin_full_mode": success_rate_by_wordnet_bin,
    }


# ======================================================================
# Main driver
# ======================================================================


def main(
    architecture: str,
    checkpoint_path: str,
    test_activations_path: str,
    precomputed_matrix: str | None,
    zeroshot_weights_path: str,
    wordnet_distance_cache: str | None,
    target_classes: list[int] | None,
    hypernym_cluster: str | None,
    num_random_target_classes: int,
    rho: float,
    spec_quantile: float,
    gen_quantile: float,
    distance_bin_edges: list[int],
    run_structural_check: bool,
    run_ablation_grid: bool,
    run_steering: bool,
    num_steering_pairs: int,
    steering_samples_per_pair: int,
    steering_alphas: list[float],
    clip_model_name: str,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    seed: int,
    output_path: str | None,
    device: str | None,
    max_test_examples: int | None,
    imagenet_ordered_dict: OrderedDict = IMAGENET2012_CLASSES,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    rng = random.Random(seed)

    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path, device=device
    )
    model.eval()

    M, k = load_feature_class_matrix(model, precomputed_matrix, rho, device)

    print("Loading ImageNet class / synset mapping...")
    wnids, synonyms = load_imagenet_classes(imagenet_ordered_dict)
    assert len(wnids) == 1000, f"Expected 1000 classes, got {len(wnids)}"

    print("Building/loading WordNet distance matrix...")
    dist_matrix = build_wordnet_distance_matrix(
        wnids, cache_path=wordnet_distance_cache
    )

    if Path(zeroshot_weights_path).exists():
        print(f"Loading precomputed zero-shot weights from {zeroshot_weights_path}")
        zeroshot_weights = (
            torch.from_numpy(np.load(zeroshot_weights_path)).float().to(device)
        )
    else:
        print("Precomputing CLIP zero-shot weights (this only needs to happen once)...")
        zeroshot_weights = build_zeroshot_weights(
            synonyms, clip_model_name=clip_model_name, device=device
        )
        Path(zeroshot_weights_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(zeroshot_weights_path, zeroshot_weights.cpu().numpy())

    if target_classes is not None:
        targets = list(target_classes)
    elif hypernym_cluster is not None:
        print(f"Selecting target classes under hypernym {hypernym_cluster}...")
        targets = classes_under_hypernym(wnids, hypernym_cluster)
    else:
        targets = []
    if num_random_target_classes > 0:
        pool = [i for i in range(1000) if i not in targets]
        targets = targets + rng.sample(pool, min(num_random_target_classes, len(pool)))
    print(f"Target classes: {targets}")

    summary: dict = {
        "architecture": architecture,
        "rho": rho,
        "target_classes": targets,
    }

    # -- Check 1: structural alignment (no forward passes needed) -----
    if run_structural_check:
        print("\n=== Structural alignment check (M vs. WordNet) ===")
        structural = structural_alignment_check(M, dist_matrix, distance_bin_edges)
        print(
            f"Spearman corr(distance, jaccard): {structural['spearman_corr_distance_vs_jaccard']:.4f}"
        )
        print(f"Mean jaccard by bin: {structural['mean_jaccard_by_wordnet_bin']}")
        summary["structural_alignment"] = structural

    # -- Data needed for checks 2 and 3 --------------------------------
    if run_ablation_grid or run_steering:
        print("\nLoading test activations...")
        x_all, labels_all = load_all_activations(
            test_activations_path, batch_size, num_workers
        )
        if max_test_examples is not None:
            x_all = x_all[:max_test_examples]
            labels_all = labels_all[:max_test_examples]

    # -- Check 2: specialist vs. generalist ablation grid --------------
    if run_ablation_grid:
        print("\n=== Specialist vs. generalist ablation grid ===")
        grid = run_specialist_generalist_grid(
            model,
            x_all,
            labels_all,
            zeroshot_weights,
            dist_matrix,
            distance_bin_edges,
            M,
            k,
            targets,
            spec_quantile,
            gen_quantile,
            rng,
            chunk_size,
            device,
        )
        for c, conditions in grid.items():
            print(f"  class {c} ({wnids[c]}):")
            for cond_name, r in conditions.items():
                print(
                    f"    [{cond_name:28s}] forget_top1={r['forget_class_accuracy_top1']:.3f} "
                    f"retain_overall_top1={r['retain_accuracy_overall_top1']:.3f} "
                    f"retain_by_bin_top1={r['retain_accuracy_by_wordnet_bin_top1']}"
                )
        summary["specialist_generalist_grid"] = {str(c): v for c, v in grid.items()}

    # -- Check 3: cross-class steering ---------------------------------
    if run_steering:
        print("\n=== Cross-class steering ===")
        num_classes = zeroshot_weights.shape[0]
        mean_activation = compute_class_conditional_mean_activation(
            model, x_all, labels_all, num_classes, chunk_size, device
        )

        if len(targets) >= 2:
            all_pairs = list(itertools.permutations(targets, 2))
        else:
            all_pairs = list(itertools.permutations(range(1000), 2))
        rng.shuffle(all_pairs)
        class_pairs = all_pairs[:num_steering_pairs]

        steering = evaluate_steering_grid(
            model,
            x_all,
            labels_all,
            zeroshot_weights,
            M,
            k,
            mean_activation,
            dist_matrix,
            distance_bin_edges,
            class_pairs,
            steering_alphas,
            spec_quantile,
            gen_quantile,
            steering_samples_per_pair,
            chunk_size,
            device,
            seed,
        )
        print(
            f"Steering success rate by WordNet bin (full mode, max alpha): "
            f"{steering['steering_success_rate_by_wordnet_bin_full_mode']}"
        )
        summary["steering"] = steering

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved results to {output_path}")

    return summary


def cli():
    parser = argparse.ArgumentParser(
        description="SoftSAE-CA editing evaluation: structural alignment, specialist/generalist "
        "ablation, and cross-class steering."
    )

    parser.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-activations-path", required=True)
    parser.add_argument("--precomputed-matrix", default=None)
    parser.add_argument("--zeroshot-weights-path", required=True)
    parser.add_argument("--wordnet-distance-cache", default=None)

    target_args = parser.add_mutually_exclusive_group(required=False)
    target_args.add_argument("--target-classes", type=int, nargs="+", default=None)
    target_args.add_argument(
        "--hypernym-cluster",
        type=str,
        default=None,
        help="wnid whose descendants form the target-class cluster, e.g. n02084071 (dog).",
    )

    parser.add_argument("--num-random-target-classes", type=int, default=10)
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument(
        "--spec-quantile",
        type=float,
        default=0.25,
        help="features with k_i at/below this global quantile are 'specialists'.",
    )
    parser.add_argument(
        "--gen-quantile",
        type=float,
        default=0.75,
        help="features with k_i at/above this global quantile are 'generalists'.",
    )
    parser.add_argument("--distance-bin-edges", type=int, nargs="+", default=[2, 4, 6])

    parser.add_argument("--run-structural-check", action="store_true", default=True)
    parser.add_argument(
        "--no-structural-check", dest="run_structural_check", action="store_false"
    )
    parser.add_argument("--run-ablation-grid", action="store_true", default=True)
    parser.add_argument(
        "--no-ablation-grid", dest="run_ablation_grid", action="store_false"
    )
    parser.add_argument("--run-steering", action="store_true", default=True)
    parser.add_argument("--no-steering", dest="run_steering", action="store_false")

    parser.add_argument("--num-steering-pairs", type=int, default=50)
    parser.add_argument("--steering-samples-per-pair", type=int, default=64)
    parser.add_argument(
        "--steering-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    )

    parser.add_argument("--clip-model-name", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)

    args = parser.parse_args()

    main(
        architecture=args.architecture,
        checkpoint_path=args.checkpoint_path,
        test_activations_path=args.test_activations_path,
        precomputed_matrix=args.precomputed_matrix,
        zeroshot_weights_path=args.zeroshot_weights_path,
        wordnet_distance_cache=args.wordnet_distance_cache,
        target_classes=args.target_classes,
        hypernym_cluster=args.hypernym_cluster,
        num_random_target_classes=args.num_random_target_classes,
        rho=args.rho,
        spec_quantile=args.spec_quantile,
        gen_quantile=args.gen_quantile,
        distance_bin_edges=args.distance_bin_edges,
        run_structural_check=args.run_structural_check,
        run_ablation_grid=args.run_ablation_grid,
        run_steering=args.run_steering,
        num_steering_pairs=args.num_steering_pairs,
        steering_samples_per_pair=args.steering_samples_per_pair,
        steering_alphas=args.steering_alphas,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        seed=args.seed,
        output_path=args.output_path,
        device=args.device,
        max_test_examples=args.max_test_examples,
    )


if __name__ == "__main__":
    cli()
