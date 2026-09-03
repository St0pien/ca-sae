"""
Concept-erasure / unlearning evaluation for SoftSAE-CA.

Pipeline for a target class c:

    CLIP image embedding x
      -> SAE encode         -> z            (already top-k-gated: zeros
                                              on non-selected coords)
      -> ablate/soften z_i for i in F_c = {i : M[i, c] > 0}
      -> SAE decode         -> x_hat
      -> zero-shot CLIP classification of x_hat against class text
         embeddings (a CLIP-external measuring stick, never touched by
         the SAE)

No forget-set gradients are used anywhere: F_c is read directly off the
feature-class matrix M (or a post-hoc estimate of it, built exactly as
in the free-classifier eval script), and the per-target-class step is
just indexing a column of M. That "editing is a lookup, not an
optimization" property is the whole point of the experiment, so nothing
downstream should require labeled forget-set gradients either.

Assumptions about the model API (matching the free-classifier script):
  - `model.encode(x)`  returns the sparse code z, [B, d], already
    top-k-gated (non-selected coords are exactly 0).
  - `model.decode(z)`  maps a code back to embedding space, [B, d] -> [B, n].
  - `model.dict_size`  is d.

Top-1 AND top-5 accuracy are tracked throughout (raw CLIP baseline, SAE
round-trip baseline, and every sweep point's forget/retain accuracy).
This matters specifically because fine-grained classes (e.g. dog
breeds) can have very weak top-1 zero-shot CLIP accuracy while still
being "roughly known" in the top-5 sense -- top-5 gives a less noisy
signal for exactly the classes where the erasure story is most
interesting, and lets you sanity-check whether a low top-1 baseline is
"CLIP has no idea" vs "CLIP is confusing within a small cluster".
"""

import argparse
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
    """
    dist[i, j] = shortest-path distance (# edges) between class i and j's
    synsets in the WordNet noun hypernym/hyponym graph. dist[i, i] = 0.

    O(C^2) synset-pair BFS calls, so this is cached to disk keyed only
    on the class list -- it never depends on the SAE checkpoint, so you
    should only ever pay this cost once.
    """
    if cache_path is not None and Path(cache_path).exists():
        return np.load(cache_path)

    c = len(wnids)
    synsets = [wnid_to_synset(w) for w in wnids]
    dist = np.zeros((c, c), dtype=np.int32)

    for i in tqdm(range(c), desc="Building WordNet distance matrix"):
        for j in range(i + 1, c):
            d = synsets[i].shortest_path_distance(synsets[j])
            if d is None:
                # Shouldn't happen for ImageNet nouns (all under
                # entity.n.01), but don't crash the run over it.
                d = 999
            dist[i, j] = d
            dist[j, i] = d

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, dist)

    return dist


def distance_to_bin(distance: int, bin_edges: list[int]) -> str:
    """
    bin_edges=[2,4,6] -> "sibling" (d<=2), "close" (2<d<=4),
    "distant" (4<d<=6), "unrelated" (d>6). distance==0 (the class
    itself) should be excluded upstream, not binned here.
    """
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
    """
    Convenience for picking a "hard regime" cluster automatically, e.g.
    all dog breeds under n02084071 (dog, domestic dog, Canis familiaris).
    Returns class indices (positions in `wnids`) whose synset has the
    given hypernym as an ancestor.
    """
    target = wnid_to_synset(hypernym_wnid)
    out = []
    for idx, w in enumerate(wnids):
        syn = wnid_to_synset(w)
        hypernym_paths = syn.hypernym_paths()
        if any(target in path for path in hypernym_paths):
            out.append(idx)
    return out


# ======================================================================
# Zero-shot CLIP classifier weights (external measuring stick)
# ======================================================================

# Reproduced from memory of the widely-shared CLIP "Prompt Engineering
# for ImageNet" notebook template ensemble. I can't fetch the original
# list to verify it here -- spot-check a handful of entries against the
# official openai/CLIP repo before trusting absolute accuracy numbers
# built on top of it. It should not affect *relative* comparisons
# between architectures/editing strategies, only absolute scale.
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
    """
    Standard CLIP zero-shot classifier construction (Radford et al.
    2021, Sec 3.1.4): for each class, average the (template x synonym)
    text embeddings, then L2-normalize.

    Uses the *official* CLIP repo (not open_clip), as requested:
        pip install git+https://github.com/openai/CLIP.git

    Returns:
        weights: [C, D] float32, L2-normalized text embedding per class.
    """
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
    """
    scores: [B, C], labels: [B]. Returns a bool tensor [B]: True where
    the true label is among the top-k highest-scoring classes.
    """
    k = min(k, scores.shape[1])
    top_idx = scores.topk(k=k, dim=1).indices  # [B, k]
    return (top_idx == labels.unsqueeze(1)).any(dim=1)


@torch.inference_mode()
def compute_raw_zeroshot_baseline(
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    target_classes: list[int],
    chunk_size: int,
    device: torch.device,
) -> dict[int, dict[str, float] | None]:
    """
    Zero-shot CLIP accuracy on the *untouched* embeddings -- no SAE
    round-trip, no editing at all. This isolates "does CLIP reliably
    recognize this class in the first place" from anything the SAE or
    the editing procedure does. Computed once, since it doesn't depend
    on any editing setting.

    Returns {class_idx: {"top1": acc, "top5": acc}}, None for classes
    with zero support in the loaded split.
    """
    correct_top1 = torch.empty(len(x_all), dtype=torch.bool)
    correct_top5 = torch.empty(len(x_all), dtype=torch.bool)

    for start in range(0, len(x_all), chunk_size):
        end = start + chunk_size
        x_batch = x_all[start:end].to(device)
        labels_batch = labels_all[start:end].to(device)

        scores = zeroshot_classify(x_batch, zeroshot_weights)
        preds = scores.argmax(dim=1)

        correct_top1[start:end] = (preds == labels_batch).cpu()
        correct_top5[start:end] = topk_correctness(scores, labels_batch, k=5).cpu()

    baseline = {}
    for c in target_classes:
        mask = labels_all == c
        if mask.sum() == 0:
            baseline[c] = None
            continue
        baseline[c] = {
            "top1": correct_top1[mask].float().mean().item(),
            "top5": correct_top5[mask].float().mean().item(),
        }
    return baseline


# ======================================================================
# Ablation strength: hard / graded / entropy-weighted (specialist vs
# generalist aware)
# ======================================================================


def compute_row_entropy(M: torch.Tensor) -> torch.Tensor:
    """
    Normalized entropy of each feature's class-claim row M[i, :], read
    as a distribution over classes.

        entropy_i = 0   pure specialist, all mass on one class
        entropy_i -> 1  budget spread broadly (generalist)

    Normalized by log(k_i) (the max entropy achievable at that
    feature's own budget) so entropy is comparable across features
    with very different k_i, rather than penalizing high-k_i features
    just for having more room to spread mass over.
    """
    eps = 1e-12
    row_sum = M.sum(dim=1, keepdim=True).clamp_min(eps)
    p = M / row_sum
    h = -(p * (p + eps).log()).sum(dim=1)

    k = M.sum(dim=1)
    max_h = k.clamp_min(1.0 + eps).log()

    normalized = torch.where(k > 1, h / max_h.clamp_min(eps), torch.zeros_like(h))
    return normalized.clamp(0.0, 1.0)


def compute_ablation_strength(
    M: torch.Tensor,
    row_entropy: torch.Tensor,
    target_class: int,
    mode: str,
) -> torch.Tensor:
    """
    Per-feature ablation strength s_i in [0, 1] for erasing target_class.

    "hard":                s_i = 1{M[i, c] > 0}
    "graded_association":  s_i = M[i, c]
    "entropy_weighted":    s_i = M[i, c] * (1 - row_entropy[i])
        Down-weights generalist features relative to specialists that
        claim the class equally strongly, since ablating a generalist
        risks collateral damage on every other class it serves.
    """
    m_c = M[:, target_class]
    if mode == "hard":
        return (m_c > 0).float()
    elif mode == "graded_association":
        return m_c.clone()
    elif mode == "entropy_weighted":
        return m_c * (1.0 - row_entropy)
    raise ValueError(f"Unknown ablation strength mode: {mode!r}")


def build_feature_order(
    candidate_features: torch.Tensor,
    k: torch.Tensor,
    order: str,
    rng: random.Random,
) -> list[int]:
    """
    Order in which F_c is progressively ablated as N sweeps 0 -> |F_c|.

    "specialist_first": ascending by k_i. Cheap, class-specific
        features go first; features shared across many classes are
        left untouched longest.
    "random": size-matched random permutation -- the control that
        isolates whether picking the RIGHT features (not just removing
        SOME features) is what drives the result.
    """
    idx = candidate_features.tolist()
    if order == "specialist_first":
        return sorted(idx, key=lambda i: k[i].item())
    elif order == "random":
        idx = idx.copy()
        rng.shuffle(idx)
        return idx
    raise ValueError(f"Unknown order: {order!r}")


# ======================================================================
# Data loading (whole split into memory -- fine at ImageNet-val scale)
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
# Core sweep for a single target class / order / strength mode
# ======================================================================


@torch.inference_mode()
def run_sweep_for_setting(
    model,
    x_all: torch.Tensor,
    labels_all: torch.Tensor,
    zeroshot_weights: torch.Tensor,
    distance_row: np.ndarray,
    bin_edges: list[int],
    strength_full: torch.Tensor,
    feature_order: list[int],
    target_class: int,
    num_sweep_points: int,
    chunk_size: int,
    device: torch.device,
):
    """
    Sweeps N = number of features from `feature_order` that are active,
    from 0 to len(feature_order). At each N:
      - builds the strength vector (zeros except the first N ordered
        features, which get `strength_full`'s value at that index)
      - edits every sample's code with that same strength vector
        (editing is a global, sample-independent surgery -- matches how
        you'd actually deploy an edited decoder)
      - zero-shot classifies the reconstruction
      - reports forget accuracy, retain accuracy overall + by WordNet
        distance bin, and embedding fidelity on retained samples --
        each as both top-1 and top-5

    Returns a list of per-N result dicts.
    """
    d = model.dict_size
    n_total = len(feature_order)
    fractions = np.linspace(0.0, 1.0, num_sweep_points)
    n_values = sorted(set(int(round(f * n_total)) for f in fractions))

    target_mask = labels_all == target_class
    retain_mask = ~target_mask

    # WordNet bin per sample, based on distance from target_class
    # (excludes the target class itself, distance 0, by construction of
    # retain_mask above).
    sample_distance = distance_row[labels_all.numpy()]
    sample_bin = np.array(
        [distance_to_bin(int(dd), bin_edges) for dd in sample_distance]
    )

    results = []

    for n in n_values:
        active = feature_order[:n]
        strength_vec = torch.zeros(d, device=device)
        if active:
            active_idx = torch.tensor(active, device=device)
            strength_vec[active_idx] = strength_full[active_idx]

        preds = torch.empty(len(x_all), dtype=torch.long)
        correct_top1 = torch.empty(len(x_all), dtype=torch.bool)
        correct_top5 = torch.empty(len(x_all), dtype=torch.bool)
        cos_sims = torch.empty(len(x_all))

        for start in range(0, len(x_all), chunk_size):
            end = start + chunk_size
            x_batch = x_all[start:end].to(device)
            labels_batch = labels_all[start:end].to(device)

            z = model.encode(x_batch)
            z_edited = z * (1.0 - strength_vec.unsqueeze(0))
            x_hat = model.decode(z_edited)

            scores = zeroshot_classify(x_hat, zeroshot_weights)
            batch_preds = scores.argmax(dim=1)
            preds[start:end] = batch_preds.cpu()
            correct_top1[start:end] = (batch_preds == labels_batch).cpu()
            correct_top5[start:end] = topk_correctness(scores, labels_batch, k=5).cpu()

            cos = torch.nn.functional.cosine_similarity(x_hat, x_batch, dim=-1)
            cos_sims[start:end] = cos.cpu()

        # ---- forget accuracy (top-1 and top-5) + confusion destination ----
        # Confusion destination is inherently a top-1 notion (which
        # single class did the sample get reassigned to), so it isn't
        # duplicated for top-5.
        target_preds = preds[target_mask]
        forget_class_accuracy = correct_top1[target_mask].float().mean().item()
        forget_class_accuracy_top5 = correct_top5[target_mask].float().mean().item()

        misclassified = target_preds[target_preds != target_class]
        if len(misclassified) > 0:
            dest_counts = torch.bincount(
                misclassified, minlength=zeroshot_weights.shape[0]
            ).float()
            dest_probs = dest_counts / dest_counts.sum()
            nz = dest_probs[dest_probs > 0]
            dest_entropy = (-(nz * nz.log()).sum()).item()
            dest_entropy_norm = (
                dest_entropy / max(np.log(len(nz)), 1e-12) if len(nz) > 1 else 0.0
            )
            top_dest_class = int(dest_counts.argmax().item())
            top_dest_fraction = (dest_counts.max() / dest_counts.sum()).item()
        else:
            dest_entropy_norm, top_dest_class, top_dest_fraction = None, None, None

        # ---- retain accuracy: overall + by WordNet bin (top-1 and top-5) ----
        retain_accuracy_overall = correct_top1[retain_mask].float().mean().item()
        retain_accuracy_overall_top5 = correct_top5[retain_mask].float().mean().item()

        retain_by_bin = {}
        retain_by_bin_top5 = {}
        for bin_name in sorted(set(sample_bin[retain_mask.numpy()])):
            bin_mask_np = (sample_bin == bin_name) & retain_mask.numpy()
            bm = torch.from_numpy(bin_mask_np)
            retain_by_bin[bin_name] = correct_top1[bm].float().mean().item()
            retain_by_bin_top5[bin_name] = correct_top5[bm].float().mean().item()

        # ---- fidelity on retained samples --------------------------
        fidelity_cosine_mean = cos_sims[retain_mask].mean().item()

        results.append(
            {
                "n_features": n,
                "fraction_of_dict": n / d,
                "fraction_of_Fc": n / max(n_total, 1),
                "forget_class_accuracy": forget_class_accuracy,
                "forget_class_accuracy_top5": forget_class_accuracy_top5,
                "retain_accuracy_overall": retain_accuracy_overall,
                "retain_accuracy_overall_top5": retain_accuracy_overall_top5,
                "retain_accuracy_by_wordnet_bin": retain_by_bin,
                "retain_accuracy_by_wordnet_bin_top5": retain_by_bin_top5,
                "fidelity_cosine_mean": fidelity_cosine_mean,
                "confusion_destination_entropy_norm": dest_entropy_norm,
                "confusion_destination_top_class": top_dest_class,
                "confusion_destination_top_fraction": top_dest_fraction,
            }
        )

    return results


def forget_retain_auc(sweep_results: list[dict], metric_suffix: str = "") -> float:
    """
    Area under the (forgetting achieved) vs retain_accuracy_overall
    curve. "Forgetting achieved" is measured *relative to the SAE's own
    N=0 round-trip baseline* (sweep_results[0], no ablation applied),
    NOT as an absolute 1 - accuracy.

    metric_suffix="" uses top-1 fields (forget_class_accuracy,
    retain_accuracy_overall); metric_suffix="_top5" uses the top-5
    counterparts, so the exact same relative-to-baseline logic applies
    to both without duplicating it.

    Why relative: sweep_results[0] already reflects whatever accuracy
    is lost to (a) CLIP's own zero-shot quality on this class and
    (b) the SAE's reconstruction error, before any editing happens.
    Measuring against absolute accuracy would credit the edit for
    "forgetting" that the classifier already exhibited beforehand --
    exactly the failure mode you'd hit on fine-grained classes CLIP
    already struggles with (e.g. dog breeds at top-1). Clamped at 0
    since accuracy occasionally ticks up slightly from baseline noise,
    which isn't "forgetting".

    Higher AUC = the edit achieves the intended forgetting while
    keeping everything else intact; lower AUC = forgetting and
    collateral damage move together, i.e. you can't have one without
    the other.
    """
    forget_key = f"forget_class_accuracy{metric_suffix}"
    retain_key = f"retain_accuracy_overall{metric_suffix}"

    baseline_forget_accuracy = sweep_results[0][forget_key]

    x = np.array(
        [max(0.0, baseline_forget_accuracy - r[forget_key]) for r in sweep_results]
    )
    y = np.array([r[retain_key] for r in sweep_results])
    order = np.argsort(x)

    # np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed in
    # later 2.x releases; support both without pinning a NumPy version.
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid_fn(y[order], x[order]))


# ======================================================================
# Matrix loading
# ======================================================================


def load_feature_class_matrix(
    model,
    precomputed_matrix: str | None,
    rho: float,
    device: torch.device,
):
    """
    Resolve the feature-class matrix M and per-feature budget vector k.

    Priority:
      1. If the model is a ClassAlignedSAE and no precomputed matrix is
         given, pull M and k directly from the model's internal state --
         no training data required.
      2. Otherwise load a precomputed empirical matrix from disk and
         derive M / k via the post-hoc thresholding procedure.

    Raises ValueError if neither condition can be satisfied.
    """
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

    M = M.to(device, dtype=torch.float32)
    k = k.to(device, dtype=torch.float32)
    return M, k


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
    num_random_control_classes: int,
    min_baseline_accuracy: float,
    rho: float,
    orders: list[str],
    strength_modes: list[str],
    num_sweep_points: int,
    distance_bin_edges: list[int],
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

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = SUPPORTED_ARCHITECTURES[architecture].from_pretrained(
        checkpoint_path, device=device
    )
    model.eval()
    d = model.dict_size

    # ------------------------------------------------------------
    # Feature-class matrix M and per-feature budget k
    # ------------------------------------------------------------
    M, k = load_feature_class_matrix(
        model=model,
        precomputed_matrix=precomputed_matrix,
        rho=rho,
        device=device,
    )
    row_entropy = compute_row_entropy(M)

    # ------------------------------------------------------------
    # Classes, synsets, WordNet distances, zero-shot weights
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Target class selection
    # ------------------------------------------------------------
    if target_classes is not None:
        targets = target_classes
    elif hypernym_cluster is not None:
        print(f"Selecting target classes under hypernym {hypernym_cluster}...")
        targets = classes_under_hypernym(wnids, hypernym_cluster)
    else:
        targets = []

    if num_random_control_classes > 0:
        pool = [i for i in range(1000) if i not in targets]
        targets = targets + rng.sample(pool, min(num_random_control_classes, len(pool)))

    print(f"Evaluating {len(targets)} target classes: {targets}")

    # ------------------------------------------------------------
    # Load test data once
    # ------------------------------------------------------------
    print("Loading test activations...")
    x_all, labels_all = load_all_activations(
        test_activations_path, batch_size, num_workers
    )
    if max_test_examples is not None:
        x_all = x_all[:max_test_examples]
        labels_all = labels_all[:max_test_examples]

    # ------------------------------------------------------------
    # Raw-CLIP-only baseline (no SAE at all), per target class, top-1
    # and top-5. Flags classes where CLIP itself already can't
    # reliably tell this class apart -- "forgetting" isn't a
    # meaningful concept there, and including them would let
    # pre-existing weakness masquerade as successful erasure. The
    # reliability gate uses top-1 (the stricter bar); top-5 is carried
    # along purely as a diagnostic to distinguish "CLIP has no idea"
    # from "CLIP is confusing within a small, plausible cluster".
    # ------------------------------------------------------------
    print("Computing raw zero-shot CLIP baseline (no SAE) per target class...")
    raw_baseline = compute_raw_zeroshot_baseline(
        x_all, labels_all, zeroshot_weights, targets, chunk_size, device
    )

    unreliable = [
        c
        for c in targets
        if raw_baseline.get(c) is not None
        and raw_baseline[c]["top1"] < min_baseline_accuracy
    ]
    if unreliable:
        print(
            f"[warn] {len(unreliable)} target classes have raw zero-shot CLIP "
            f"top-1 accuracy below {min_baseline_accuracy:.2f} before any editing -- "
            f"these are flagged 'reliable': false below and excluded from the "
            f"collateral-correlation summary, since 'forgetting' isn't well-defined "
            f"when the classifier already couldn't recognize the class:"
        )
        for c in unreliable:
            b = raw_baseline[c]
            print(
                f"    class {c} ({wnids[c]}): "
                f"raw top1 = {b['top1']:.3f}, raw top5 = {b['top5']:.3f}"
            )

    # ------------------------------------------------------------
    # Main sweep loop
    # ------------------------------------------------------------
    all_results = {}
    collateral_predictors = (
        []
    )  # sum_{i in F_c} k_i, per target class (hard mode, full ablation)
    collateral_observed = []  # 1 - retain_accuracy_overall at full ablation (hard mode)

    for c in tqdm(targets, desc="Target classes"):
        f_c = (M[:, c] > 0).nonzero(as_tuple=True)[0]
        if len(f_c) == 0:
            print(
                f"[warn] class {c} ({wnids[c]}) has no claiming features under this M; skipping."
            )
            continue

        c_baseline = raw_baseline.get(c)

        all_results[c] = {
            "wnid": wnids[c],
            "num_claiming_features": len(f_c),
            "raw_clip_baseline_accuracy_top1": (
                c_baseline["top1"] if c_baseline else None
            ),
            "raw_clip_baseline_accuracy_top5": (
                c_baseline["top5"] if c_baseline else None
            ),
            "reliable": c_baseline is not None
            and c_baseline["top1"] >= min_baseline_accuracy,
            # filled in below, from sweep[0]
            "sae_roundtrip_baseline_forget_accuracy_top1": None,
            "sae_roundtrip_baseline_forget_accuracy_top5": None,
            "by_setting": {},
        }

        for order in orders:
            feature_order = build_feature_order(f_c, k, order, rng)

            for mode in strength_modes:
                strength_full = compute_ablation_strength(M, row_entropy, c, mode)

                sweep = run_sweep_for_setting(
                    model=model,
                    x_all=x_all,
                    labels_all=labels_all,
                    zeroshot_weights=zeroshot_weights,
                    distance_row=dist_matrix[c],
                    bin_edges=distance_bin_edges,
                    strength_full=strength_full,
                    feature_order=feature_order,
                    target_class=c,
                    num_sweep_points=num_sweep_points,
                    chunk_size=chunk_size,
                    device=device,
                )

                key = f"{order}__{mode}"
                all_results[c]["by_setting"][key] = {
                    "sweep": sweep,
                    "forget_retain_auc": forget_retain_auc(sweep),
                    "forget_retain_auc_top5": forget_retain_auc(
                        sweep, metric_suffix="_top5"
                    ),
                }

                # sweep[0] is always N=0 (no ablation): the SAE's own
                # round-trip baseline, isolated from the raw-CLIP
                # baseline above. Only needs recording once per class.
                if (
                    all_results[c]["sae_roundtrip_baseline_forget_accuracy_top1"]
                    is None
                ):
                    all_results[c]["sae_roundtrip_baseline_forget_accuracy_top1"] = (
                        sweep[0]["forget_class_accuracy"]
                    )
                    all_results[c]["sae_roundtrip_baseline_forget_accuracy_top5"] = (
                        sweep[0]["forget_class_accuracy_top5"]
                    )

                if (
                    order == "specialist_first"
                    and mode == "hard"
                    and all_results[c]["reliable"]
                ):
                    collateral_predictors.append(float(k[f_c].sum().item()))
                    collateral_observed.append(
                        1.0 - sweep[-1]["retain_accuracy_overall"]
                    )

    # ------------------------------------------------------------
    # Predicted (sum k_i over F_c) vs. observed collateral correlation
    # ------------------------------------------------------------
    if len(collateral_predictors) >= 2:
        corr = float(np.corrcoef(collateral_predictors, collateral_observed)[0, 1])
    else:
        corr = None

    summary = {
        "architecture": architecture,
        "rho": rho,
        "num_target_classes": len(all_results),
        "orders": orders,
        "strength_modes": strength_modes,
        "num_sweep_points": num_sweep_points,
        "distance_bin_edges": distance_bin_edges,
        "predicted_vs_observed_collateral_correlation": corr,
        "predicted_collateral_values": collateral_predictors,
        "observed_collateral_values": collateral_observed,
        "results_by_class": all_results,
    }

    print("\n=== Unlearning / Concept-Erasure Evaluation Summary ===")
    print(f"Target classes evaluated:              {len(all_results)}")
    print(
        f"  ...of which flagged unreliable:      {sum(1 for r in all_results.values() if not r['reliable'])}"
        f" (raw CLIP top-1 baseline < {min_baseline_accuracy:.2f})"
    )
    print(
        f"Predicted-vs-observed collateral corr:  {corr}  (reliable classes only, n={len(collateral_predictors)})"
    )
    for c, r in all_results.items():
        flag = "" if r["reliable"] else "  [UNRELIABLE BASELINE]"
        print(
            f"  class {c:4d} ({r['wnid']}) "
            f"raw_clip_baseline(top1/top5)={r['raw_clip_baseline_accuracy_top1']:.3f}/"
            f"{r['raw_clip_baseline_accuracy_top5']:.3f} "
            f"sae_roundtrip_baseline(top1/top5)={r['sae_roundtrip_baseline_forget_accuracy_top1']:.3f}/"
            f"{r['sae_roundtrip_baseline_forget_accuracy_top5']:.3f}{flag}"
        )
        for key, v in r["by_setting"].items():
            print(
                f"      [{key:35s}] forget-retain AUC (top1/top5) = "
                f"{v['forget_retain_auc']:.4f} / {v['forget_retain_auc_top5']:.4f}"
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return summary


def cli():
    parser = argparse.ArgumentParser(
        description="Unlearning/concept-erasure evaluation via SoftSAE-CA's feature-class matrix M."
    )

    parser.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=list(SUPPORTED_ARCHITECTURES.keys()),
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--test-activations-path", required=True)

    parser.add_argument(
        "--precomputed-matrix",
        default=None,
        help=(
            "Path to a precomputed empirical feature-class matrix (.pt). "
            "Required when the model is not a ClassAlignedSAE; omit to use "
            "the model's built-in M when it is a ClassAlignedSAE."
        ),
    )

    parser.add_argument(
        "--zeroshot-weights-path",
        required=True,
        help="Path to load/save the precomputed [1000, D] zero-shot text embeddings.",
    )
    parser.add_argument("--wordnet-distance-cache", default=None)

    target_args = parser.add_mutually_exclusive_group(required=False)
    target_args.add_argument(
        "--target-classes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit class indices to erase.",
    )
    target_args.add_argument(
        "--hypernym-cluster",
        type=str,
        default=None,
        help="wnid whose descendants form the target-class cluster, e.g. n02084071 (dog).",
    )

    parser.add_argument(
        "--num-random-control-classes",
        type=int,
        default=10,
        help="Additional randomly chosen, unrelated classes to erase as an easy-regime control.",
    )
    parser.add_argument(
        "--min-baseline-accuracy",
        type=float,
        default=0.2,
        help=(
            "Target classes whose raw (no-SAE) top-1 zero-shot CLIP accuracy falls "
            "below this are flagged 'reliable': false and excluded from the collateral-"
            "correlation summary, since 'forgetting' isn't well-defined for a class "
            "CLIP couldn't reliably recognize in the first place."
        ),
    )

    parser.add_argument("--rho", type=float, default=5.0)

    parser.add_argument(
        "--orders",
        nargs="+",
        default=["specialist_first", "random"],
        choices=["specialist_first", "random"],
    )
    parser.add_argument(
        "--strength-modes",
        nargs="+",
        default=["hard", "graded_association", "entropy_weighted"],
        choices=["hard", "graded_association", "entropy_weighted"],
    )
    parser.add_argument("--num-sweep-points", type=int, default=8)
    parser.add_argument("--distance-bin-edges", type=int, nargs="+", default=[2, 4, 6])

    parser.add_argument("--clip-model-name", default="ViT-B/32")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Batch size for the encode/edit/decode/classify inner loop.",
    )

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
        num_random_control_classes=args.num_random_control_classes,
        min_baseline_accuracy=args.min_baseline_accuracy,
        rho=args.rho,
        orders=args.orders,
        strength_modes=args.strength_modes,
        num_sweep_points=args.num_sweep_points,
        distance_bin_edges=args.distance_bin_edges,
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
