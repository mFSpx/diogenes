# DARWIN HAMMER — match 18, survivor 4
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:25:07Z

"""Hybrid Stylometric‑Geometric Model
===================================

This module fuses two previously independent algorithms:

* **Parent A** – ``hard_truth_math.py``: extracts a low‑dimensional
  stylometric fingerprint from raw text (frequency of functional word
  categories) and provides a deterministic hash for indexing.
* **Parent B** – ``hybrid_geometric_product_voronoi_partition``: treats
  points in ℝⁿ, builds a Voronoi diagram from a set of seed points and
  defines a Clifford‑algebra‑style geometric product on *blades* (ordered
  sets of basis indices).

The mathematical bridge is the observation that a stylometric fingerprint
is naturally a point in a Euclidean vector space.  By interpreting each
fingerprint as an *n‑dimensional point* we can:

1. **Partition** a collection of texts into Voronoi cells using a few
   fingerprints as seeds.
2. **Combine** the categorical information of texts inside each cell with
   a *geometric product* on blades whose basis vectors correspond to the
   functional‑word categories.

The resulting hybrid system yields, for every Voronoi region, a centroid
vector (average stylometry) together with a single blade that algebraically
encodes the distribution of categories inside the region.

The public API consists of three core functions demonstrating the hybrid
operation:

* ``stylometry_features`` – compute a normalized category vector.
* ``voronoi_partition`` – assign fingerprint points to the nearest seed.
* ``region_blade_product`` – map texts to blades and multiply them per
  region using the Clifford‑algebra product.

A small smoke test is provided under ``if __name__ == "__main__"``
demonstrating end‑to‑end usage."""

import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities (adapted)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Return a list of lowercase words (ASCII letters + optional apostrophe)."""
    import re

    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Return a normalized frequency vector for each FUNCTION_CATS."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def stable_hash(text: str) -> int:
    """Deterministic hash used for trigram indexing."""
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a ``dim``‑dimensional stylometric fingerprint.
    The first eight entries correspond to the categories in ``FUNCTION_CATS``;
    remaining entries are zero‑padded.
    """
    vec = np.zeros(dim, dtype=float)
    cat_vec = lsm_vector(text)
    # Preserve a deterministic ordering of categories
    ordered_cats = list(FUNCTION_CATS.keys())
    for i, cat in enumerate(ordered_cats):
        if i >= dim:
            break
        vec[i] = cat_vec.get(cat, 0.0)
    # Normalise to unit length (optional, improves Euclidean behaviour)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ----------------------------------------------------------------------
# Parent B – geometric / Clifford algebra utilities (adapted)
# ----------------------------------------------------------------------
Point = Tuple[float, ...]  # n‑dimensional point


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(
        range(len(seeds)),
        key=lambda i: (euclidean_distance(point, seeds[i]), i),
    )


def voronoi_partition(seeds: List[Point], points: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to the Voronoi cell of the nearest seed."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest_point(p, seeds)].append(p)
    return regions


def _canonical_blade(indices: Iterable[int]) -> Tuple[FrozenSet[int], int]:
    """
    Return a canonical (sorted, duplicate‑free) blade together with the sign
    incurred by re‑ordering the basis vectors.

    In Euclidean Clifford algebra e_i^2 = 1, therefore duplicate indices cancel
    pairwise without affecting the sign.
    """
    cnt = Counter(indices)
    # Remove even duplicates
    for i in list(cnt):
        if cnt[i] % 2 == 0:
            del cnt[i]
        else:
            cnt[i] = 1
    distinct = sorted(cnt.elements())
    # Compute sign of permutation that sorts the original kept list
    original = [i for i in indices if cnt.get(i, 0) == 1]
    sign = 1
    # Simple bubble‑sort count
    for i in range(len(original)):
        for j in range(len(original) - 1 - i):
            if original[j] > original[j + 1]:
                original[j], original[j + 1] = original[j + 1], original[j]
                sign = -sign
    return frozenset(distinct), sign


def _multiply_blades(a: FrozenSet[int], b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two blades in Euclidean Clifford algebra.
    The result is another blade with a sign determined by the ordering
    required to bring the concatenated index list into canonical form.
    """
    # Concatenate the two index sequences (order matters)
    concatenated = list(a) + list(b)
    return _canonical_blade(concatenated)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def text_to_blade(text: str) -> FrozenSet[int]:
    """
    Map a text to a blade whose basis vectors correspond to the functional
    categories that have a non‑zero frequency in the stylometric vector.
    The category order is the same as in ``FUNCTION_CATS``; the index of a
    category is its position in that order.
    """
    cat_vec = lsm_vector(text)
    ordered = list(FUNCTION_CATS.keys())
    indices = [i for i, cat in enumerate(ordered) if cat_vec.get(cat, 0.0) > 0.0]
    return frozenset(indices)


def region_blade_product(
    texts: List[str],
) -> Tuple[FrozenSet[int], int]:
    """
    Multiply the blades of all texts in *texts* using the geometric product.
    Returns the resulting blade (as a frozenset of basis indices) and the
    accumulated sign (+1 or –1).
    """
    if not texts:
        return frozenset(), 1
    # Initialise with the blade of the first text
    result_blade = text_to_blade(texts[0])
    sign = 1
    for txt in texts[1:]:
        next_blade = text_to_blade(txt)
        result_blade, s = _multiply_blades(result_blade, next_blade)
        sign *= s
    return result_blade, sign


def hybrid_analysis(
    texts: List[str],
    seed_count: int = 2,
) -> Dict[int, Dict[str, Any]]:
    """
    End‑to‑end hybrid pipeline:

    1. Convert each text into a stylometric point (96‑dim numpy array).
    2. Choose ``seed_count`` points as Voronoi seeds (deterministically by
       stable hash order).
    3. Partition all points into Voronoi cells.
    4. For each cell compute:
       * ``centroid`` – average stylometric vector (numpy array).
       * ``blade``   – geometric product of the blades of the texts belonging
         to the cell.
    Returns a dictionary keyed by region index.
    """
    if not texts:
        return {}

    # 1. Stylometric points
    points = [tuple(stylometry_features(t).tolist()) for t in texts]

    # 2. Deterministic seed selection using stable_hash
    hashed = [(stable_hash(t), idx) for idx, t in enumerate(texts)]
    hashed.sort()  # ascending hash order
    seed_indices = [idx for _, idx in hashed[:seed_count]]
    seeds = [points[i] for i in seed_indices]

    # 3. Voronoi partition
    regions = voronoi_partition(seeds, points)

    # 4. Assemble results
    results: Dict[int, Dict[str, Any]] = {}
    for region_idx, region_points in regions.items():
        if not region_points:
            continue
        # Centroid
        arr = np.array(region_points, dtype=float)
        centroid = arr.mean(axis=0)

        # Retrieve original texts that belong to this region
        # (by matching point tuples – safe because we used deterministic vectors)
        belonging_texts = [
            txt
            for txt, pt in zip(texts, points)
            if pt in region_points
        ]

        blade, blade_sign = region_blade_product(belonging_texts)

        results[region_idx] = {
            "centroid": centroid,
            "blade": blade,
            "blade_sign": blade_sign,
            "texts": belonging_texts,
        }
    return results


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think therefore I am.",
        "The quick brown fox jumps over the lazy dog.",
        "She doesn't like apples, but she loves oranges.",
        "We will meet at 5pm tomorrow, if you are available.",
        "All that glitters is not gold.",
        "Never gonna give you up, never gonna let you down.",
        "Can you believe it? It's amazing!",
        "Without doubt, the experiment succeeded.",
    ]

    analysis = hybrid_analysis(sample_texts, seed_count=3)

    for region, data in analysis.items():
        print(f"Region {region}:")
        print(f"  Centroid (first 8 dims): {data['centroid'][:8]}")
        print(f"  Blade indices: {sorted(data['blade'])}")
        print(f"  Blade sign: {data['blade_sign']}")
        print(f"  Text count: {len(data['texts'])}")
        print()
    sys.exit(0)