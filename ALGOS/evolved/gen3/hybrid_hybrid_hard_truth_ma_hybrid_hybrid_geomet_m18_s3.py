# DARWIN HAMMER — match 18, survivor 3
# gen: 3
# parent_a: hybrid_hard_truth_math_model_pool_m8_s5.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s4.py (gen2)
# born: 2026-05-29T23:25:07Z

"""Hybrid Stylometric‑Geometric Model
===================================

This module fuses two previously independent algorithms:

* **Parent A** – a stylometric extractor that builds a high‑dimensional
  frequency fingerprint (`stylometry_features`) from textual data.
* **Parent B** – a geometric toolkit that constructs Voronoi partitions of
  n‑dimensional points and provides a minimal Clifford‑algebra blade
  implementation (`_canonical_blade`, `_multiply_blades`).

The mathematical bridge is the observation that a stylometric fingerprint is
simply a point in ℝⁿ (n = 96 in the original implementation).  By treating each
fingerprint as a geometric point we can:

1. Use Euclidean distance to assign texts to the Voronoi cell of the nearest
   seed (``voronoi_partition``).
2. Map the presence of linguistic function‑category tokens onto a Clifford
   blade (one basis vector per category) and combine blades with the geometric
   operations of Parent B.

The resulting hybrid functions expose a unified workflow:
text → vector → geometric region & algebraic signature.

Only the Python standard library and NumPy are required.
"""

import re
import hashlib
import math
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, FrozenSet, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS = {
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
    """Deterministic 48‑bit integer hash used for seed generation."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a ``dim``‑dimensional stylometric fingerprint.
    The original implementation used 96 dimensions; we obtain a base vector
    of length ``len(FUNCTION_CATS)`` (8) and tile it to reach the requested size.
    """
    base = np.array(list(lsm_vector(text).values()), dtype=float)
    if dim % base.size != 0:
        # pad with zeros to the nearest multiple
        pad_len = dim - (dim % base.size)
        base = np.concatenate([base, np.zeros(pad_len, dtype=float)])
    repeats = dim // base.size
    return np.tile(base, repeats)


# ----------------------------------------------------------------------
# Parent B – geometric / Clifford‑algebra utilities
# ----------------------------------------------------------------------
Point = Tuple[float, ...]  # n‑dimensional point


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_point(point: Point, seeds: List[Point]) -> int:
    """Return the index of the seed closest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean_distance(point, seeds[i]), i))


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
    # Cancel even repetitions
    for i in list(cnt):
        if cnt[i] % 2 == 0:
            del cnt[i]
        else:
            cnt[i] = 1
    distinct = sorted(cnt.elements())

    # Compute sign of the permutation that sorts the original (uncancelled) list
    sign = 1
    original = [i for i in indices if cnt.get(i, 0) == 1]
    # Simple bubble‑sort sign computation
    for i in range(len(original)):
        for j in range(len(original) - 1 - i):
            if original[j] > original[j + 1]:
                original[j], original[j + 1] = original[j + 1], original[j]
                sign = -sign
    return frozenset(distinct), sign


def _multiply_blades(a: FrozenSet[int], b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """
    Multiply two blades using the Euclidean Clifford product.
    The result is a new blade together with an overall sign.
    """
    # Concatenate the index lists, then canonicalise
    combined = list(a) + list(b)
    return _canonical_blade(combined)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def text_to_point(text: str, dim: int = 96) -> np.ndarray:
    """
    Convert a piece of text into a geometric point (numpy vector) using the
    stylometric fingerprint from Parent A.
    """
    return stylometry_features(text, dim=dim)


def build_voronoi_from_corpus(
    corpus_texts: List[str],
    seed_texts: List[str],
    dim: int = 96,
) -> Dict[int, List[str]]:
    """
    Create a Voronoi partition of *corpus_texts* where each seed is the
    stylometric point of a *seed_text*.  Returns a mapping from seed index to the
    list of corpus texts that fall into that cell.
    """
    seeds = [tuple(text_to_point(t, dim)) for t in seed_texts]
    points = [tuple(text_to_point(t, dim)) for t in corpus_texts]
    region_points = voronoi_partition(seeds, points)

    # Translate back from points to the original strings
    point_to_text = {p: txt for p, txt in zip(points, corpus_texts)}
    return {
        idx: [point_to_text[p] for p in region_points[idx]]
        for idx in region_points
    }


def text_blade_signature(text: str) -> Tuple[FrozenSet[int], int]:
    """
    Encode the presence of function‑category tokens as a Clifford blade.
    Each category receives a unique basis index (0‑7).  The blade is the
    canonicalised wedge product of the indices whose category occurs at least
    once in the text.
    """
    present_cats = {
        idx
        for idx, cat in enumerate(FUNCTION_CATS)
        if any(tok in FUNCTION_CATS[cat] for tok in words(text))
    }
    return _canonical_blade(present_cats)


def hybrid_distance(text_a: str, text_b: str, dim: int = 96) -> float:
    """
    A composite distance that blends:
    * Euclidean distance between stylometric vectors.
    * A penalty of 0 or 2 depending on whether the blade signatures have the same sign.
    The result is still a metric‑like scalar suitable for nearest‑seed queries.
    """
    v_a = text_to_point(text_a, dim)
    v_b = text_to_point(text_b, dim)
    euclid = euclidean_distance(tuple(v_a), tuple(v_b))

    blade_a, sign_a = text_blade_signature(text_a)
    blade_b, sign_b = text_blade_signature(text_b)

    # If the blades are identical (including sign) no penalty; otherwise add 2.
    blade_penalty = 0.0 if (blade_a == blade_b and sign_a == sign_b) else 2.0
    return euclid + blade_penalty


def multiply_text_blades(text_a: str, text_b: str) -> Tuple[FrozenSet[int], int]:
    """
    Multiply the Clifford blades derived from two texts.
    This demonstrates the algebraic side of the hybrid model.
    """
    blade_a, _ = text_blade_signature(text_a)
    blade_b, _ = text_blade_signature(text_b)
    return _multiply_blades(blade_a, blade_b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_corpus = [
        "I think therefore I am.",
        "The quick brown fox jumps over the lazy dog!",
        "She couldn't understand why the results were not significant.",
        "We shall overcome the challenges of the future.",
        "Never forget that the universe is vast and mysterious."
    ]

    seed_corpus = [
        "I love programming and mathematics.",
        "Nature provides endless inspiration."
    ]

    # 1. Build Voronoi partition based on stylometric points
    regions = build_voronoi_from_corpus(sample_corpus, seed_corpus, dim=96)
    print("Voronoi regions (seed index -> texts):")
    for idx, texts in regions.items():
        print(f" Seed {idx}:")
        for t in texts:
            print(f"   • {t}")

    # 2. Compute hybrid distances between a pair of sentences
    d = hybrid_distance(sample_corpus[0], sample_corpus[2])
    print(f"\nHybrid distance between sentence 0 and 2: {d:.4f}")

    # 3. Multiply their algebraic signatures
    blade_res, sign_res = multiply_text_blades(sample_corpus[0], sample_corpus[2])
    print(f"\nBlade multiplication result: indices={sorted(blade_res)}, sign={sign_res}")

    # Simple sanity check that the code runs without exception
    sys.exit(0)