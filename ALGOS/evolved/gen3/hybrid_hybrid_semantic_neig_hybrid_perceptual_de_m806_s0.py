# DARWIN HAMMER — match 806, survivor 0
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# parent_b: hybrid_perceptual_dedupe_hybrid_privacy_sketc_m292_s0.py (gen2)
# born: 2026-05-29T23:31:13Z

"""Hybrid Semantic‑Morphology & Perceptual‑Privacy System
Parents:
- hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (semantic similarity + morphology recovery priority)
- hybrid_perceptual_dedupe_hybrid_privacy_sketc_m292_s0.py (perceptual hashing + count‑min sketch privacy risk)

Mathematical Bridge:
The reconstruction‑risk score *r* (0‑1) derived from the Count‑Min sketch is used to
modulate the morphology‑derived recovery priority *p* before it is combined with the
semantic cosine similarity *c*. The unified hybrid neighbor score is

    h(i, j) = α·c(v_i,v_j) + (1‑α)·p(m_j)·(1‑r)

where α∈[0,1] balances pure semantic closeness against a privacy‑aware physical robustness.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent A – part B)
# ----------------------------------------------------------------------
class Morphology:
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Semantic similarity (Parent A – part A)
# ----------------------------------------------------------------------
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two 1‑D vectors."""
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("Both inputs must be 1‑D arrays")
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ----------------------------------------------------------------------
# Perceptual hashing & privacy sketch (Parent B)
# ----------------------------------------------------------------------
def compute_ph_dhash(values: List[float]) -> int:
    """Perceptual hash‑lite (difference hash) of a numeric sequence."""
    dhash = 0
    for i in range(len(values) - 1):
        dhash = (dhash << 1) | int(values[i] > values[i + 1])
    return dhash


def compute_ph_phash(values: List[float]) -> int:
    """Perceptual hash‑lite (average hash) of a numeric sequence."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    phash = 0
    for v in values[:64]:
        phash = (phash << 1) | int(v >= avg)
    return phash


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Count‑Min sketch for a multiset of string identifiers."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def estimate_unique_quasi_identifiers(sketch: List[List[int]], width: int, depth: int) -> int:
    """Estimate the number of distinct identifiers using the sketch."""
    estimates = [sum(1 for cnt in row if cnt > 0) for row in sketch[:depth]]
    return int(np.mean(estimates))


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk score ∈[0,1] = unique / total (clipped)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


# ----------------------------------------------------------------------
# Hybrid operations (newly invented)
# ----------------------------------------------------------------------
def hybrid_neighbor_score(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    morph_j: Morphology,
    sketch: List[List[int]],
    total_records: int,
    alpha: float = 0.5,
) -> float:
    """
    Unified hybrid score h(i,j) = α·c + (1‑α)·p·(1‑r).

    Parameters
    ----------
    vec_i, vec_j : np.ndarray
        Semantic feature vectors of the two items.
    morph_j : Morphology
        Physical morphology of the candidate neighbour (j).
    sketch : List[List[int]]
        Count‑Min sketch built from quasi‑identifiers of the whole dataset.
    total_records : int
        Size of the dataset (used for risk normalisation).
    alpha : float
        Balance factor between semantic similarity and morphology priority.

    Returns
    -------
    float
        Hybrid neighbor score in [0,1].
    """
    # 1. Semantic term
    c = cosine_similarity(vec_i, vec_j)

    # 2. Morphology term
    p = recovery_priority(morph_j)

    # 3. Privacy risk term
    uniq = estimate_unique_quasi_identifiers(sketch, width=len(sketch[0]), depth=len(sketch))
    r = reconstruction_risk_score(uniq, total_records)

    # 4. Convex combination with privacy modulation
    hybrid = alpha * c + (1.0 - alpha) * p * (1.0 - r)
    return max(0.0, min(1.0, hybrid))


def dedupe_similar_records(
    records: List[List[float]],
    threshold: int = 5,
    alpha: float = 0.6,
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of records that are both perceptually similar (small Hamming distance)
    and hybrid‑neighbor close (high hybrid score).

    Returns a list of tuples (idx_i, idx_j, hybrid_score).
    """
    n = len(records)
    # Build perceptual hashes (average hash) for all records
    phashes = [compute_ph_phash(rec) for rec in records]

    # Build a sketch from stringified records (as quasi‑identifiers)
    identifiers = [",".join(f"{v:.2f}" for v in rec) for rec in records]
    sketch = count_min_sketch(identifiers)

    results: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            # Hamming filter
            if hamming_distance(phashes[i], phashes[j]) > threshold:
                continue
            # Random morphology for demo purposes
            morph_j = Morphology(
                length=random.uniform(0.5, 2.0),
                width=random.uniform(0.5, 2.0),
                height=random.uniform(0.5, 2.0),
                mass=random.uniform(0.1, 5.0),
            )
            vec_i = np.array(records[i])
            vec_j = np.array(records[j])
            score = hybrid_neighbor_score(
                vec_i, vec_j, morph_j, sketch, total_records=n, alpha=alpha
            )
            if score > 0.5:  # arbitrary acceptance cut
                results.append((i, j, score))
    return results


def privacy_aware_morphology_analysis(
    morphologies: List[Morphology],
    quasi_identifiers: List[str],
    alpha: float = 0.4,
) -> List[float]:
    """
    Produce a list of privacy‑adjusted recovery priorities for a collection of morphologies.

    Each priority is reduced proportionally to the reconstruction risk derived from the
    provided quasi‑identifiers.
    """
    sketch = count_min_sketch(quasi_identifiers)
    uniq = estimate_unique_quasi_identifiers(sketch, width=len(sketch[0]), depth=len(sketch))
    risk = reconstruction_risk_score(uniq, total_records=len(quasi_identifiers))

    adjusted = []
    for morph in morphologies:
        p = recovery_priority(morph)
        # blend with a dummy semantic term (here set to 0) to keep the same formula shape
        adjusted_score = alpha * 0.0 + (1.0 - alpha) * p * (1.0 - risk)
        adjusted.append(max(0.0, min(1.0, adjusted_score)))
    return adjusted


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    random.seed(42)
    np.random.seed(42)

    # 10 records, each a 8‑dimensional float vector
    records = [list(np.random.rand(8)) for _ in range(10)]

    # Run deduplication + hybrid scoring
    pairs = dedupe_similar_records(records, threshold=8, alpha=0.7)
    print("Hybrid similar pairs (i, j, score):")
    for i, j, s in pairs:
        print(f"  ({i}, {j}) → {s:.3f}")

    # Morphology analysis with privacy awareness
    morphs = [
        Morphology(
            length=random.uniform(0.5, 3.0),
            width=random.uniform(0.5, 3.0),
            height=random.uniform(0.5, 3.0),
            mass=random.uniform(0.2, 10.0),
        )
        for _ in range(5)
    ]
    # Fake quasi‑identifiers (e.g., hashed user attributes)
    qids = [f"user{idx}_{random.randint(0,1000)}" for idx in range(20)]

    adjusted_priorities = privacy_aware_morphology_analysis(morphs, qids, alpha=0.3)
    print("\nPrivacy‑aware recovery priorities:")
    for idx, val in enumerate(adjusted_priorities):
        print(f"  Morphology {idx}: {val:.3f}")

    sys.exit(0)