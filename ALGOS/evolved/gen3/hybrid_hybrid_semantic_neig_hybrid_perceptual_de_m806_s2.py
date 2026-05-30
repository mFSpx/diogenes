# DARWIN HAMMER — match 806, survivor 2
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# parent_b: hybrid_perceptual_dedupe_hybrid_privacy_sketc_m292_s0.py (gen2)
# born: 2026-05-29T23:31:13Z

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent A – part B)
# ----------------------------------------------------------------------
class Morphology:
    """Physical description of an object."""
    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (unit‑less)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Geometric flatness (unit‑less)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical proxy for how quickly an object can right itself."""
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def morphology_similarity(m1: Morphology, m2: Morphology) -> float:
    """Similarity of two morphologies based on normalised Euclidean distance."""
    vec1 = np.array([m1.length, m1.width, m1.height, m1.mass])
    vec2 = np.array([m2.length, m2.width, m2.height, m2.mass])
    diff = np.linalg.norm(vec1 - vec2)
    # Scale by a reasonable upper bound (here 5× the max possible range)
    scale = 5.0 * np.max(vec1)
    return max(0.0, 1.0 - diff / scale)


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
def compute_phash(values: List[float], bits: int = 64) -> int:
    """
    Average‑hash style perceptual hash for a numeric sequence.
    Uses the full sequence (up to ``bits`` elements) and returns an ``int``.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    phash = 0
    for v in values[:bits]:
        phash = (phash << 1) | int(v >= avg)
    # Pad remaining bits with zeros if the sequence is shorter than ``bits``.
    phash <<= max(0, bits - len(values))
    return phash


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


class CountMinSketch:
    """
    Simple Count‑Min sketch with deterministic hash functions.
    Provides both insertion and point‑query (frequency estimate).
    """

    def __init__(self, width: int = 256, depth: int = 4):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width = width
        self.depth = depth
        self.table: List[List[int]] = [[0] * width for _ in range(depth)]

    def _hash(self, item: str, d: int) -> int:
        """Deterministic hash for row ``d``."""
        digest = hashlib.sha256(f"{d}:{item}".encode()).digest()
        # Use first 8 bytes as an integer, then modulo width
        return int.from_bytes(digest[:8], "big") % self.width

    def add(self, item: str, count: int = 1) -> None:
        """Insert ``count`` copies of ``item``."""
        for d in range(self.depth):
            idx = self._hash(item, d)
            self.table[d][idx] += count

    def estimate(self, item: str) -> int:
        """Return the minimum count across rows (standard CMS estimator)."""
        return min(self.table[d][self._hash(item, d)] for d in range(self.depth))

    def distinct_estimate(self) -> int:
        """
        Rough distinct count using the linear‑counting heuristic on each row.
        This is cheaper than HyperLogLog and works reasonably for moderate widths.
        """
        estimates = []
        for row in self.table:
            zero_cnt = row.count(0)
            if zero_cnt == 0:
                # All cells occupied → fallback to width
                estimates.append(self.width)
            else:
                est = -self.width * math.log(zero_cnt / self.width)
                estimates.append(est)
        return int(np.mean(estimates))


# ----------------------------------------------------------------------
# Privacy‑aware risk computation
# ----------------------------------------------------------------------
def reconstruction_risk(freq: int, total: int) -> float:
    """
    Risk score ∈[0,1] based on how unique an identifier is.
    A low frequency (especially 1) yields high risk.
    """
    if total <= 0:
        return 0.0
    # Inverse frequency normalised: 1 → max risk, total → min risk
    prob_unique = max(0.0, 1.0 - (freq - 1) / (total - 1)) if total > 1 else 1.0
    return max(0.0, min(1.0, prob_unique))


# ----------------------------------------------------------------------
# Deeply integrated hybrid score
# ----------------------------------------------------------------------
def hybrid_neighbor_score(
    vec_i: np.ndarray,
    vec_j: np.ndarray,
    morph_i: Morphology,
    morph_j: Morphology,
    sketch: CountMinSketch,
    identifier_j: str,
    total_records: int,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """
    Deep hybrid score combining:

    * semantic similarity (c)
    * morphology similarity (m_sim)
    * morphology‑derived priority (p_j)
    * privacy risk of the candidate (r_j)

    The formula is a convex combination of three terms:

        h = α·c + β·m_sim·p_j·(1‑r_j) + (1‑α‑β)·p_j·(1‑r_j)

    where ``α,β∈[0,1]`` and ``α+β≤1``.
    This gives the model flexibility to weight pure semantics,
    joint morphology similarity, and privacy‑aware priority independently.
    """
    if not (0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and alpha + beta <= 1.0):
        raise ValueError("alpha and beta must satisfy 0≤α,β and α+β≤1")

    # 1. Semantic term
    c = cosine_similarity(vec_i, vec_j)

    # 2. Morphology similarity and priority
    m_sim = morphology_similarity(morph_i, morph_j)
    p_j = recovery_priority(morph_j)

    # 3. Privacy risk for the candidate neighbour
    freq_j = sketch.estimate(identifier_j)
    r_j = reconstruction_risk(freq_j, total_records)

    # 4. Combine
    term_sem = alpha * c
    term_morph_sim = beta * m_sim * p_j * (1.0 - r_j)
    term_morph_prio = (1.0 - alpha - beta) * p_j * (1.0 - r_j)

    hybrid = term_sem + term_morph_sim + term_morph_prio
    return max(0.0, min(1.0, hybrid))


# ----------------------------------------------------------------------
# Dedupe routine with deeper integration
# ----------------------------------------------------------------------
def dedupe_similar_records(
    records: List[List[float]],
    morphologies: List[Morphology],
    hamming_threshold: int = 8,
    alpha: float = 0.5,
    beta: float = 0.3,
    sketch_width: int = 256,
    sketch_depth: int = 4,
) -> List[Tuple[int, int, float]]:
    """
    Identify record pairs that are both perceptually similar
    (via Hamming distance on average‑hash) and hybrid‑neighbor close.

    Parameters
    ----------
    records : List[List[float]]
        Numeric feature vectors.
    morphologies : List[Morphology]
        Physical description aligned with ``records``.
    hamming_threshold : int
        Maximum Hamming distance to consider two records perceptually similar.
    alpha, beta : float
        Weighting parameters for the hybrid score (see ``hybrid_neighbor_score``).
    sketch_width, sketch_depth : int
        Parameters for the underlying Count‑Min sketch.

    Returns
    -------
    List[Tuple[int, int, float]]
        List of (idx_i, idx_j, hybrid_score) for qualifying pairs.
    """
    if len(records) != len(morphologies):
        raise ValueError("records and morphologies must have the same length")

    n = len(records)

    # 1. Build perceptual hashes for all records (average hash)
    phashes = [compute_phash(rec) for rec in records]

    # 2. Build a Count‑Min sketch from stringified identifiers.
    #    Here we treat each record's rounded vector as its quasi‑identifier.
    sketch = CountMinSketch(width=sketch_width, depth=sketch_depth)
    identifiers: List[str] = []
    for rec in records:
        identifier = ",".join(f"{v:.2f}" for v in rec)
        identifiers.append(identifier)
        sketch.add(identifier)

    results: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            # Hamming filter – fast pre‑screen
            if hamming_distance(phashes[i], phashes[j]) > hamming_threshold:
                continue

            vec_i = np.array(records[i])
            vec_j = np.array(records[j])
            morph_i = morphologies[i]
            morph_j = morphologies[j]

            score = hybrid_neighbor_score(
                vec_i=vec_i,
                vec_j=vec_j,
                morph_i=morph_i,
                morph_j=morph_j,
                sketch=sketch,
                identifier_j=identifiers[j],
                total_records=n,
                alpha=alpha,
                beta=beta,
            )
            if score > 0.5:  # configurable acceptance cut
                results.append((i, j, score))

    return results


# ----------------------------------------------------------------------
# Privacy‑aware morphology analysis (batch version)
# ----------------------------------------------------------------------
def privacy_aware_morphology_analysis(
    morphologies: List[Morphology],
    quasi_identifiers: List[str],
    total_records: int,
    sketch_width: int = 256,
    sketch_depth: int = 4,
    alpha: float = 0.4,
) -> List[float]:
    """
    Return privacy‑adjusted recovery priorities for a collection of morphologies.

    Each priority ``p`` is multiplied by ``(1‑r)`` where ``r`` is the reconstruction
    risk of the corresponding identifier as estimated by a Count‑Min sketch.
    """
    if len(morphologies) != len(quasi_identifiers):
        raise ValueError("Lengths of morphologies and identifiers must match")

    sketch = CountMinSketch(width=sketch_width, depth=sketch_depth)
    for ident in quasi_identifiers:
        sketch.add(ident)

    adjusted: List[float] = []
    for morph, ident in zip(morphologies, quasi_identifiers):
        p = recovery_priority(morph)
        freq = sketch.estimate(ident)
        r = reconstruction_risk(freq, total_records)
        adjusted.append(alpha * p * (1.0 - r) + (1.0 - alpha) * p)
    return adjusted


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data for a quick sanity check
    random.seed(42)
    np.random.seed(42)

    demo_records = [list(np.random.rand(10)) for _ in range(20)]
    demo_morphs = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0),
        )
        for _ in range(20)
    ]

    pairs = dedupe_similar_records(
        records=demo_records,
        morphologies=demo_morphs,
        hamming_threshold=6,
        alpha=0.4,
        beta=0.3,
    )
    print(f"Found {len(pairs)} candidate pairs")
    for i, j, sc in pairs[:5]:
        print(f"({i}, {j}) → {sc:.3f}")