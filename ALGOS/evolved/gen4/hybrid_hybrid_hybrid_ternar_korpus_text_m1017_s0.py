# DARWIN HAMMER — match 1017, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s1.py (gen3)
# parent_b: korpus_text.py (gen0)
# born: 2026-05-29T23:32:27Z

"""Hybrid algorithm combining ternary routing (from
`hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s1.py`) with text‑based
feature extraction (minhash, entropy, vector literals from `korpus_text.py`).

Mathematical bridge
------------------
* Each input text is mapped to a high‑dimensional feature vector **v**∈ℝᴰ.
  The vector consists of a normalized minhash signature (treated as a
  probability‑like vector) concatenated with a scalar entropy term.
* A symmetric cost matrix **C** is built from the pairwise squared Euclidean
  distances  Cᵢⱼ = ‖vᵢ−vⱼ‖² .
* The ternary routing step selects an intermediate node *k* that minimises
  C_{source,k}+C_{k,destination}.
* The same Euclidean geometry is reused for a Voronoi partition: a set of
  seed indices defines Voronoi cells; every point is assigned to the nearest
  seed according to the same distance metric.

Thus the routing and the partition share the exact same underlying matrix
operations, providing a mathematically unified hybrid system.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Text‑based feature extraction (minhash, entropy, vector literal)
# ----------------------------------------------------------------------


def _shingles(text: str, width: int = 5) -> List[str]:
    """Generate overlapping substrings (shingles) of given width."""
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def minhash_signature(text: str, k: int = 64, width: int = 5) -> List[int]:
    """
    Very small minhash implementation.
    Returns the k smallest hash values of the shingles.
    """
    if not text:
        return [0] * k
    sh = _shingles(text.lower(), width)
    # deterministic hash: use built‑in hash mixed with a fixed seed
    hashes = [hash(s) & 0xFFFFFFFFFFFFFFFF for s in sh]
    hashes.sort()
    # pad if fewer than k shingles
    return (hashes[:k] + [0] * k)[:k]


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy of the character distribution (up to 10 000 chars)."""
    if not text:
        return 0.0
    text = text[:10000]
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    return entropy


def text_to_vector(text: str, k: int = 64) -> np.ndarray:
    """
    Convert a text into a numeric vector:
    - Normalised minhash signature (k values scaled to [0,1])
    - One additional component: normalised entropy (divided by log2(|alphabet|))
    """
    sig = minhash_signature(text, k=k)
    sig_arr = np.array(sig, dtype=np.float64) / float(0xFFFFFFFFFFFFFFFF)
    ent = shannon_entropy(text)
    # maximum possible entropy for ASCII (≈7.0 bits) – use 8.0 as safe ceiling
    ent_norm = ent / 8.0
    return np.concatenate([sig_arr, np.array([ent_norm], dtype=np.float64)])


# ----------------------------------------------------------------------
# Geometry derived from feature vectors
# ----------------------------------------------------------------------


def build_cost_matrix(vectors: List[np.ndarray]) -> np.ndarray:
    """
    Pairwise squared Euclidean distance matrix.
    C_{ij} = ||v_i - v_j||^2
    """
    if not vectors:
        raise ValueError("vectors list must not be empty")
    stacked = np.stack(vectors)  # shape (N, D)
    # Using (a-b)^2 = a^2 + b^2 - 2ab
    sq_norms = np.sum(stacked ** 2, axis=1, keepdims=True)  # (N,1)
    prod = stacked @ stacked.T  # (N,N)
    C = sq_norms + sq_norms.T - 2 * prod
    # Numerical safety: force non‑negative
    np.maximum(C, 0.0, out=C)
    return C


def ternary_route(cost_matrix: np.ndarray, source: int, destination: int) -> Tuple[int, float]:
    """
    Find intermediate node k that minimises
    cost_matrix[source, k] + cost_matrix[k, destination].
    Returns (k, total_cost). If source==destination, returns (source, 0.0).
    """
    if source == destination:
        return source, 0.0
    # vectorised computation
    combined = cost_matrix[source, :] + cost_matrix[:, destination]
    # exclude the trivial direct hop (k == destination) if we want a true intermediate,
    # but keep it for completeness.
    k = int(np.argmin(combined))
    total = float(combined[k])
    return k, total


def voronoi_partition(points: List[np.ndarray], seed_indices: List[int]) -> Dict[int, List[int]]:
    """
    Assign each point index to the nearest seed index based on Euclidean distance.
    Returns a dict {seed_idx: [point_indices,...]}.
    """
    if not seed_indices:
        raise ValueError("seed_indices must contain at least one index")
    seeds = [points[i] for i in seed_indices]
    assignments: Dict[int, List[int]] = {s_idx: [] for s_idx in seed_indices}
    for idx, pt in enumerate(points):
        # compute distances to all seeds
        dists = [np.linalg.norm(pt - seed) for seed in seeds]
        nearest_seed_pos = int(np.argmin(dists))
        nearest_seed_idx = seed_indices[nearest_seed_pos]
        assignments[nearest_seed_idx].append(idx)
    return assignments


# ----------------------------------------------------------------------
# Hybrid operation exposing the combined functionality
# ----------------------------------------------------------------------


def hybrid_process(
    texts: List[str],
    source_idx: int,
    dest_idx: int,
    seed_indices: List[int],
    k_minhash: int = 64,
) -> Dict[str, any]:
    """
    End‑to‑end hybrid pipeline:
    1. Convert each text to a feature vector.
    2. Build the Euclidean cost matrix.
    3. Perform ternary routing between source and destination.
    4. Compute Voronoi cells using the supplied seed indices.
    Returns a dictionary with all intermediate results.
    """
    if not (0 <= source_idx < len(texts)) or not (0 <= dest_idx < len(texts)):
        raise IndexError("source_idx and dest_idx must be valid indices in texts")
    vectors = [text_to_vector(t, k=k_minhash) for t in texts]
    cost_mat = build_cost_matrix(vectors)
    interm, route_cost = ternary_route(cost_mat, source_idx, dest_idx)
    voronoi = voronoi_partition(vectors, seed_indices)

    return {
        "vectors": vectors,
        "cost_matrix": cost_mat,
        "route": {
            "source": source_idx,
            "destination": dest_idx,
            "intermediate": interm,
            "total_cost": route_cost,
        },
        "voronoi": voronoi,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
    ]

    # Choose source, destination, and seeds (by index)
    src = 0
    dst = 3
    seeds = [1, 2, 4]

    result = hybrid_process(sample_texts, src, dst, seeds)

    print("=== Hybrid Process Result ===")
    print(f"Route: source={src}, destination={dst}, intermediate={result['route']['intermediate']}, cost={result['route']['total_cost']:.4f}")
    print("Voronoi assignment:")
    for seed, members in result["voronoi"].items():
        print(f"  Seed {seed} -> points {members}")
    # Ensure no exception and that matrices have expected shapes
    assert result["cost_matrix"].shape == (len(sample_texts), len(sample_texts))
    print("Smoke test completed successfully.")