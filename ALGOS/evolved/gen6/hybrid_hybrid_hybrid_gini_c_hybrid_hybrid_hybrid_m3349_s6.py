# DARWIN HAMMER — match 3349, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_gini_coeffici_hybrid_hybrid_worksh_m2277_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_model__m51_s0.py (gen3)
# born: 2026-05-29T23:49:32Z

import math
import sys
import pathlib
from datetime import datetime as dt
from typing import Iterable, List, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = object
Graph = Dict[Node, List[Node]]
FeatureVec = List[float]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """
    Compute the Gini coefficient of a non‑negative collection.
    Returns 0 for empty or all‑zero input.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if (xs < 0).any():
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * xs_sorted)) / (n * xs_sorted.sum())


def cosine_similarity_matrix(groups: Sequence[str],
                             feature_vecs: Dict[str, FeatureVec]) -> np.ndarray:
    """
    Cosine similarity matrix for the given groups.
    """
    n = len(groups)
    mat = np.zeros((n, n), dtype=float)
    for i, g1 in enumerate(groups):
        v1 = np.asarray(feature_vecs[g1], dtype=float)
        norm1 = np.linalg.norm(v1)
        for j, g2 in enumerate(groups):
            if i > j:
                continue
            v2 = np.asarray(feature_vecs[g2], dtype=float)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = float(np.dot(v1, v2) / (norm1 * norm2))
            mat[i, j] = mat[j, i] = sim
    return mat


def weekday_phase(dow: int) -> float:
    """
    Convert a weekday (0‑6) into a phase in radians.
    """
    if not 0 <= dow <= 6:
        raise ValueError("weekday must be in range 0‑6")
    return (2.0 * math.pi) * (dow / 7.0)


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_weight_vector(groups: Sequence[str],
                         feature_vecs: Dict[str, FeatureVec],
                         dow: int) -> np.ndarray:
    """
    Produce a probability vector that blends:
      * sinusoidal weekday pattern,
      * inter‑group similarity (cosine),
      * distribution inequality (Gini).

    The steps are:
      1. Compute a cosine similarity matrix.
      2. Derive per‑group similarity scores (row means).
      3. Compute the Gini coefficient of those scores.
      4. Build a sinusoidal base pattern.
      5. Modulate the base with an exponential factor driven by similarity
         and tempered by the Gini coefficient.
      6. Normalise to sum to 1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")

    # 1‑3: similarity & inequality
    sim_mat = cosine_similarity_matrix(groups, feature_vecs)
    sim_scores = sim_mat.mean(axis=1)                     # average similarity per group
    gini = gini_coefficient(sim_scores)

    # 4: sinusoidal base (always non‑negative)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = weekday_phase(dow)
    base = 1.0 + np.sin(base_angles + phase)               # range [0, 2]

    # 5: similarity‑driven modulation
    #   beta scales the influence; we map gini∈[0,1] → beta∈[0,2]
    beta = 2.0 * gini
    mod_factor = np.exp(beta * (sim_scores - sim_scores.mean()))

    # Combine and normalise
    raw_weights = base * mod_factor
    weights = raw_weights / raw_weights.sum()
    return weights


def gpu_memory_allocation(groups: Sequence[str],
                          memory_matrix: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """
    Allocate GPU memory to each group.

    The allocation is the matrix‑vector product ``memory_matrix @ weights``.
    This respects both the inter‑group memory interaction (the matrix) and
    the probability distribution given by ``weights``.
    """
    if memory_matrix.shape[0] != memory_matrix.shape[1]:
        raise ValueError("memory_matrix must be square")
    if memory_matrix.shape[0] != len(groups):
        raise ValueError("memory_matrix size must match number of groups")
    if weights.shape != (len(groups),):
        raise ValueError("weights length must match number of groups")
    allocation = memory_matrix @ weights
    return allocation


def hybrid_gpu_memory_allocation(groups: Sequence[str],
                                 feature_vecs: Dict[str, FeatureVec],
                                 memory_matrix: np.ndarray,
                                 dow: int) -> np.ndarray:
    """
    End‑to‑end hybrid allocation: compute weights then allocate memory.
    """
    weights = hybrid_weight_vector(groups, feature_vecs, dow)
    return gpu_memory_allocation(groups, memory_matrix, weights)


# ----------------------------------------------------------------------
# Demo / simple test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example groups and feature vectors
    groups = ["A", "B", "C"]
    feature_vecs = {
        "A": [1.0, 0.0, 0.0],
        "B": [0.0, 1.0, 0.0],
        "C": [0.0, 0.0, 1.0],
    }

    # Symmetric memory demand matrix (e.g., pairwise bandwidth)
    memory_matrix = np.array([
        [1000, 500, 500],
        [500, 1000, 500],
        [500, 500, 1000],
    ], dtype=float)

    dow = dt.now().weekday()
    weights = hybrid_weight_vector(groups, feature_vecs, dow)
    allocation = hybrid_gpu_memory_allocation(groups, feature_vecs, memory_matrix, dow)

    print("Hybrid Weight Vector:", weights)
    print("Hybrid GPU Memory Allocation:", allocation)