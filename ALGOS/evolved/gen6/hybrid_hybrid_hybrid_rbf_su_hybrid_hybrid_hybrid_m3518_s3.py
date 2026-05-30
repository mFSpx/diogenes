# DARWIN HAMMER — match 3518, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1075_s0.py (gen5)
# born: 2026-05-29T23:50:28Z

"""Hybrid RBF‑Krampus‑Hoeffding Fusion
Parents:
- hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s0.py (RBF similarity + Hoeffding bound)
- hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1075_s0.py (Ollivier‑Ricci curvature + Hoeffding bound)

Mathematical bridge:
Both parents rely on the Hoeffding bound to decide when a Hoeffding tree node should split.
Parent A provides a similarity matrix **S** derived from binary‑hashed feature vectors
(using Hamming distance). Parent B supplies a scalar curvature **κ** for each node,
computed from a high‑level feature dictionary (Ollivier‑Ricci curvature).

The fusion treats the curvature values as a node‑wise signal **c** and propagates it
through the similarity graph by the linear operation **ĉ = S · c**.  The resulting
vector **ĉ** captures both local geometric curvature and global similarity
structure.  A split decision is then made by comparing the gain gap of the
propagated curvatures against the Hoeffding bound, thus unifying the two
topologies in a single, mathematically coherent system.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Set, Tuple

import numpy as np

Node = Hashable
Graph = Dict[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function used in similarity weighting."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Very simple perceptual hash: threshold each component against the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit hashes."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S where
    S[i, j] = 1 - (Hamming distance of phashes) / 64.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[n])) for n in nodes]

    for i in range(n):
        for j in range(i, n):
            if i == j:
                S[i, j] = 1.0
            else:
                d = hamming_distance(hashes[i], hashes[j])
                sim = 1.0 - d / 64.0
                S[i, j] = sim
                S[j, i] = sim
    return S, nodes


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def extract_full_features(_: str) -> dict[str, float]:
    """Placeholder: returns a deterministic feature dictionary."""
    return {
        "visceral_ratio": 0.5,
        "tech_ratio": 0.3,
        "legal_osint_ratio": 0.2,
        "ledger_density": 0.1,
        "recursion_score": 0.4,
        "directive_ratio": 0.6,
        "target_density": 0.7,
        "forensic_shield_ratio": 0.8,
        "poetic_entropy": 0.9,
        "dissociative_index": 0.1,
        "wrath_velocity": 0.2,
        "bureaucratic_weaponization_index": 0.3,
        "resource_exhaustion_metric": 0.4,
    }


def ollivier_ricci_curvature(features: dict[str, float]) -> float:
    """Simplified Ollivier‑Ricci curvature: arithmetic mean of feature values."""
    if not features:
        return 0.0
    return sum(features.values()) / len(features)


# ----------------------------------------------------------------------
# Shared Hoeffding bound (identical in both parents)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used to decide if a split is statistically significant."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fused_curvature(
    node_features: Dict[Node, FeatureVec],
    text_by_node: Dict[Node, str],
) -> Tuple[np.ndarray, List[Node]]:
    """
    Compute curvature for each node from its associated text, then propagate it
    through the similarity graph defined by the RBF‑derived matrix S.

    Returns:
        c_hat: (n,) array of similarity‑weighted curvatures.
        nodes: list of node identifiers preserving order.
    """
    # 1. similarity matrix from raw numeric feature vectors
    S, nodes = similarity_matrix(node_features)

    # 2. curvature vector κ for each node
    curvature_vals = np.empty(len(nodes), dtype=np.float64)
    for idx, node in enumerate(nodes):
        feats = extract_full_features(text_by_node.get(node, ""))
        curvature_vals[idx] = ollivier_ricci_curvature(feats)

    # 3. similarity‑weighted propagation
    c_hat = S @ curvature_vals
    return c_hat, nodes


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑tree split query."""
    should_split: bool
    bound: float
    gain_gap: float
    node_index: int


def hoeffding_split_decision(
    propagated_curvature: np.ndarray,
    delta: float = 0.05,
    min_samples: int = 30,
) -> SplitDecision:
    """
    Decide whether a node should split based on the spread of its propagated
    curvature values.

    The algorithm treats each entry of `propagated_curvature` as a candidate
    gain.  The best and second‑best gains define a gap; if the gap exceeds the
    Hoeffding bound, a split is warranted.
    """
    if propagated_curvature.size == 0:
        raise ValueError("curvature array is empty")

    # Identify the two largest gains and their indices
    sorted_idx = np.argsort(propagated_curvature)[::-1]
    best_idx = sorted_idx[0]
    best_gain = propagated_curvature[best_idx]
    second_gain = propagated_curvature[sorted_idx[1]] if propagated_curvature.size > 1 else 0.0
    gain_gap = best_gain - second_gain

    # Use range r = max - min as Hoeffding bound input
    r = float(propagated_curvature.max() - propagated_curvature.min())
    bound = hoeffding_bound(r, delta, max(min_samples, 1))

    should = gain_gap > bound
    return SplitDecision(should_split=should, bound=bound, gain_gap=gain_gap, node_index=int(best_idx))


def hybrid_update(
    node_features: Dict[Node, FeatureVec],
    text_by_node: Dict[Node, str],
    delta: float = 0.05,
    min_samples: int = 30,
) -> Tuple[SplitDecision, np.ndarray]:
    """
    High‑level API that computes the fused curvature, evaluates a split decision,
    and returns both the decision and the propagated curvature vector.
    """
    propagated, nodes = fused_curvature(node_features, text_by_node)
    decision = hoeffding_split_decision(propagated, delta=delta, min_samples=min_samples)
    return decision, propagated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph with three nodes
    node_features_example: Dict[Node, FeatureVec] = {
        "A": [random.random() for _ in range(8)],
        "B": [random.random() for _ in range(8)],
        "C": [random.random() for _ in range(8)],
    }

    # Associate each node with a dummy text (the content is irrelevant for the placeholder extractor)
    text_by_node_example: Dict[Node, str] = {
        "A": "alpha",
        "B": "bravo",
        "C": "charlie",
    }

    decision, propagated = hybrid_update(node_features_example, text_by_node_example)

    print("Propagated curvature vector:", propagated)
    print("Split decision:", decision)
    sys.exit(0)