# DARWIN HAMMER — match 685, survivor 2
# gen: 5
# parent_a: hoeffding_tree.py (gen0)
# parent_b: hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py (gen4)
# born: 2026-05-29T23:30:28Z

"""Hybrid Hoeffding‑Gini Split Engine
Integrates:
* Parent A – *hoeffding_tree.py*: Hoeffding bound based split decision.
* Parent B – *hybrid_gini_coefficient_hybrid_hybrid_rbf_su_m344_s1.py*: Gini coefficient,
  perceptual hash, and similarity matrix utilities.

Mathematical bridge:
The Gini coefficient **G** of the feature distribution at a node quantifies
inequality.  We map **G** through a Gaussian kernel γ(G)=exp(-(ε·G)²) (ε is a
scale) and use it to *shrink* the effective range **r** in the Hoeffding bound:
 r̂ = r·γ(G).  
A more heterogeneous node (high G) yields a smaller γ, tightening the bound
and requiring stronger evidence to split, while a homogeneous node (low G)
relaxes the bound.  The adjusted bound is then fed to the standard Hoeffding
split logic.  The similarity matrix built from perceptual hashes provides an
auxiliary view of node relationships that can be used for downstream tasks
(e.g., drift detection)."""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Hashable, Sequence, List, Dict, Tuple, Iterable

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Hoeffding utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


# ----------------------------------------------------------------------
# Parent B – Gini and similarity utilities
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative sequence."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used to modulate the Hoeffding range."""
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    """Very simple perceptual hash: 1‑bit per value (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def similarity_matrix(features: Dict[Hashable, Sequence[float]]) -> Tuple[np.ndarray, List[Hashable]]:
    """
    Build a symmetric similarity matrix S where
    S[i, j] = 1 - (Hamming distance / max_bits).
    """
    nodes = list(features.keys())
    n = len(nodes)
    max_bits = 64
    S = np.empty((n, n), dtype=np.float64)

    hashes = [compute_phash(list(features[node])) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            hd = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - hd / max_bits
            S[i, j] = S[j, i] = sim
    return S, nodes


# ----------------------------------------------------------------------
# Hybrid core: Gini‑aware Hoeffding bound & split decision
# ----------------------------------------------------------------------
def adjusted_hoeffding_bound(gini: float, r: float, delta: float, n: int, epsilon: float = 1.0) -> float:
    """
    Adjust the Hoeffding range r by a Gaussian factor derived from the Gini coefficient.
    The higher the Gini (more inequality), the smaller the effective range,
    tightening the bound.
    """
    gamma = gaussian(gini, epsilon)          # γ ∈ (0,1]
    r_adj = r * gamma
    return hoeffding_bound(r_adj, delta, n)


def should_split_hybrid(
    best_gain: float,
    second_best_gain: float,
    gini: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
    epsilon: float = 1.0,
) -> SplitDecision:
    """
    Hybrid split decision that uses a Gini‑adjusted Hoeffding bound.
    """
    eps = adjusted_hoeffding_bound(gini, r, delta, n, epsilon)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
def _demo():
    # Dummy streaming node statistics
    feature_values_node = {
        "node_A": [random.random() for _ in range(30)],
        "node_B": [random.random() ** 2 for _ in range(30)],
        "node_C": [0.1 for _ in range(30)],
    }

    # 1️⃣ Similarity matrix of nodes
    sim_mat, node_order = similarity_matrix(feature_values_node)
    print("Similarity matrix:\n", sim_mat)

    # 2️⃣ Compute Gini for a chosen node
    gini_A = gini_coefficient(feature_values_node["node_A"])
    print(f"Gini(node_A) = {gini_A:.4f}")

    # 3️⃣ Hybrid split decision
    best_gain = 0.12
    second_best_gain = 0.07
    r = 1.0          # range of the gain (e.g., information gain ∈ [0,1])
    delta = 0.05
    n = 150          # observed examples at the node
    decision = should_split_hybrid(
        best_gain,
        second_best_gain,
        gini_A,
        r,
        delta,
        n,
        tie_threshold=0.02,
        epsilon=1.5,
    )
    print("Hybrid split decision:", decision)


if __name__ == "__main__":
    _demo()