# DARWIN HAMMER — match 2030, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s2.py (gen5)
# born: 2026-05-29T23:40:35Z

"""Hybrid Sheaf‑Cohomology & Regret‑Weighted Decision Engine.

Parents:
- hybrid_hybrid_sheaf_cohomol_hybrid_ternary_lens__m23_s2.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m527_s2.py

Mathematical bridge:
Both parents manipulate linear objects together with a stochastic pruning
policy.  The sheaf module builds a coboundary matrix Δ from restriction
scalars on graph edges.  The decision module supplies a *regret‑weighted
strategy* – a normalized scalar weight for each node – and a scalar
epistemic‑certainty factor.  By scaling each edge’s restriction map with the
average regret‑weight of its incident nodes we obtain a *weighted sheaf*.
The same exponential pruning probability `p(t)=λ·exp(-α·t)` is then applied
to edges, yielding a pruned coboundary matrix whose null‑space dimension
reflects both topological connectivity and the underlying decision‑theoretic
information.

The module implements:
1. `compute_regret_weights` – turn a list of `Action`s into normalized
   regret‑weighted node scalars.
2. `build_hybrid_sheaf` – construct the coboundary matrix from those
   scalars, prune edges with `p(t)`, and return the matrix together with the
   retained edge list.
3. `hybrid_nullspace_dimension` – compute `dim ker(Δ)` via SVD.
4. `gini_coefficient` – auxiliary statistic on the regret‑weight vector.

All functions rely only on `numpy`, `math`, `random`, `sys`, and `pathlib`. """

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Decision‑theoretic data structures (from Parent B)
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_FACTOR = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}


@dataclass(frozen=True)
class Action:
    """Action with cost, baseline probability, and epistemic certainty."""
    cost: float
    probability: float
    epistemic_certainty: str

    def epistemic_factor(self) -> float:
        """Map certainty flag to a scalar in [0,1]."""
        return _EPISTEMIC_FACTOR.get(self.epistemic_certainty.upper(), 0.0)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decreasing pruning probability p(t)=lam·exp(-α·t)."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def compute_regret_weights(actions: List[Action]) -> np.ndarray:
    """
    Compute a normalized regret‑weighted vector w∈ℝⁿ for n actions.

    Regret for action i is `r_i = max(cost) - cost_i`.
    Raw weight = r_i * probability_i * epistemic_factor_i.
    The vector is finally normalized to sum to 1 (or left as zero vector
    if all raw weights are zero).
    """
    if not actions:
        raise ValueError("Action list must not be empty")

    costs = np.array([a.cost for a in actions], dtype=float)
    probs = np.array([a.probability for a in actions], dtype=float)
    epi_factors = np.array([a.epistemic_factor() for a in actions], dtype=float)

    max_cost = np.max(costs)
    regrets = max_cost - costs
    raw_weights = regrets * probs * epi_factors

    total = raw_weights.sum()
    if total == 0.0:
        # fallback to uniform distribution to avoid division by zero
        return np.full_like(raw_weights, 1.0 / len(raw_weights))
    return raw_weights / total


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D non‑negative array.
    G = (∑_{i,j}|x_i-x_j|) / (2·n·∑x_i)
    """
    if values.ndim != 1:
        raise ValueError("Gini coefficient expects a 1‑D array")
    if np.any(values < 0):
        raise ValueError("Gini coefficient expects non‑negative values")
    n = values.size
    if n == 0:
        return 0.0
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    # The classic formula simplified for efficiency
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return gini


def build_hybrid_sheaf(
    actions: List[Action],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Construct a pruned coboundary matrix Δ for a complete graph on the actions.

    - Node i receives the scalar w_i = regret_weight_i.
    - For each undirected edge (i, j) we assign a restriction scalar
      r_{ij} = (w_i + w_j) / 2.
    - Edge (i, j) is kept with probability 1 - p(t) where p(t) is the
      exponential pruning probability.
    - The resulting matrix has shape (n, m) where n = #actions and m = #kept edges.
      Column k corresponds to edge (i, j) and contains +r_{ij} at row i,
      -r_{ij} at row j, and zeros elsewhere.
    The function returns the matrix and the list of retained edges with their
    restriction scalars.
    """
    n = len(actions)
    if n < 2:
        raise ValueError("At least two actions are required to build a sheaf")

    # 1. Regret‑weighted node scalars
    w = compute_regret_weights(actions)  # shape (n,)

    # 2. Enumerate all possible undirected edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # 3. Stochastically prune edges
    keep_edges: List[Tuple[int, int, float]] = []
    for i, j in all_edges:
        if random.random() > prune_probability(t, lam, alpha):
            r = (w[i] + w[j]) / 2.0
            keep_edges.append((i, j, r))

    m = len(keep_edges)
    if m == 0:
        # Degenerate case: no edges survive; return a zero matrix
        return np.zeros((n, 0), dtype=float), keep_edges

    # 4. Assemble coboundary matrix Δ
    delta = np.zeros((n, m), dtype=float)
    for col, (i, j, r) in enumerate(keep_edges):
        delta[i, col] = r
        delta[j, col] = -r

    return delta, keep_edges


def hybrid_nullspace_dimension(delta: np.ndarray, tol: float = 1e-9) -> int:
    """
    Compute dim ker(Δ) via singular‑value decomposition.
    Nullity = n_rows - rank(Δ), where rank is the number of singular values
    larger than `tol`.
    """
    if delta.ndim != 2:
        raise ValueError("Delta must be a 2‑D matrix")
    u, s, vt = np.linalg.svd(delta, full_matrices=False)
    rank = np.sum(s > tol)
    return delta.shape[0] - rank


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a small set of actions covering all epistemic flags
    sample_actions = [
        Action(cost=10.0, probability=0.9, epistemic_certainty="FACT"),
        Action(cost=7.5, probability=0.6, epistemic_certainty="PROBABLE"),
        Action(cost=5.0, probability=0.4, epistemic_certainty="POSSIBLE"),
        Action(cost=12.0, probability=0.2, epistemic_certainty="BULLSHIT"),
        Action(cost=9.0, probability=0.8, epistemic_certainty="SURE_MAYBE"),
    ]

    t = 2.0  # time parameter for pruning
    delta, edges = build_hybrid_sheaf(sample_actions, t)

    print(f"Retained {len(edges)} edges out of {len(sample_actions)*(len(sample_actions)-1)//2}")
    print("First few retained edges (i, j, restriction):", edges[:5])

    null_dim = hybrid_nullspace_dimension(delta)
    print(f"Null‑space dimension of the pruned coboundary matrix: {null_dim}")

    weights = compute_regret_weights(sample_actions)
    gini = gini_coefficient(weights)
    print(f"Regret‑weight vector: {weights}")
    print(f"Gini coefficient of regret weights: {gini:.4f}")

    sys.exit(0)