# DARWIN HAMMER — match 1334, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s1.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s0.py (gen4)
# born: 2026-05-29T23:35:37Z

"""Hybrid Algorithm Combining Caputo Fractional Dynamics, NLMS Adaptation, and Epistemic‑Certainty Edge Weights.

Parents
-------
* hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s1.py – provides a Caputo fractional derivative implementation and a basic tree‑cost evaluator.
* hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s0.py – implements Normalized Least Mean Squares (NLMS) prediction/update and Bayesian‑inspired epistemic‑certainty edge weighting.

Mathematical Bridge
-------------------
For every candidate edge *e = (i, j)* we fuse the three ingredients:

1. **NLMS decision scores** `s_i, s_j` obtained from the current NLMS weight vector and feature vectors
   `x_i, x_j`. Their normalized sum forms a *prior* probability.

2. **Epistemic certainty** `c(e) ∈ [0,1]`. Using the prior we compute a Bayesian marginal
   
   marginal = bayes_marginal(prior, 1 - c(e), c(e) * 0.1)
   
   which quantifies the effective “error likelihood’’ of the edge.

3. **Temporal fractional dynamics** of the NLMS prediction error series `e[t]`. The Caputo
   fractional derivative of order `α ∈ (0,1)`,
   
   D^α e(t) = 1/Γ(1-α) ∫_0^t (t‑τ)^{-α} e(τ) dτ
   
   is approximated by the discrete `caputo_derivative` routine. The magnitude of this
   derivative modulates the edge weight, rewarding edges that belong to a “smooth’’ error
   evolution.

The hybrid edge weight is therefore

w(e) = d(i,j) * (1 - marginal) * (1 + |D^α e(t)|) + ε

where `d(i,j)` is the Euclidean distance between node coordinates and `ε` avoids zero
weights.  The set of hybrid weights feeds a standard Kruskal minimum‑spanning‑tree
construction, while an `EndpointCircuitBreaker` can prune edges that exceed a failure
threshold.

The resulting system simultaneously adapts NLMS coefficients, respects epistemic
certainty, and reacts to the fractional temporal structure of prediction errors.
"""

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple circuit‑breaker that blocks further processing after N failures."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    _LANCZOS_G = 7
    _LANCZOS_C = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        z_minus_one = z - 1
        a = _LANCZOS_C[0]
        for i in range(1, len(_LANCZOS_C)):
            a += _LANCZOS_C[i] / (z_minus_one + i)
        t = z_minus_one + _LANCZOS_G + 0.5
        return math.sqrt(2 * math.pi) * t ** (z_minus_one + 0.5) * math.exp(-t) * a


def caputo_derivative(alpha: float, t: int, series: List[float]) -> float:
    """Discrete Caputo fractional derivative of order `alpha` evaluated at integer time `t`."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be in (0,1) for the Caputo derivative.")
    if t == 0:
        return 0.0
    integral = 0.0
    for tau in range(t):
        kernel = (t - tau) ** (1 - alpha)
        integral += series[tau] * kernel / gamma_lanczos(2 - alpha)
    return integral / gamma_lanczos(1 - alpha)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS adaptation step.
    Returns the updated weight vector and the instantaneous error.
    """
    y = nlms_predict(weights, x)
    e = target - y
    norm_x_sq = max(np.dot(x, x), eps)
    step = (mu / (norm_x_sq + eps)) * e * x
    new_weights = weights + step
    return new_weights, e


def bayes_marginal(prior: float, likelihood: float, fp: float) -> float:
    """
    Simple Bayesian marginalization.
    Returns P(error | evidence) ≈ (prior * likelihood) /
    (prior * likelihood + (1‑prior)*(1‑likelihood) + fp)
    """
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * (1 - likelihood) + fp
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    i: int,
    j: int,
    nodes: Dict[int, Tuple[float, float]],
    nlms_weights: np.ndarray,
    x_i: np.ndarray,
    x_j: np.ndarray,
    certainty: float,
    alpha: float,
    error_series: List[float],
    eps: float = 1e-12,
) -> float:
    """
    Compute the hybrid weight for edge (i, j).

    Parameters
    ----------
    i, j : int
        Node identifiers.
    nodes : dict
        Mapping node → (x, y) coordinates.
    nlms_weights : np.ndarray
        Current NLMS coefficient vector.
    x_i, x_j : np.ndarray
        Feature vectors associated with the two nodes.
    certainty : float
        Epistemic certainty factor c(e) ∈ [0,1].
    alpha : float
        Fractional order for the Caputo derivative.
    error_series : list[float]
        Historical NLMS prediction errors; length must be ≥ current time index.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    weight : float
        Hybrid edge weight.
    """
    # 1. NLMS decision scores
    s_i = nlms_predict(nlms_weights, x_i)
    s_j = nlms_predict(nlms_weights, x_j)

    # 2. Prior from NLMS scores (normalized)
    prior_num = s_i + s_j
    prior = prior_num / (prior_num + eps)

    # 3. Bayesian marginal using epistemic certainty
    likelihood = 1.0 - certainty
    fp = certainty * 0.1
    marginal = bayes_marginal(prior, likelihood, fp)

    # 4. Physical Euclidean distance
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    d = math.hypot(xi - xj, yi - yj)

    # 5. Fractional dynamics factor from error series
    t = len(error_series) - 1  # current discrete time index
    fractional_factor = abs(caputo_derivative(alpha, t, error_series))

    # 6. Hybrid weight
    weight = d * (1.0 - marginal) * (1.0 + fractional_factor) + eps
    return weight


def kruskal_mst(
    nodes: Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int]],
    edge_weights: Dict[Tuple[int, int], float],
    breaker: EndpointCircuitBreaker,
) -> List[Tuple[int, int]]:
    """
    Build a Minimum Spanning Tree using Kruskal's algorithm with an optional
    circuit‑breaker that can abort edge insertion after repeated failures.

    Returns the list of edges belonging to the MST.
    """
    parent = {v: v for v in nodes}
    rank = {v: 0 for v in nodes}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[ru] > rank[rv]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1
        return True

    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda e: edge_weights.get(e, edge_weights.get((e[1], e[0]), float("inf"))))

    mst = []
    for e in sorted_edges:
        if not breaker.allow():
            # Circuit breaker closed – stop processing further edges
            break
        try:
            if union(e[0], e[1]):
                mst.append(e)
                breaker.record_success()
        except Exception:
            breaker.record_failure()
    return mst


def hybrid_mst_pipeline(
    nodes: Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int]],
    init_weights: np.ndarray,
    feature_map: Dict[int, np.ndarray],
    targets: Dict[int, float],
    certainties: Dict[Tuple[int, int], float],
    alpha: float = 0.4,
) -> Tuple[List[Tuple[int, int]], np.ndarray, List[float]]:
    """
    End‑to‑end hybrid pipeline:
    1. Run one NLMS update per node (using its feature vector and target).
    2. Collect the error series.
    3. Compute hybrid edge weights.
    4. Build a minimum‑spanning tree with a circuit‑breaker.

    Returns
    -------
    mst_edges : list of edges in the tree
    final_weights : NLMS weight vector after updates
    error_series : list of NLMS errors (used for fractional dynamics)
    """
    # ------------------------------------------------------------------
    # Step 1 – NLMS adaptation per node
    # ------------------------------------------------------------------
    error_series: List[float] = []
    weights = init_weights.copy()
    for node_id, x_vec in feature_map.items():
        target = targets.get(node_id, 0.0)
        weights, err = nlms_update(weights, x_vec, target)
        error_series.append(err)

    # ------------------------------------------------------------------
    # Step 2 – Hybrid edge weighting
    # ------------------------------------------------------------------
    edge_weights: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        c = certainties.get((i, j), certainties.get((j, i), 0.5))
        w = hybrid_edge_weight(
            i,
            j,
            nodes,
            weights,
            feature_map[i],
            feature_map[j],
            certainty=c,
            alpha=alpha,
            error_series=error_series,
        )
        edge_weights[(i, j)] = w

    # ------------------------------------------------------------------
    # Step 3 – Minimum‑spanning tree with circuit breaker
    # ------------------------------------------------------------------
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    mst = kruskal_mst(nodes, edges, edge_weights, breaker)

    return mst, weights, error_series


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph with 5 nodes
    random.seed(42)
    np.random.seed(42)

    num_nodes = 5
    nodes = {i: (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_nodes)}

    # Fully connected edge set
    edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]

    # Feature vectors (dimension 3) and random targets
    dim = 3
    feature_map = {i: np.random.randn(dim) for i in range(num_nodes)}
    targets = {i: random.uniform(-1, 1) for i in range(num_nodes)}

    # Random epistemic certainty per edge
    certainties = {(i, j): random.uniform(0.0, 1.0) for (i, j) in edges}

    # Initialise NLMS weights to zeros
    init_weights = np.zeros(dim)

    mst_edges, final_weights, errors = hybrid_mst_pipeline(
        nodes,
        edges,
        init_weights,
        feature_map,
        targets,
        certainties,
        alpha=0.3,
    )

    print("Nodes:")
    for nid, coord in nodes.items():
        print(f"  {nid}: {coord}")

    print("\nMST edges (node pairs):")
    for e in mst_edges:
        print(f"  {e}")

    print("\nFinal NLMS weights:", final_weights)
    print("Error series:", errors)