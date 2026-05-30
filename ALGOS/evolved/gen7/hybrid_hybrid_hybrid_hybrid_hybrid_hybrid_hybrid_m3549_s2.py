# DARWIN HAMMER — match 3549, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_minimu_m90_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s2.py (gen6)
# born: 2026-05-29T23:50:38Z

"""Hybrid algorithm merging:
- PARENT ALGORITHM A: probabilistic risk (Bayesian update) + epistemic certainty flags.
- PARENT ALGORITHM B: sheaf‑cohomology incidence (coboundary) + regret‑weighted exponential pruning.

Mathematical bridge:
Epistemic certainty flags are turned into scalar confidence factors that re‑weight
graph edge lengths. Those re‑weighted lengths become the edge costs of a
minimum‑cost spanning tree (MST).  Node priors are updated with Bayesian
marginals (parent A) and then assembled into a risk vector.  The coboundary (incidence)
matrix of the sheaf (parent B) maps node risks to edge‑wise regret values.
The Gini coefficient of the regret distribution feeds the exponential pruning
probability `p(t)=λ·exp(-α·t)`, which probabilistically removes high‑regret edges
from the MST.  The resulting pruned edge set is the unified output of the hybrid
system.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and simple utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping flags → confidence factor (higher = more trustworthy)
FLAG_CONFIDENCE = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.3,
}


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points (parent A)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update (parent A)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability given marginal (parent A)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# 1. Epistemic‑aware edge weighting
# ----------------------------------------------------------------------
def flag_confidence(flag: str) -> float:
    """Translate an epistemic flag into a numeric confidence factor."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {flag!r}")
    return FLAG_CONFIDENCE[flag]


def compute_edge_weight(
    a: str,
    b: str,
    points: Dict[str, Point],
    flags: Dict[str, str],
    priors: Dict[str, float],
    *,
    false_positive: float = 0.05,
    likelihood: float = 0.9,
) -> float:
    """
    Combine Euclidean distance, epistemic confidence, and Bayesian‑updated risk
    into a single scalar cost for the edge (a, b).
    """
    # Base geometric cost
    geo = length(points[a], points[b])

    # Epistemic scaling – geometric cost is penalised if either endpoint is
    # epistemically weak.
    conf_a = flag_confidence(flags[a])
    conf_b = flag_confidence(flags[b])
    epistemic_factor = (conf_a + conf_b) / 2.0  # average confidence

    # Bayesian risk update for each endpoint
    marginal_a = bayes_marginal(priors[a], likelihood, false_positive)
    posterior_a = bayes_update(priors[a], likelihood, marginal_a)

    marginal_b = bayes_marginal(priors[b], likelihood, false_positive)
    posterior_b = bayes_update(priors[b], likelihood, marginal_b)

    risk_factor = (posterior_a + posterior_b) / 2.0

    # Final weight: larger distance, lower confidence, higher risk ⇒ larger cost
    return geo / epistemic_factor * (1.0 + risk_factor)


def build_weighted_graph(
    points: Dict[str, Point],
    flags: Dict[str, str],
    priors: Dict[str, float],
) -> Dict[Edge, float]:
    """Create a complete undirected graph with hybrid edge weights."""
    nodes = list(points.keys())
    weights: Dict[Edge, float] = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            w = compute_edge_weight(a, b, points, flags, priors)
            weights[(a, b)] = w
    return weights


# ----------------------------------------------------------------------
# 2. Minimum‑cost spanning tree (Prim)
# ----------------------------------------------------------------------
def prim_mst(nodes: List[str], edge_weights: Dict[Edge, float]) -> List[Edge]:
    """Return the edges of a minimum‑cost spanning tree using Prim's algorithm."""
    if not nodes:
        return []

    visited = {nodes[0]}
    mst: List[Edge] = []
    # Candidate edges: (weight, (u, v))
    candidates: List[Tuple[float, Edge]] = []

    def push_edges(u: str) -> None:
        for v in nodes:
            if v == u or v in visited:
                continue
            key = (u, v) if (u, v) in edge_weights else (v, u)
            candidates.append((edge_weights[key], key))

    push_edges(nodes[0])

    while len(visited) < len(nodes):
        # Select the cheapest edge crossing the cut
        candidates.sort(key=lambda x: x[0])
        weight, (u, v) = candidates.pop(0)
        if v in visited and u in visited:
            continue
        new_node = v if v not in visited else u
        visited.add(new_node)
        mst.append((u, v))
        push_edges(new_node)

    return mst


# ----------------------------------------------------------------------
# 3. Sheaf‑cohomology incidence (coboundary) matrix
# ----------------------------------------------------------------------
def incidence_matrix(nodes: List[str], edges: List[Edge]) -> np.ndarray:
    """
    Build the (unsigned) incidence matrix Δ ∈ ℝ^{|E|×|V|}.
    Row i corresponds to edge i, column j to node j; entry is 1 if node
    participates in the edge, else 0.
    """
    node_index = {n: i for i, n in enumerate(nodes)}
    mat = np.zeros((len(edges), len(nodes)), dtype=float)
    for i, (u, v) in enumerate(edges):
        mat[i, node_index[u]] = 1.0
        mat[i, node_index[v]] = 1.0
    return mat


def regret_vector(
    incidence: np.ndarray, risk_vector: np.ndarray
) -> np.ndarray:
    """
    Compute edge‑wise regret as Δ · r, where r is the vector of node risks.
    """
    return incidence @ risk_vector


# ----------------------------------------------------------------------
# 4. Gini coefficient of a numeric array
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient (0 = perfect equality, 1 = maximal inequality)."""
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals, dtype=float)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return float(gini)


# ----------------------------------------------------------------------
# 5. Exponential regret‑weighted pruning (parent B)
# ----------------------------------------------------------------------
def exponential_prune(
    edges: List[Edge],
    regrets: np.ndarray,
    lambda_: float,
    alpha: float,
    t: float,
) -> List[Edge]:
    """
    Probabilistically drop edges according to p(t)=λ·exp(-α·t) weighted by
    normalized regret.  Higher regret ⇒ higher chance of removal.
    """
    if not edges:
        return []

    prob_base = lambda_ * math.exp(-alpha * t)
    max_regret = regrets.max() if regrets.size else 1.0
    kept: List[Edge] = []
    for edge, regret in zip(edges, regrets):
        # Scale base probability by (regret / max_regret)
        p_remove = prob_base * (regret / max_regret)
        if random.random() >= p_remove:
            kept.append(edge)
    return kept


# ----------------------------------------------------------------------
# 6. Full hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_pipeline(
    points: Dict[str, Point],
    flags: Dict[str, str],
    priors: Dict[str, float],
    *,
    false_positive: float = 0.05,
    likelihood: float = 0.9,
    lambda_: float = 0.9,
    alpha: float = 0.1,
    t: float = 5.0,
) -> List[Edge]:
    """
    Execute the combined algorithm:
    1. Build weighted graph (geometric + epistemic + Bayesian risk).
    2. Extract MST (minimum‑cost tree).
    3. Construct incidence matrix Δ for the MST.
    4. Compute node risk vector (posterior probabilities).
    5. Derive regret vector = Δ·risk.
    6. Compute Gini of regret (for diagnostics only).
    7. Prune edges with exponential regret‑weighted strategy.
    Returns the final set of edges.
    """
    # 1. Weighted graph
    edge_weights = build_weighted_graph(points, flags, priors)

    # 2. MST
    nodes = list(points.keys())
    mst_edges = prim_mst(nodes, edge_weights)

    # 3. Incidence matrix
    Δ = incidence_matrix(nodes, mst_edges)

    # 4. Posterior risk vector (using Bayesian update)
    risk_vals = []
    for n in nodes:
        marginal = bayes_marginal(priors[n], likelihood, false_positive)
        posterior = bayes_update(priors[n], likelihood, marginal)
        risk_vals.append(posterior)
    r = np.array(risk_vals, dtype=float)

    # 5. Regret per edge
    regrets = regret_vector(Δ, r)

    # 6. Gini (optional diagnostic)
    gini = gini_coefficient(regrets)
    print(f"Gini coefficient of edge regrets: {gini:.4f}")

    # 7. Pruning
    final_edges = exponential_prune(mst_edges, regrets, lambda_, alpha, t)
    return final_edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic scenario with 5 nodes
    sample_points = {
        "A": (0.0, 0.0),
        "B": (1.0, 2.0),
        "C": (3.0, 1.0),
        "D": (4.0, 4.0),
        "E": (2.0, 5.0),
    }

    sample_flags = {
        "A": "FACT",
        "B": "PROBABLE",
        "C": "POSSIBLE",
        "D": "BULLSHIT",
        "E": "SURE_MAYBE",
    }

    # Prior risk probabilities (e.g., probability of a fault)
    sample_priors = {
        "A": 0.02,
        "B": 0.10,
        "C": 0.05,
        "D": 0.30,
        "E": 0.15,
    }

    result = hybrid_pipeline(
        sample_points,
        sample_flags,
        sample_priors,
        false_positive=0.02,
        likelihood=0.85,
        lambda_=0.7,
        alpha=0.05,
        t=3.0,
    )

    print("Final edge set after hybrid processing:")
    for u, v in result:
        print(f"  {u} – {v}")