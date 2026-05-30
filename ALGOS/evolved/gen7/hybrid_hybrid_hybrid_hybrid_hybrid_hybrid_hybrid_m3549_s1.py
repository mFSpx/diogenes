# DARWIN HAMMER — match 3549, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_minimu_m90_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s2.py (gen6)
# born: 2026-05-29T23:50:38Z

"""Hybrid Algorithm integrating Hybrid Privacy Model (Parent A) and Sheaf‑Cohomology with Regret‑Weighted Pruning (Parent B).

Mathematical Bridge
-------------------
* **Bayesian risk** from Parent A (bayes_update, bayes_marginal) is used to
  re‑weight the **edge weights** of a graph.
* The **epistemic certainty flags** are mapped to numeric confidence factors
  and applied both to edge weighting and to the regret values of candidate
  actions.
* The graph’s **incidence (coboundary) matrix** Δ, the core matrix operation of
  the sheaf‑cohomology in Parent B, is built from the same edge set whose
  weights now embed Bayesian risk and epistemic confidence.
* A **regret‑weighted pruning** step (exponential decay `p(t)=λ·exp(-α·t)`) uses
  the Gini coefficient of the regret distribution to adapt the pruning
  probability, again modulated by epistemic confidence.
* The final hybrid decision combines the weighted coboundary information with
  the pruned candidate set, yielding a decision that respects both privacy‑
  aware risk assessment and topological‑cohomological structure.

The module provides three core functions demonstrating this integration:
`compute_edge_weights`, `build_coboundary_matrix`, and `regret_weighted_pruning`,
plus a convenience `hybrid_decision` orchestrator.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and simple utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[int, int]                     # indices of points in a node list
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Numeric confidence factors for each epistemic flag (higher = more certain)
EPISTEMIC_CONFIDENCE: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood·prior + false_positive·(1‑prior)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, false_positive: float) -> float:
    """Posterior P(H|E) using Bayes rule."""
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def gini_coefficient(values: Iterable[float]) -> float:
    """Classic Gini coefficient for a non‑empty list of non‑negative numbers."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient requires non‑negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_edge_weights(
    points: List[Point],
    edges: List[Edge],
    prior_risk: float,
    likelihood: float,
    false_positive: float,
    flag_map: Dict[Edge, str],
) -> np.ndarray:
    """
    Produce a weight vector `w` for each edge.

    Weight = Euclidean length × (1‑posterior_risk) × epistemic_confidence.

    The posterior risk is obtained via Bayesian update using the same
    `prior_risk`, `likelihood`, and `false_positive` for every edge.
    """
    posterior = bayes_update(prior_risk, likelihood, false_positive)
    w = np.empty(len(edges), dtype=float)

    for idx, (i, j) in enumerate(edges):
        d = length(points[i], points[j])
        flag = flag_map.get((i, j), "POSSIBLE")
        conf = EPISTEMIC_CONFIDENCE.get(flag, 0.5)
        w[idx] = d * (1.0 - posterior) * conf
    return w

def build_coboundary_matrix(num_nodes: int, edges: List[Edge]) -> np.ndarray:
    """
    Construct the (oriented) coboundary matrix Δ ∈ ℝ^{|E|×|V|}.

    For each edge (i → j) we set Δ[e,i] = -1, Δ[e,j] = +1, all other entries 0.
    This matrix is the discrete analogue of the exterior derivative in sheaf
    cohomology and will be used together with the edge weights.
    """
    m = len(edges)
    Δ = np.zeros((m, num_nodes), dtype=float)
    for e_idx, (i, j) in enumerate(edges):
        Δ[e_idx, i] = -1.0
        Δ[e_idx, j] = 1.0
    return Δ

def regret_weighted_pruning(
    candidates: List[Dict],
    time_step: int,
    lam: float = 0.9,
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Prune a list of candidate actions using a decreasing‑exponential probability
    and a regret‑weighted adjustment.

    Each candidate dict must contain:
        - 'regret': float ≥ 0
        - 'flag'  : one of EPISTEMIC_FLAGS

    The base pruning probability is p(t)=λ·exp(-α·t).  We then scale the
    probability for each candidate by:
        scale = (regret / max_regret) * epistemic_confidence

    The Gini coefficient of the regret distribution modulates the overall
    aggressiveness: higher inequality (high Gini) → lower pruning probability.
    """
    if not candidates:
        return []

    base_p = lam * math.exp(-alpha * time_step)
    regrets = [c["regret"] for c in candidates]
    max_regret = max(regrets) if max(regrets) > 0 else 1.0
    gini = gini_coefficient(regrets)

    # Adjust global probability by Gini (more unequal → less aggressive pruning)
    global_p = base_p * (1.0 - gini * 0.5)   # clamp to [0, base_p]

    survivors = []
    for cand in candidates:
        conf = EPISTEMIC_CONFIDENCE.get(cand["flag"], 0.5)
        scale = (cand["regret"] / max_regret) * conf
        p = global_p * scale
        if random.random() > p:   # keep if not pruned
            survivors.append(cand)
    return survivors

def hybrid_decision(
    points: List[Point],
    edges: List[Edge],
    edge_flags: Dict[Edge, str],
    prior_risk: float,
    likelihood: float,
    false_positive: float,
    candidates: List[Dict],
    time_step: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    End‑to‑end hybrid pipeline:

    1. Compute Bayesian‑aware edge weights.
    2. Build the coboundary matrix Δ.
    3. Apply regret‑weighted pruning to the candidate actions.
    4. Return (weights, Δ, pruned_candidates) for downstream use.
    """
    w = compute_edge_weights(
        points, edges, prior_risk, likelihood, false_positive, edge_flags
    )
    Δ = build_coboundary_matrix(len(points), edges)
    pruned = regret_weighted_pruning(candidates, time_step)
    return w, Δ, pruned

# ----------------------------------------------------------------------
# Simple data class for candidates (optional convenience)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """Container for a decision candidate."""
    id: int
    regret: float
    flag: str

    def to_dict(self) -> Dict:
        return {"id": self.id, "regret": self.regret, "flag": self.flag}

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny geometric graph
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    eds = [(0, 1), (1, 2), (2, 0)]
    # Assign epistemic flags per edge
    eflags = {
        (0, 1): "FACT",
        (1, 2): "PROBABLE",
        (2, 0): "POSSIBLE",
    }

    # Bayesian risk parameters (example values)
    prior = 0.3          # prior probability of a privacy breach
    like = 0.7           # likelihood of observing evidence given breach
    fp = 0.1             # false‑positive rate

    # Candidate actions
    raw_candidates = [
        {"id": 1, "regret": 2.5, "flag": "FACT"},
        {"id": 2, "regret": 5.0, "flag": "PROBABLE"},
        {"id": 3, "regret": 1.0, "flag": "POSSIBLE"},
        {"id": 4, "regret": 3.5, "flag": "BULLSHIT"},
    ]

    # Run the hybrid decision pipeline
    weights, coboundary, survivors = hybrid_decision(
        points=pts,
        edges=eds,
        edge_flags=eflags,
        prior_risk=prior,
        likelihood=like,
        false_positive=fp,
        candidates=raw_candidates,
        time_step=4,
    )

    # Simple sanity prints (no external dependencies)
    print("Edge weights:", weights)
    print("Coboundary matrix Δ:\n", coboundary)
    print("Surviving candidates after pruning:", survivors)

    # Verify dimensions
    assert weights.shape[0] == len(eds), "Weight vector length mismatch"
    assert coboundary.shape == (len(eds), len(pts)), "Coboundary shape mismatch"
    print("Smoke test passed.")