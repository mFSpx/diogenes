# DARWIN HAMMER — match 4984, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m944_s0.py (gen5)
# parent_b: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s1.py (gen5)
# born: 2026-05-29T23:59:07Z

"""Hybrid Dendritic‑NLMS Analyzer (HDNA)

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m944_s0.py (Bayesian‑NLMS edge weighting)
- hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s1.py (Hodgkin‑Huxley dendritic current + regret‑weighted ternary decision)

Mathematical bridge:
The Bayesian update from Parent A is repurposed to adapt the sodium‑channel
conductance *g_Na* of each dendritic compartment in Parent B.
For an edge *e* with physical distance *d_e* we treat a prior edge weight
*w_e* as a prior probability. The membrane potential *V_e* (scaled to [0,1])
acts as a likelihood, while a regret‑derived weight from the ternary‑decision
analysis supplies the false‑positive term. The posterior weight
*w'_e = bayes_update(w_e, likelihood, marginal)* scales the base conductance
*g_Na⁰*, yielding a conductance *g_Na(e) = g_Na⁰·w'_e*. This couples spatial
information, Bayesian uncertainty, and biophysical dynamics into a single
unified system.

The module provides three core functions that demonstrate the hybrid operation:
1. `update_conductances` – performs the Bayesian weight update and returns
   conductances per edge.
2. `compute_dendritic_currents` – evaluates the Hodgkin‑Huxley sodium current
   using the adapted conductances.
3. `hybrid_step` – integrates the above, maps currents to a ternary alphabet,
   and produces regret‑weighted decision probabilities.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
NodeId = str

# ----------------------------------------------------------------------
# Parent‑A utilities (Bayesian‑NLMS)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability given prior, likelihood and marginal."""
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Parent‑B utilities (Hodgkin‑Huxley + Regret‑Weighted Ternary)
# ----------------------------------------------------------------------
def calculate_regret_weighted_probabilities(actions: List["MathAction"]) -> np.ndarray:
    """Simple regret weighting: probability ∝ expected value."""
    total = sum(a.expected_value for a in actions)
    if total == 0:
        return np.full(len(actions), 1.0 / len(actions))
    return np.array([a.expected_value / total for a in actions])


def map_membrane_potentials_to_ternary(V: np.ndarray, thresholds: List[float] = None) -> np.ndarray:
    """Map continuous potentials to {-1, 0, 1}."""
    if thresholds is None:
        thresholds = [-50.0, 0.0]  # typical HH resting and threshold values
    ternary = np.zeros_like(V, dtype=int)
    ternary[V < thresholds[0]] = -1
    ternary[(V >= thresholds[0]) & (V < thresholds[1])] = 0
    ternary[V >= thresholds[1]] = 1
    return ternary


# ----------------------------------------------------------------------
# Data classes for regret analysis (from Parent B)
# ----------------------------------------------------------------------
class MathAction:
    def __init__(self, id: str, expected_value: float, cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk


class MathCounterfactual:
    def __init__(self, action_id: str, outcome_value: float, probability: float = 1.0):
        self.action_id = action_id
        self.outcome_value = outcome_value
        self.probability = probability


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def update_conductances(
    node_positions: Dict[NodeId, Point],
    edges: List[Edge],
    prior_weights: Dict[Edge, float],
    V_edge: Dict[Edge, float],
    actions: List[MathAction],
    base_g_Na: float = 120.0,
) -> Dict[Edge, float]:
    """
    Bayesian update of edge weights and conversion to sodium conductances.

    * prior_weights* – initial NLMS‑style edge probabilities (∈[0,1]).
    * V_edge* – membrane potential associated with each edge (mV).
    * actions* – list of possible actions for regret weighting.

    Returns a mapping edge → adapted conductance g_Na(e).
    """
    # Regret‑derived false‑positive term (same for all edges)
    regret_probs = calculate_regret_weighted_probabilities(actions)
    false_positive = 1.0 - np.mean(regret_probs)  # simple complement of average regret weight

    # Scale potentials to a likelihood in [0,1]
    V_vals = np.array(list(V_edge.values()))
    V_min, V_max = V_vals.min(), V_vals.max()
    if V_max == V_min:
        likelihoods = {e: 0.5 for e in V_edge}  # degenerate case
    else:
        likelihoods = {
            e: (V_edge[e] - V_min) / (V_max - V_min) for e in V_edge
        }

    # Distance‑based priors (optional reinforcement of spatial info)
    distances = {
        e: length(node_positions[e[0]], node_positions[e[1]]) for e in edges
    }
    max_dist = max(distances.values()) if distances else 1.0
    distance_factor = {e: 1.0 - (d / max_dist) for e, d in distances.items()}

    # Combine original prior with distance factor
    combined_prior = {
        e: prior_weights.get(e, 0.5) * distance_factor.get(e, 1.0) for e in edges
    }

    # Perform Bayesian update per edge
    conductances = {}
    for e in edges:
        prior = combined_prior[e]
        likelihood = likelihoods.get(e, 0.5)
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        conductances[e] = base_g_Na * posterior  # scale base conductance
    return conductances


def compute_dendritic_currents(
    V: Dict[Edge, float],
    m: Dict[Edge, float],
    h: Dict[Edge, float],
    g_Na_edge: Dict[Edge, float],
    E_Na: float = 50.0,
) -> Dict[Edge, float]:
    """
    Hodgkin‑Huxley sodium current I_Na = g_Na·m³·h·(V‑E_Na) per edge.
    """
    currents = {}
    for e in V:
        g = g_Na_edge.get(e, 120.0)  # fallback to base value
        I_Na = g * (m[e] ** 3) * h[e] * (V[e] - E_Na)
        currents[e] = I_Na
    return currents


def hybrid_step(
    node_positions: Dict[NodeId, Point],
    edges: List[Edge],
    prior_weights: Dict[Edge, float],
    V_series: np.ndarray,
    m_series: np.ndarray,
    h_series: np.ndarray,
    actions: List[MathAction],
) -> Tuple[Dict[Edge, float], np.ndarray, np.ndarray]:
    """
    Execute one hybrid iteration:

    1. Assign a membrane potential to each edge from V_series (averaged).
    2. Update conductances via Bayesian edge weighting.
    3. Compute sodium currents with the adapted conductances.
    4. Map currents to ternary values.
    5. Produce regret‑weighted decision probabilities.

    Returns (currents, ternary_array, regret_probs).
    """
    # 1. Edge‑wise potentials and gating variables (simple averaging)
    V_edge = {e: np.mean(V_series) for e in edges}
    m_edge = {e: np.mean(m_series) for e in edges}
    h_edge = {e: np.mean(h_series) for e in edges}

    # 2. Conductance adaptation
    g_Na_edge = update_conductances(
        node_positions, edges, prior_weights, V_edge, actions, base_g_Na=120.0
    )

    # 3. Sodium currents
    currents = compute_dendritic_currents(V_edge, m_edge, h_edge, g_Na_edge, E_Na=50.0)

    # 4. Ternary mapping of currents
    current_array = np.array(list(currents.values()))
    ternary = map_membrane_potentials_to_ternary(current_array, thresholds=[-20.0, 20.0])

    # 5. Regret‑weighted probabilities (unchanged from Parent B)
    regret_probs = calculate_regret_weighted_probabilities(actions)

    return currents, ternary, regret_probs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    node_positions = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 0.866),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    # Uniform prior weights
    prior_weights = {e: 0.5 for e in edges}

    # Synthetic HH state series (single‑step example)
    rng = np.random.default_rng(42)
    V_series = rng.normal(loc=-30.0, scale=5.0, size=100)  # mV
    m_series = rng.uniform(0.0, 1.0, size=100)
    h_series = rng.uniform(0.0, 1.0, size=100)

    # Define actions for regret analysis
    actions = [
        MathAction("act1", expected_value=10.0),
        MathAction("act2", expected_value=5.0),
        MathAction("act3", expected_value=2.0),
    ]

    currents, ternary, regret_probs = hybrid_step(
        node_positions,
        edges,
        prior_weights,
        V_series,
        m_series,
        h_series,
        actions,
    )

    print("Edge currents (I_Na):")
    for e, I in currents.items():
        print(f"  {e}: {I:.3f} µA/cm²")
    print("\nTernary representation of currents:", ternary)
    print("\nRegret‑weighted decision probabilities:", regret_probs)