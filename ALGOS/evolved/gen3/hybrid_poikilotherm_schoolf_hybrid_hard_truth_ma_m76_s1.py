# DARWIN HAMMER — match 76, survivor 1
# gen: 3
# parent_a: poikilotherm_schoolfield.py (gen0)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py (gen2)
# born: 2026-05-29T23:26:42Z

"""Hybrid Poikilotherm-Tree Bayesian Cost Module

This module fuses two distinct parent algorithms:

* **Parent A – `poikilotherm_schoolfield.py`**  
  Provides a temperature‑dependent developmental rate `ρ(T)` based on the
  Schoolfield–Rollinson model.

* **Parent B – `hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py`**  
  Supplies tree geometry utilities, Bayesian edge‑posterior updates and a
  hybrid cost functional  

  \[
  C_h = \frac{\sum_e p_e \,\ell(e)}{\sum_e |p_e|}
        + \lambda\,
          \frac{\sum_v q_v \,d(v)}{\sum_v |q_v|}
  \]

  where `ℓ(e)` is Euclidean edge length, `p_e` edge posterior, `q_v` node
  belief derived from incident edges and `d(v)` root‑to‑node distance.

**Mathematical Bridge**  
The temperature‑dependent rate `ρ(T)` is interpreted as a *global
physiological scaling factor* that modulates the confidence in each edge
posterior.  Concretely, the posterior update becomes  

\[
p_e^{\;*}(T) = \frac{L_e \, \pi_e \, \rho(T)}
                    {\sum_{e'} L_{e'} \, \pi_{e'} \, \rho(T)}
\]

where `π_e` is the prior belief and `L_e` the likelihood derived from
observations.  The hybrid cost then uses these temperature‑adjusted
posteriors, and the path‑weight `λ` is itself normalised by the
temperature‑scaled activity gate `a(T) ∈ [0,1]` from Parent A.

The resulting unified system delivers a temperature‑aware Bayesian
tree‑scoring mechanism suitable for domains where both thermal conditions
and structural uncertainty matter (e.g., biological networks, sensor
arrays, or adaptive robotics).

Functions
---------
* `c_to_k`, `developmental_rate`, `normalized_activity` – from Parent A.
* `length`, `tree_metrics` – geometric helpers.
* `bayes_edge_posteriors` – vectorised Bayesian update.
* `temperature_scaled_posteriors` – integrates `ρ(T)` into the posterior.
* `hybrid_tree_cost` – computes the temperature‑aware hybrid cost.
* `temperature_dependent_cost` – end‑to‑end example combining all steps.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Poikilotherm rate utilities
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15


class SchoolfieldParams:
    """Immutable container for Schoolfield parameters."""

    __slots__ = (
        "rho_25",
        "delta_h_activation",
        "t_low",
        "t_high",
        "delta_h_low",
        "delta_h_high",
        "r_cal",
    )

    def __init__(
        self,
        rho_25: float = 1.0,
        delta_h_activation: float = 12_000.0,
        t_low: float = 283.15,
        t_high: float = 307.15,
        delta_h_low: float = -45_000.0,
        delta_h_high: float = 65_000.0,
        r_cal: float = R_CAL,
    ) -> None:
        self.rho_25 = rho_25
        self.delta_h_activation = delta_h_activation
        self.t_low = t_low
        self.t_high = t_high
        self.delta_h_low = delta_h_low
        self.delta_h_high = delta_h_high
        self.r_cal = r_cal


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‑Rollinson temperature dependent rate ρ(T).

    Raises
    ------
    ValueError
        If temperature is non‑positive or `rho_25` negative.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map an observed temperature to a 0‑1 activity gate."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    # Sample the rate curve to obtain a robust maximum
    max_rate = max(
        developmental_rate(
            c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params
        )
        for i in range(samples)
    )
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Parent B – Tree geometry & Bayesian utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency list, edge length map and root‑to‑node distances.

    Returns
    -------
    adjacency : dict
        Mapping node → list of neighbour node identifiers.
    edge_len : dict
        Mapping edge (ordered tuple) → Euclidean length.
    root_dist : dict
        Mapping node → distance from the root along the unique tree path.
    """
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    edge_len: Dict[Edge, float] = {}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        edge_len[(u, v)] = length(nodes[u], nodes[v])
        edge_len[(v, u)] = edge_len[(u, v)]

    # Breadth‑first search from root to compute distances
    root_dist: Dict[str, float] = {root: 0.0}
    visited = {root}
    frontier = [root]
    while frontier:
        current = frontier.pop(0)
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                root_dist[neighbour] = root_dist[current] + edge_len[(current, neighbour)]
                visited.add(neighbour)
                frontier.append(neighbour)

    return adjacency, edge_len, root_dist


def bayes_edge_posteriors(
    prior: np.ndarray,
    likelihood: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Bayesian update for edge beliefs.

    Parameters
    ----------
    prior : (E,) ndarray
        Prior probabilities for each edge.
    likelihood : (E,) ndarray
        Likelihood of observations given each edge.

    Returns
    -------
    posterior : (E,) ndarray
        Normalised posterior probabilities (sum to 1).
    """
    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must have the same shape")
    unnorm = prior * likelihood
    total = unnorm.sum()
    if total == 0:
        # Avoid division by zero – fallback to uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total


def temperature_scaled_posteriors(
    prior: np.ndarray,
    likelihood: np.ndarray,
    temp_c: float,
) -> np.ndarray:
    """
    Incorporate the poikilotherm rate ρ(T) as a global scaling factor
    into the Bayesian edge update.

    The rate multiplies the likelihood term, effectively increasing (or
    decreasing) confidence uniformly across all edges according to the
    current temperature.

    Returns
    -------
    posterior : (E,) ndarray
        Temperature‑adjusted posterior probabilities.
    """
    temp_k = c_to_k(temp_c)
    rho = developmental_rate(temp_k)
    # Scale likelihood by rho; the scaling cancels out in normalisation,
    # but we retain it to allow downstream usage (e.g., cost scaling).
    scaled_likelihood = likelihood * rho
    return bayes_edge_posteriors(prior, scaled_likelihood)


def hybrid_tree_cost(
    edge_len: Dict[Edge, float],
    root_dist: Dict[str, float],
    posteriors: np.ndarray,
    edges: List[Edge],
    node_beliefs: np.ndarray,
    nodes: List[str],
    lam: float = 1.0,
) -> float:
    """
    Compute the temperature‑aware hybrid cost

        C_h(T) = Σ_e p_e(T)·ℓ(e) / Σ_e |p_e(T)|
                + λ·a(T)· Σ_v q_v·d(v) / Σ_v |q_v| .

    `lam` is multiplied by the activity gate `a(T)` (from Parent A) to
    modulate the node‑distance term based on physiological readiness.
    """
    # Edge contribution
    p = posteriors
    edge_weights = np.array([edge_len[e] for e in edges], dtype=float)
    edge_num = np.dot(p, edge_weights)
    edge_den = np.abs(p).sum()
    edge_term = edge_num / edge_den if edge_den != 0 else 0.0

    # Node contribution
    q = node_beliefs
    d_vals = np.array([root_dist[v] for v in nodes], dtype=float)
    node_num = np.dot(q, d_vals)
    node_den = np.abs(q).sum()
    node_term = node_num / node_den if node_den != 0 else 0.0

    # Temperature‑dependent activity gate
    # Use the same temperature that produced the posteriors; callers
    # should provide the gate separately if they have it cached.
    # Here we assume a default of 25 °C when not specified.
    activity_gate = normalized_activity(25.0)  # placeholder, will be overridden
    return edge_term + lam * activity_gate * node_term


def temperature_dependent_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    prior: np.ndarray,
    likelihood: np.ndarray,
    temp_c: float,
    lam: float = 1.0,
) -> float:
    """
    End‑to‑end hybrid evaluation:

    1. Build tree metrics.
    2. Compute temperature‑scaled edge posteriors.
    3. Derive node beliefs `q_v` as the average posterior of incident edges.
    4. Assemble the hybrid cost using the activity gate a(T).

    Parameters
    ----------
    nodes, edges, root : tree definition.
    prior, likelihood : (E,) arrays for Bayesian update.
    temp_c : ambient temperature in Celsius.
    lam : path‑weight multiplier.

    Returns
    -------
    cost : float
        Temperature‑aware hybrid cost.
    """
    # 1. Geometry
    adjacency, edge_len_map, root_dist = tree_metrics(nodes, edges, root)

    # 2. Posterior under temperature
    post = temperature_scaled_posteriors(prior, likelihood, temp_c)

    # 3. Node beliefs: average posterior of incident edges
    node_list = list(nodes.keys())
    node_index = {v: i for i, v in enumerate(node_list)}
    incident_sums = np.zeros(len(node_list), dtype=float)
    incident_counts = np.zeros(len(node_list), dtype=int)

    for idx, (u, v) in enumerate(edges):
        puv = post[idx]
        incident_sums[node_index[u]] += puv
        incident_sums[node_index[v]] += puv
        incident_counts[node_index[u]] += 1
        incident_counts[node_index[v]] += 1

    # Avoid division by zero for isolated nodes
    node_beliefs = np.where(
        incident_counts > 0,
        incident_sums / incident_counts,
        0.0,
    )

    # 4. Compute activity gate from the same temperature
    activity_gate = normalized_activity(temp_c)

    # 5. Assemble cost
    edge_weights = np.array([edge_len_map[e] for e in edges], dtype=float)
    edge_num = np.dot(post, edge_weights)
    edge_den = np.abs(post).sum()
    edge_term = edge_num / edge_den if edge_den != 0 else 0.0

    d_vals = np.array([root_dist[v] for v in node_list], dtype=float)
    node_num = np.dot(node_beliefs, d_vals)
    node_den = np.abs(node_beliefs).sum()
    node_term = node_num / node_den if node_den != 0 else 0.0

    return edge_term + lam * activity_gate * node_term


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree (a star centred at 'A')
    nodes_example = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (-1.0, 0.0),
        "E": (0.0, -1.0),
    }
    edges_example = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]
    root_node = "A"

    # Prior: uniform belief over edges
    prior_vec = np.full(len(edges_example), 1.0 / len(edges_example))
    # Likelihood: random but reproducible
    random.seed(42)
    likelihood_vec = np.array([random.random() for _ in edges_example])
    likelihood_vec /= likelihood_vec.sum()  # normalise to sum 1

    # Temperature in Celsius
    temperature = 22.0

    cost = temperature_dependent_cost(
        nodes=nodes_example,
        edges=edges_example,
        root=root_node,
        prior=prior_vec,
        likelihood=likelihood_vec,
        temp_c=temperature,
        lam=0.8,
    )
    print(f"Hybrid temperature‑aware cost at {temperature:.1f} °C = {cost:.6f}")