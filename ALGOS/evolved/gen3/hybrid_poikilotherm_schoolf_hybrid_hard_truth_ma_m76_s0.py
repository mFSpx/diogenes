# DARWIN HAMMER — match 76, survivor 0
# gen: 3
# parent_a: poikilotherm_schoolfield.py (gen0)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py (gen2)
# born: 2026-05-29T23:26:42Z

"""Hybrid Poikilotherm-Tree Model

This module fuses two parent algorithms:

* **Algorithm A** – `poikilotherm_schoolfield.py` – provides a
  temperature‑dependent activity function `normalized_activity` based on the
  Schoolfield‐Rollinson rate equation.
* **Algorithm B** – `hybrid_hard_truth_math_hybrid_minimum_cost__m12_s3.py` –
  supplies tree‑metric utilities, Bayesian edge‑posterior updates and a hybrid
  cost formulation.

**Mathematical bridge**

For every edge *e* we treat the Schoolfield activity `a(e)` (computed from a
temperature that is mapped from the geometric length of the edge) as a
temperature‑dependent scaling of the Bayesian prior belief `π_e`.  The posterior
belief becomes  

\[
p_e = \frac{π_e \; a(e)}{\sum_{e'} π_{e'} \; a(e')} .
\]

Thus the *expected* edge length used in the stylometry term of the hybrid
cost is the Schoolfield‑weighted expectation  

\[
\mathbb{E}[ℓ] = \frac{\sum_e p_e \, ℓ(e)}{\sum_e |p_e|},
\]

exactly the form of Algorithm B, but now the probabilities are temperature‑
modulated by Algorithm A.  Node beliefs `q_v` are derived from incident edge
posteriors, and the full hybrid cost is  

\[
C_{\text{hyb}} = \mathbb{E}[ℓ] \;+\; λ \,
\frac{\sum_v q_v \, d(v)}{\sum_v |q_v|},
\]

where `λ` itself is taken as the normalized activity at the mean operating
temperature of the system.

The three public functions below demonstrate this integration:

* `temperature_weighted_posteriors` – Bayesian edge posteriors scaled by
  Schoolfield activity.
* `hybrid_stylometry` – expected edge length under the temperature‑weighted
  posteriors.
* `hybrid_tree_cost` – full hybrid cost including the node‑distance term.

All code is pure Python 3, uses only the standard library and NumPy, and can be
executed directly."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Schoolfield poikilotherm utilities
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


def c_to_k(celsius: float) -> float:
    """Celsius → Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‐Rollinson temperature dependent rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def normalized_activity(
    temp_c: float,
    low_c: float = 5.0,
    high_c: float = 40.0,
    samples: int = 141,
) -> float:
    """
    Map an observed operating temperature to a 0‥1 activity gate.
    The activity is the rate normalised by the maximal rate observed in the
    temperature interval ``[low_c, high_c]``.
    """
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(
        developmental_rate(
            c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)),
            params,
        )
        for i in range(samples)
    )
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))


# ----------------------------------------------------------------------
# Algorithm B – Tree utilities and Bayesian scoring
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
    Build adjacency, edge‑length map and root‑to‑node distances.

    Returns
    -------
    adjacency : dict mapping node → list of neighbour nodes
    edge_len  : dict mapping (parent, child) → Euclidean length
    root_dist : dict mapping node → distance from *root* along the unique path
    """
    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Edge, float] = {}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        edge_len[(u, v)] = edge_len[(v, u)] = length(nodes[u], nodes[v])

    # BFS to compute root distances and orient edges (parent→child)
    root_dist: Dict[str, float] = {root: 0.0}
    parent: Dict[str, str] = {}
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nxt in adjacency[cur]:
            if nxt in root_dist:
                continue
            root_dist[nxt] = root_dist[cur] + edge_len[(cur, nxt)]
            parent[nxt] = cur
            queue.append(nxt)

    # orient edges consistently as (parent, child)
    oriented_edge_len: Dict[Edge, float] = {}
    for child, par in parent.items():
        oriented_edge_len[(par, child)] = edge_len[(par, child)]

    return adjacency, oriented_edge_len, root_dist


def temperature_weighted_posteriors(
    edges: List[Edge],
    edge_lengths: Dict[Edge, float],
    base_prior: float = 1.0,
    temp_range: Tuple[float, float] = (5.0, 40.0),
    params: SchoolfieldParams = SchoolfieldParams(),
) -> Dict[Edge, float]:
    """
    Compute Bayesian edge posteriors where each edge's prior is multiplied by a
    temperature‑dependent activity derived from its geometric length.

    The edge temperature is obtained by linearly mapping the edge length onto
    the supplied Celsius interval ``temp_range`` and then passing it through
    ``normalized_activity`` (Algorithm A).  The posterior is the normalised
    product of the uniform prior and the activity.
    """
    # Uniform prior for every edge
    priors = np.full(len(edges), base_prior, dtype=float)

    # Map lengths to temperatures (Celsius)
    lengths = np.array([edge_lengths[e] for e in edges], dtype=float)
    min_len, max_len = lengths.min(), lengths.max()
    # Avoid division by zero for degenerate trees
    if max_len - min_len < 1e-12:
        temps_c = np.full_like(lengths, (temp_range[0] + temp_range[1]) / 2.0)
    else:
        temps_c = temp_range[0] + (lengths - min_len) * (temp_range[1] - temp_range[0]) / (max_len - min_len)

    # Activity per edge using Algorithm A
    vec_norm_activity = np.vectorize(lambda t: normalized_activity(t, low_c=temp_range[0], high_c=temp_range[1]))
    activities = vec_norm_activity(temps_c)

    # Posterior proportional to prior * activity
    unnorm_post = priors * activities
    total = unnorm_post.sum()
    if total == 0.0:
        # fallback to uniform posterior
        post = np.full_like(unnorm_post, 1.0 / len(unnorm_post))
    else:
        post = unnorm_post / total

    return {e: float(p) for e, p in zip(edges, post)}


def hybrid_stylometry(
    edges: List[Edge],
    edge_lengths: Dict[Edge, float],
    posteriors: Dict[Edge, float],
) -> float:
    """
    Expected edge length under temperature‑weighted posteriors.

    Implements the numerator/denominator pattern of Algorithm B but with the
    posterior probabilities supplied by ``temperature_weighted_posteriors``.
    """
    numer = sum(posteriors[e] * edge_lengths[e] for e in edges)
    denom = sum(abs(posteriors[e]) for e in edges)
    return numer / denom if denom != 0.0 else 0.0


def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    lambda_weight: float = 1.0,
    temp_range: Tuple[float, float] = (5.0, 40.0),
    params: SchoolfieldParams = SchoolfieldParams(),
) -> float:
    """
    Full hybrid cost:

        C_h = E[ℓ] + λ * ( Σ_v q_v·d(v) / Σ_v |q_v| )

    where

    * ``E[ℓ]`` is the temperature‑weighted expected edge length,
    * ``q_v`` is the node belief defined as the sum of posterior probabilities of
      incident edges,
    * ``d(v)`` is the root‑to‑node distance,
    * ``λ`` is taken as the normalized activity at the mean temperature of the
      system (the mean of the edge‑derived temperatures).
    """
    adjacency, oriented_edge_len, root_dist = tree_metrics(nodes, edges, root)

    # Posterior probabilities (temperature weighted)
    post = temperature_weighted_posteriors(edges, oriented_edge_len, temp_range=temp_range, params=params)

    # Expected edge length term
    exp_len = hybrid_stylometry(edges, oriented_edge_len, post)

    # Node beliefs q_v = sum of incident edge posteriors
    node_belief: Dict[str, float] = {n: 0.0 for n in nodes}
    for (u, v), prob in post.items():
        node_belief[u] += prob
        node_belief[v] += prob

    # Node‑distance term
    numer_node = sum(node_belief[n] * root_dist[n] for n in nodes)
    denom_node = sum(abs(node_belief[n]) for n in nodes)
    node_term = numer_node / denom_node if denom_node != 0.0 else 0.0

    # λ as activity at the mean edge‑derived temperature
    # Re‑use the temperature mapping from ``temperature_weighted_posteriors``
    lengths = np.array([oriented_edge_len[e] for e in edges], dtype=float)
    min_len, max_len = lengths.min(), lengths.max()
    if max_len - min_len < 1e-12:
        mean_temp_c = (temp_range[0] + temp_range[1]) / 2.0
    else:
        temps_c = temp_range[0] + (lengths - min_len) * (temp_range[1] - temp_range[0]) / (max_len - min_len)
        mean_temp_c = float(temps_c.mean())
    lam = normalized_activity(mean_temp_c, low_c=temp_range[0], high_c=temp_range[1])

    return exp_len + lam * node_term


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _build_sample_tree() -> Tuple[Dict[str, Point], List[Edge], str]:
    """Create a tiny rooted tree for quick testing."""
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
        "E": (2.0, 0.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")]
    root = "A"
    return nodes, edges, root


if __name__ == "__main__":
    # Build a simple tree
    nodes, edges, root = _build_sample_tree()

    # Compute hybrid cost
    cost = hybrid_tree_cost(nodes, edges, root)

    # Print a concise report
    print(f"Hybrid tree cost: {cost:.6f}")

    # Additional sanity checks (no exceptions)
    adjacency, edge_len, root_dist = tree_metrics(nodes, edges, root)
    post = temperature_weighted_posteriors(edges, edge_len)
    exp_len = hybrid_stylometry(edges, edge_len, post)
    print(f"Expected edge length (temp‑weighted): {exp_len:.6f}")

    # Verify that posteriors sum to 1
    total_post = sum(post.values())
    print(f"Sum of posterior probabilities: {total_post:.6f}")

    sys.exit(0)