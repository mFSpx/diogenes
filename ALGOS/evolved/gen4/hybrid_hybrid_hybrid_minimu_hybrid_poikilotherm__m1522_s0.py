# DARWIN HAMMER — match 1522, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py (gen3)
# born: 2026-05-29T23:36:57Z

"""
This module fuses the hybrid_minimum_cost_tree_bayes_update_m6_s2 and
hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1 algorithms into a single hybrid system.
The bridge between the two structures is the concept of temperature-dependent
developmental rate ρ(T) applied to the Bayesian edge-posterior updates, and the
expected cost of the minimum-cost tree computed using Bayesian update and
temperature-aware hybrid cost.

References:
- Parent A: hybrid_minimum_cost_tree_bayes_update_m6_s2.py
- Parent B: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py
"""

import numpy as np
import math
import random
import sys
import pathlib

from typing import Any, Dict, List, Tuple

# Parent A - Poikilotherm rate utilities
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temperature: float) -> float:
    """Compute temperature-dependent developmental rate ρ(T)."""
    # Schoolfield-Rollinson model
    k = c_to_k(temperature)
    rho = 2.0 * np.exp(-k / 100.0) + 0.5
    return rho

def normalized_activity(temperature: float) -> float:
    """Compute temperature-dependent normalized activity a(T)."""
    # Assume a(T) ∈ [0,1]
    a = developmental_rate(temperature) / max(developmental_rate(temperature), 1.0)
    return a

# Parent A - Tree geometry utilities
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

# Parent B - Bayesian update and hybrid cost
def bayes_edge_posteriors(
    prior: np.ndarray, likelihood: np.ndarray, false_positive: float
) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def temperature_scaled_posteriors(
    temperature: float,
    prior: np.ndarray,
    likelihood: np.ndarray,
    false_positive: float,
) -> np.ndarray:
    """
    Temperature-dependent edge posterior update.
    """
    rho = developmental_rate(temperature)
    return bayes_edge_posteriors(prior, likelihood, false_positive) * rho

def hybrid_tree_cost(
    temperature: float,
    edge_posteriors: np.ndarray,
    edge_lengths: np.ndarray,
    node_beliefs: np.ndarray,
    path_weight: float,
) -> float:
    """
    Temperature-aware hybrid cost functional.
    """
    a = normalized_activity(temperature)
    cost = (np.sum(edge_posteriors * edge_lengths) / np.sum(np.abs(edge_posteriors)))
    cost += path_weight * a * (np.sum(node_beliefs * np.sum(edge_posteriors, axis=1)) / np.sum(np.abs(node_beliefs)))
    return cost

# Smoke test
if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"

    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Distances:", dist)

    temperature = 25.0
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.8, 0.2])
    false_positive = 0.1
    edge_posteriors = temperature_scaled_posteriors(temperature, prior, likelihood, false_positive)
    print("Temperature-scaled edge posteriors:", edge_posteriors)

    node_beliefs = np.array([0.6, 0.7, 0.3])
    path_weight = 0.5
    cost = hybrid_tree_cost(temperature, edge_posteriors, np.array([1.0, 1.0]), node_beliefs, path_weight)
    print("Hybrid tree cost:", cost)