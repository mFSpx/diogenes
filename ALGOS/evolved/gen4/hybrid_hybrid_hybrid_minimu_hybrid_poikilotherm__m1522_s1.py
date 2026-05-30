# DARWIN HAMMER — match 1522, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py (gen3)
# born: 2026-05-29T23:36:57Z

"""
This module integrates the hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0 and 
hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1 algorithms into a single hybrid system.
The bridge between the two structures is the concept of temperature-dependent developmental rate 
ρ(T) from the Schoolfield-Rollinson model, which modulates the confidence in each edge posterior 
derived from the Bayesian update in the hybrid_minimum_cost_tree_bayes_update_m6_s2 algorithm.

The mathematical interface is established by interpreting the temperature-dependent rate ρ(T) 
as a global physiological scaling factor that adjusts the posterior update, and then using 
these temperature-adjusted posteriors in the hybrid cost functional.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1-p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P
    """
    return prior * likelihood / marginal

def developmental_rate(temperature: float) -> float:
    """
    Schoolfield-Rollinson model for temperature-dependent developmental rate.
    """
    return 1 / (1 + math.exp(-(temperature - 20) / 5))

def normalized_activity(temperature: float) -> float:
    """
    Normalized activity gate.
    """
    return 1 / (1 + math.exp(-(temperature - 25) / 2))

def temperature_scaled_posteriors(prior: np.ndarray, likelihood: np.ndarray, temperature: float) -> np.ndarray:
    """
    Temperature-adjusted posterior update.
    """
    rate = developmental_rate(temperature)
    return (likelihood * prior * rate) / (likelihood * prior * rate + (1 - prior) * rate)

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: np.ndarray,
    likelihood: np.ndarray,
    temperature: float,
) -> float:
    """
    Temperature-aware hybrid cost functional.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    posterior = temperature_scaled_posteriors(prior, likelihood, temperature)
    edge_cost = sum(posterior * edge_len[(a, b)] for a, b in edges)
    node_cost = sum(dist[n] for n in nodes)
    return edge_cost + normalized_activity(temperature) * node_cost

def temperature_dependent_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: np.ndarray,
    likelihood: np.ndarray,
    temperature: float,
) -> float:
    """
    End-to-end example combining all steps.
    """
    return hybrid_tree_cost(nodes, edges, root, prior, likelihood, temperature)

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    prior = np.array([0.5, 0.5, 0.5])
    likelihood = np.array([0.7, 0.7, 0.7])
    temperature = 20
    print(temperature_dependent_cost(nodes, edges, root, prior, likelihood, temperature))