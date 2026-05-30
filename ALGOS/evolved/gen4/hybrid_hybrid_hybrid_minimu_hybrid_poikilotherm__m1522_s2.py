# DARWIN HAMMER — match 1522, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py (gen3)
# born: 2026-05-29T23:36:57Z

"""
This module fuses the hybrid_minimum_cost_tree_bayes_update_m6_s2 and 
hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1 algorithms into a single 
hybrid system. The mathematical bridge between the two structures is the concept 
of temperature-dependent developmental rate ρ(T) applied to the Bayesian 
update of edge posteriors in the minimum-cost tree.

The temperature-dependent rate ρ(T) is interpreted as a global physiological 
scaling factor that modulates the confidence in each edge posterior. 
Concretely, the posterior update becomes  

p_e^{*}(T) = L_e * π_e * ρ(T) / Σ(L_e' * π_e' * ρ(T))

where π_e is the prior belief and L_e the likelihood derived from observations. 
The hybrid cost then uses these temperature-adjusted posteriors.
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
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

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

def c_to_k(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(kelvin: float) -> float:
    """Schoolfield–Rollinson temperature-dependent developmental rate ρ(T)."""
    t_ref = 298.15  # Reference temperature in Kelvin
    e_a = 54300  # Activation energy in J/mol
    r = 8.314  # Gas constant in J/(mol*K)
    return math.exp((e_a / r) * (1 / t_ref - 1 / kelvin))

def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Vectorised posterior:  P
    """
    return likelihood * prior / marginal

def temperature_scaled_posteriors(
    prior: np.ndarray, 
    likelihood: np.ndarray, 
    temperature_celsius: float
) -> np.ndarray:
    """
    Temperature-adjusted posteriors.
    """
    kelvin = c_to_k(temperature_celsius)
    rho = developmental_rate(kelvin)
    return (likelihood * prior * rho) / np.sum(likelihood * prior * rho)

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: np.ndarray,
    likelihood: np.ndarray,
    temperature_celsius: float,
) -> float:
    """
    Temperature-aware hybrid cost.
    """
    _, edge_len, dist = tree_metrics(nodes, edges, root)
    posteriors = temperature_scaled_posteriors(prior, likelihood, temperature_celsius)
    cost = np.sum(posteriors * np.array([edge_len[edge] for edge in edges])) / np.sum(np.abs(posteriors))
    return cost

if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0)
    }
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.7, 0.3])
    temperature_celsius = 25.0

    cost = hybrid_tree_cost(nodes, edges, root, prior, likelihood, temperature_celsius)
    print("Hybrid tree cost:", cost)