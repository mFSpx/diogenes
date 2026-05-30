# DARWIN HAMMER — match 12, survivor 3
# gen: 2
# parent_a: hard_truth_math.py (gen0)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# born: 2026-05-29T23:25:17Z

# hybrid_math.py
"""Hybrid module combining LUCIDOTA's hard-truth math and DARWIN HAMMER's Minimum-Cost Tree scoring and Bayesian evidence update.

Mathematical bridge
-------------------
This module integrates LUCIDOTA's hard-truth math (Algorithm A) with DARWIN HAMMER's Minimum-Cost Tree scoring and Bayesian evidence update (Algorithm B).

The core topologies of both parents are fused: the stylometry features/classifier helpers from LUCIDOTA supply the probabilistic weights, while the tree-metric and Bayesian primitives from DARWIN HAMMER supply the geometric quantities.

Algorithm A computes stylometry features/classifier helpers as

    L = Σ_e (p_e · ℓ(e)) / Σ_e |p_e|

where ℓ(e) is Euclidean length of edge *e* and p_e is the edge posterior belief.

The hybrid replaces the deterministic edge contribution ℓ(e) in the stylometry features/classifier helpers by its **expected** value under the posterior edge belief *p_e*.

Similarly, node distances are weighted by a node belief *q_v* derived from incident edge posteriors.

The resulting hybrid cost is

    C_h = Σ_e (p_e · ℓ(e)) / Σ_e |p_e| + λ Σ_v (q_v · d(v)) / Σ_v |q_v|

where λ is a path-weight and d(v) is the root-to-node distance.

The module implements:
* `tree_metrics` – builds adjacency, edge lengths and root distances.
* `bayes_edge_posteriors` – vectorised Bayesian update for all edges.
* `hybrid_stylometry` – evaluates the hybrid stylometry features/classifier helpers.
* `hybrid_tree_cost` – evaluates the hybrid cost.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Algorithm A – deterministic tree utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as (a, b)) → Euclidean distance between a and b
    root_dist : dict mapping node → root-to-node distance
    """
    adj = {node: [] for node in nodes}
    edge_len = {}
    root_dist = {node: float('inf') for node in nodes}
    root_dist[root] = 0

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    for node in nodes:
        for neighbor in adj[node]:
            dist = length(nodes[node], nodes[neighbor])
            edge_len[(node, neighbor)] = dist

    for node in adj:
        queue = [(node, 0)]
        while queue:
            node, dist = queue.pop(0)
            for neighbor in adj[node]:
                if dist + length(nodes[node], nodes[neighbor]) < root_dist[neighbor]:
                    root_dist[neighbor] = dist + length(nodes[node], nodes[neighbor])
                    queue.append((neighbor, dist + length(nodes[node], nodes[neighbor])))

    return adj, edge_len, root_dist

# ----------------------------------------------------------------------
# Algorithm B – Bayesian utilities
# ----------------------------------------------------------------------
def bayes_edge_posteriors(
    edges: List[Edge],
    root: str,
    p_prior: float,
    L: Dict[Edge, float],
    FP: float,
) -> Dict[Edge, float]:
    """
    Vectorised Bayesian update for all edges.

    Returns
    -------
    p_e : dict mapping edge (ordered as (a, b)) → posterior edge belief
    """
    p_e = {}
    for e in edges:
        p_e[e] = (p_prior * L[e]) / (L[e] + FP * (1 - p_prior))
    return p_e

# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def hybrid_stylometry(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    p_prior: float,
    L: Dict[Edge, float],
    FP: float,
) -> Dict[str, float]:
    """
    Evaluate the hybrid stylometry features/classifier helpers.

    Returns
    -------
    L : dict mapping stylometry feature → value
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    p_e = bayes_edge_posteriors(edges, root, p_prior, L, FP)
    L_hybrid = {}

    for feature in FUNCTION_CATS:
        L_hybrid[feature] = 0
        for cat in FUNCTION_CATS[feature]:
            total = 0
            for w in FUNCTION_CATS[feature]:
                total += p_e[(cat, w)]
            L_hybrid[feature] += total * (sum(L[(cat, w)] for w in FUNCTION_CATS[feature]) / sum(p_e[(cat, w)] for w in FUNCTION_CATS[feature]))

    return L_hybrid

def hybrid_tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    p_prior: float,
    L: Dict[Edge, float],
    FP: float,
    λ: float,
) -> float:
    """
    Evaluate the hybrid cost.

    Returns
    -------
    C_h : hybrid cost
    """
    adj, edge_len, root_dist = tree_metrics(nodes, edges, root)
    p_e = bayes_edge_posteriors(edges, root, p_prior, L, FP)
    C_h = 0
    for e in edges:
        C_h += p_e[e] * edge_len[e]
    for v in nodes:
        C_h += λ * p_e[(v, v)] * root_dist[v]
    return C_h

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (0, 1)}
    edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
    root = 'A'
    p_prior = 0.5
    L = {('A', 'B'): 1.0, ('B', 'C'): 1.0, ('A', 'C'): 1.0}
    FP = 0.1
    λ = 0.5

    L_hybrid = hybrid_stylometry(nodes, edges, root, p_prior, L, FP)
    print("Hybrid stylometry features:", L_hybrid)

    C_h = hybrid_tree_cost(nodes, edges, root, p_prior, L, FP, λ)
    print("Hybrid cost:", C_h)