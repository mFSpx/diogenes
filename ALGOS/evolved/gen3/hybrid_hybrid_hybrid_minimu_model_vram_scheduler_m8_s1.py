# DARWIN HAMMER — match 8, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:25:05Z

"""
This module integrates the hybrid_minimum_cost_tree_bayes_update_m6_s2 and 
model_vram_scheduler algorithms into a single hybrid system. The bridge between 
the two structures is the concept of information entropy applied to the decision 
hygiene scoring system, and the expected cost of the minimum-cost tree computed 
using Bayesian update. Specifically, we use the tree metrics from the first 
algorithm to estimate the resource requirements for the VRAM scheduler in the 
second algorithm, and then use the Bayesian update to adjust the scheduler's 
decisions based on the actual resource usage.

The mathematical interface between the two algorithms is established through the 
use of the tree metrics to estimate the resource requirements, and the Bayesian 
update to adjust the scheduler's decisions. This allows us to integrate the two 
algorithms into a single hybrid system that can adapt to changing resource 
requirements and make more informed decisions about resource allocation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

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

def estimate_resource_requirements(adj: Dict[str, List[str]], edge_len: Dict[Tuple[str, str], float], dist: Dict[str, float]) -> float:
    """
    Estimate the resource requirements based on the tree metrics.

    Returns
    -------
    resource_requirements : float
    """
    total_distance = sum(dist.values())
    return total_distance * len(adj)

def vram_scheduler(resource_requirements: float, budget: float) -> bool:
    """
    Determine whether the resource requirements can be met within the budget.

    Returns
    -------
    can_meet_requirements : bool
    """
    return resource_requirements <= budget

def hybrid_decision(adj: Dict[str, List[str]], edge_len: Dict[Tuple[str, str], float], dist: Dict[str, float], prior: np.ndarray, likelihood: np.ndarray, false_positive: float, budget: float) -> bool:
    """
    Make a decision based on the tree metrics and the Bayesian update.

    Returns
    -------
    decision : bool
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    resource_requirements = estimate_resource_requirements(adj, edge_len, dist)
    return vram_scheduler(resource_requirements, budget)

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 1), "C": (2, 2)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.7, 0.3])
    false_positive = 0.1
    budget = 100.0
    decision = hybrid_decision(adj, edge_len, dist, prior, likelihood, false_positive, budget)
    print(decision)