# DARWIN HAMMER — match 773, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:30:56Z

"""
Hybrid algorithm fusing hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py and 
hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py.

The mathematical bridge between the two structures is built by applying the Fisher information 
score to the decision hygiene scoring system of the second parent, and using the resulting 
information density to weight the expected cost of the minimum-cost tree computed using 
Bayesian update.

This hybrid system integrates the strengths of both parents: the Fisher information score 
for directional parameters, and the minimum-cost tree with Bayesian update for decision 
making under uncertainty.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


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


def hybrid_decision_hygiene(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    theta: float,
    center: float,
    width: float,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float], np.ndarray]:
    """
    Hybrid decision hygiene using Fisher information score.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    posterior : np.ndarray
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # Compute Fisher information score
    fisher_inf = fisher_score(theta, center, width)

    # Compute prior probabilities
    prior = np.array([1.0 / len(nodes) for _ in nodes])

    # Compute likelihoods
    likelihood = np.array([math.exp(-dist[node]) for node in nodes])

    # Compute marginal probabilities
    marginal = bayes_marginal(prior, likelihood, 0.1)

    # Compute posterior probabilities
    posterior = bayes_update(prior, likelihood, marginal)

    # Weight posterior probabilities using Fisher information score
    weighted_posterior = posterior * fisher_inf

    return adj, edge_len, dist, weighted_posterior


def smoke_test():
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    theta = 0.5
    center = 0.5
    width = 0.1

    adj, edge_len, dist, posterior = hybrid_decision_hygiene(nodes, edges, root, theta, center, width)

    print("Adjacency:", adj)
    print("Edge lengths:", edge_len)
    print("Distances:", dist)
    print("Posterior probabilities:", posterior)


if __name__ == "__main__":
    smoke_test()