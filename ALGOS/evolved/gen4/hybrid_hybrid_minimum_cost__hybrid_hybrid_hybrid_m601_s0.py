# DARWIN HAMMER — match 601, survivor 0
# gen: 4
# parent_a: hybrid_minimum_cost_tree_bayes_update_m6_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s2.py (gen3)
# born: 2026-05-29T23:30:04Z

"""
Hybrid Module: Fusing Hybrid Minimum Cost Tree Bayes Update and Hybrid Endpoint-Tropical Max-Plus Engine

This module combines the strengths of two parent algorithms:

*   **Parent A** – `hybrid_minimum_cost_tree_bayes_update_m6_s2.py`: A hybrid algorithm that integrates Minimum-Cost Tree scoring and Bayesian evidence update.
*   **Parent B** – `hybrid_hybrid_hybrid_endpoint_hybrid_hoeffding_tree_m44_s2.py`: A hybrid algorithm that fuses endpoint selection using a health score and tropical ReLU network evaluations with the Hoeffding bound.

The mathematical bridge between the two parents lies in the fact that the expected edge lengths and node distances in the Minimum-Cost Tree can be viewed as a linear region in the input space of a tropical ReLU network. The health scores of the endpoints can be used as weights for the edges in the Minimum-Cost Tree. The Hoeffding bound can be used to decide when to update the edge posteriors and node beliefs in the Minimum-Cost Tree.

The hybrid algorithm therefore:

1.  Computes the health-related quantities from the endpoint pool.
2.  Builds a tropical ReLU network that maps the endpoint health scores to linear regions in the input space.
3.  Uses the Hoeffding bound to decide when to update the edge posteriors and node beliefs.
4.  Evaluates the hybrid cost function using the updated posteriors and beliefs.

The three public functions below illustrate the hybrid workflow.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

@dataclass
class Endpoint:
    health_score: float
    recovery_priority: float

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r].

    Parameters
    ----------
    r : float
        Range of the random variable (max – min). Must be > 0.
    delta : float
        Desired failure probability, 0 < delta < 1.
    n : int
        Number of independent observations (must be > 0).

    Returns
    -------
    float
        The Hoeffding bound.
    """
    return r * math.sqrt((math.log(2 / delta)) / (2 * n))

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
    edge_len : dict mapping edge (ordered as tuple) → Euclidean length
    node_dist : dict mapping node → root distance
    """
    adj = {node: [] for node in nodes}
    edge_len = {}
    node_dist = {root: 0}

    for edge in edges:
        a, b = edge
        edge_len[edge] = length(nodes[a], nodes[b])
        adj[a].append(b)
        adj[b].append(a)

    stack = [root]
    while stack:
        node = stack.pop()
        for neighbour in adj[node]:
            if neighbour not in node_dist:
                node_dist[neighbour] = node_dist[node] + edge_len[(node, neighbour)]
                stack.append(neighbour)

    return adj, edge_len, node_dist

def bayes_edge_posteriors(
    prior: float, likelihood: float, false_positive_rate: float
) -> float:
    """Compute the posterior edge belief using Bayes' theorem."""
    return (prior * likelihood) / (likelihood * prior + false_positive_rate * (1 - prior))

def hybrid_tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: float,
    likelihood: float,
    false_positive_rate: float,
    lambda_: float,
) -> float:
    """
    Evaluate the hybrid cost function.

    Parameters
    ----------
    nodes : dict mapping node → coordinates
    edges : list of edges (ordered as tuples)
    root : root node
    prior : prior probability
    likelihood : likelihood
    false_positive_rate : false positive rate
    lambda_ : path weight

    Returns
    -------
    float
        The hybrid cost.
    """
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    edge_posteriors = {edge: bayes_edge_posteriors(prior, likelihood, false_positive_rate) for edge in edges}

    cost = 0
    for edge, posterior in edge_posteriors.items():
        cost += posterior * edge_len[edge]

    node_beliefs = {}
    for node in nodes:
        node_beliefs[node] = np.mean([edge_posteriors[(node, neighbour)] for neighbour in adj[node]])

    for node, belief in node_beliefs.items():
        cost += lambda_ * belief * node_dist[node]

    return cost

def tropical_relu_network(
    endpoints: List[Endpoint], health_score_weights: List[float]
) -> List[float]:
    """
    Evaluate a tropical ReLU network.

    Parameters
    ----------
    endpoints : list of endpoints
    health_score_weights : list of weights for health scores

    Returns
    -------
    list
        The output of the tropical ReLU network.
    """
    return [endpoint.health_score * weight for endpoint, weight in zip(endpoints, health_score_weights)]

if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    prior = 0.5
    likelihood = 0.8
    false_positive_rate = 0.2
    lambda_ = 1.0

    endpoints = [Endpoint(health_score=0.7, recovery_priority=0.3), Endpoint(health_score=0.4, recovery_priority=0.6)]
    health_score_weights = [0.5, 0.5]

    cost = hybrid_tree_cost(nodes, edges, root, prior, likelihood, false_positive_rate, lambda_)
    print("Hybrid cost:", cost)

    tropical_output = tropical_relu_network(endpoints, health_score_weights)
    print("Tropical ReLU network output:", tropical_output)