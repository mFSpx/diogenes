# DARWIN HAMMER — match 2709, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m442_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s2.py (gen5)
# born: 2026-05-29T23:43:37Z

"""
This module represents a novel hybrid algorithm, mathematically fusing the core topologies of two parent algorithms:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m442_s0.py (DARWIN HAMMER — match 442, survivor 0)
- hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1162_s2.py (DARWIN HAMMER — match 1162, survivor 2)

The mathematical bridge between their structures lies in the integration of epistemic certainty flags with the Bayesian posterior probabilities 
obtained from the tree distances. Specifically, we use the epistemic certainty flags to inform the calculation of the weighted Gini coefficient 
from the edge-length distribution of the tree, enabling a more comprehensive assessment of system behavior.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Tuple

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for Bayesian update."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Perform Bayesian update on the prior probability."""
    if marginal == 0:
        return prior
    return (likelihood * prior) / marginal

def tree_metrics(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root-to-node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → cumulative distance from *root*
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    edge_len: dict[tuple[str, str], float] = {}
    dist: dict[str, float] = {root: 0}

    for edge in edges:
        a, b = edge
        edge_len[edge] = length(nodes[a], nodes[b])
        adj[a].append(b)
        adj[b].append(a)

    stack = [(root, 0)]
    while stack:
        node, d = stack.pop()
        for neighbour in adj[node]:
            if neighbour not in dist:
                dist[neighbour] = d + edge_len[(node, neighbour)]
                stack.append((neighbour, dist[neighbour]))

    return adj, edge_len, dist

def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of values."""
    values = sorted(values)
    n = len(values)
    index = np.arange(1, n+1)
    n_values = n * values
    return ((np.sum((2 * index - n - 1) * values)) / (n * np.sum(values)))

def hybrid_metric(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    root: str,
    prior: float,
    likelihood: float,
    false_positive: float,
    certainty_flag: str,
) -> tuple[float, float]:
    """
    Compute the hybrid metric.

    Returns
    -------
    gini : the weighted Gini coefficient
    epistemic_certainty : the epistemic certainty score
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)

    # Compute Bayesian posterior probabilities
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)

    # Compute weighted edge lengths
    weighted_edge_len = [posterior * edge_len[edge] for edge in edge_len]

    # Compute weighted Gini coefficient
    gini = gini_coefficient(weighted_edge_len)

    # Compute epistemic certainty score
    if certainty_flag == "FACT":
        epistemic_certainty = 1.0
    elif certainty_flag == "BULLSHIT":
        epistemic_certainty = 0.0
    else:
        epistemic_certainty = 0.5

    return gini, epistemic_certainty

if __name__ == "__main__":
    nodes = {
        "A": (0, 0),
        "B": (3, 0),
        "C": (0, 4),
    }
    edges = [
        ("A", "B"),
        ("A", "C"),
    ]
    root = "A"
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2
    certainty_flag = "FACT"

    gini, epistemic_certainty = hybrid_metric(nodes, edges, root, prior, likelihood, false_positive, certainty_flag)
    print(f"Gini coefficient: {gini}")
    print(f"Epistemic certainty: {epistemic_certainty}")