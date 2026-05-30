# DARWIN HAMMER — match 1162, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py (gen4)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (gen1)
# born: 2026-05-29T23:33:07Z

"""Hybrid Tree‑Gini‑Doomsday Algorithm
===================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – *hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m318_s0.py* –  
  builds a Euclidean minimum‑cost tree, computes root‑to‑node distances and
  performs a Bayesian update on those distances.

* **Parent B** – *hybrid_doomsday_calendar_gini_coefficient_m49_s1.py* –  
  evaluates the Gini coefficient of a numeric distribution and combines it
  with the weekday obtained from the Doomsday algorithm.

**Mathematical bridge**

The Bayesian posterior probabilities obtained from the tree distances are
interpreted as *weights* for the edge‑length distribution of the tree.
Those weighted edge lengths constitute a non‑negative data set on which the
Gini coefficient can be evaluated.  
The resulting inequality measure is then blended with the weekday (0‑6)
derived from the Doomsday algorithm, using the same weighting scheme that
parent B employed (weekday‑dependent weight).  The final hybrid metric
captures structural information of the tree, probabilistic confidence from
the Bayesian update, and temporal context from the calendar.

The implementation below provides a self‑contained, executable Python 3
module that demonstrates this fusion through three public functions and a
smoke‑test block.
"""

import math
import random
import sys
import pathlib
import datetime as dt
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
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
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → cumulative distance from *root*
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]  # undirected access convenience

    # Depth‑first traversal to accumulate distances from the root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                stack.append(nxt)

    return adj, edge_len, dist


def bayesian_update(
    node_dist: Dict[str, float],
    sigma: float = 1.0,
) -> Dict[str, float]:
    """
    Perform a simple Gaussian Bayesian update on node distances.

    Prior is uniform; likelihood ∝ exp(‑d² / (2σ²)).
    The posterior is normalised to sum to 1 and interpreted as a weight
    for each node.

    Parameters
    ----------
    node_dist : mapping node → distance from root
    sigma : standard deviation of the Gaussian likelihood

    Returns
    -------
    posterior : mapping node → posterior probability (weight)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    unnorm = {n: math.exp(- (d ** 2) / (2 * sigma ** 2)) for n, d in node_dist.items()}
    total = sum(unnorm.values())
    if total == 0:
        # fallback to uniform weights
        n = len(node_dist)
        return {n: 1.0 / n for n in node_dist}
    return {n: p / total for n, p in unnorm.items()}


def weighted_edge_distribution(
    edges: List[Tuple[str, str]],
    edge_len: Dict[Tuple[str, str], float],
    node_weights: Dict[str, float],
) -> List[float]:
    """
    Produce a list of edge‑length values weighted by the posterior probabilities
    of their incident nodes.  For each edge (u, v) we assign the weight
    w = (weight_u + weight_v) / 2 and return w * length.

    This creates a non‑negative numeric series suitable for Gini analysis.
    """
    values: List[float] = []
    for u, v in edges:
        w = (node_weights.get(u, 0.0) + node_weights.get(v, 0.0)) / 2.0
        values.append(w * edge_len[(u, v)])
    return values


def gini_coefficient(values: Iterable[float]) -> float:
    """Calculate the Gini coefficient of a non‑negative value set."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += (2 * i - n - 1) * x
    return cumulative / (n * sum(xs))


def doomsday_weekday(year: int, month: int, day: int) -> int:
    """
    Return the weekday (0 = Monday … 6 = Sunday) for the given Gregorian date
    using Python's datetime (the Doomsday algorithm is conceptually the same).
    """
    return dt.date(year, month, day).weekday()


def hybrid_tree_gini_doomsday(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    date: Tuple[int, int, int],
    sigma: float = 1.0,
) -> float:
    """
    Core hybrid operation.

    1. Build tree metrics (adjacency, edge lengths, root distances).
    2. Apply Bayesian update to obtain node weights.
    3. Generate a weighted edge‑length distribution.
    4. Compute its Gini coefficient.
    5. Obtain the weekday for *date*.
    6. Blend Gini and weekday using the weekday‑dependent weight
       w = weekday / 7 (identical to parent B).

    Returns the blended hybrid metric.
    """
    _, edge_len, node_dist = tree_metrics(nodes, edges, root)
    node_weights = bayesian_update(node_dist, sigma=sigma)
    weighted_vals = weighted_edge_distribution(edges, edge_len, node_weights)
    gini = gini_coefficient(weighted_vals)

    year, month, day = date
    weekday = doomsday_weekday(year, month, day)  # 0‑6
    weight = weekday / 7.0
    return weight * gini + (1.0 - weight) * weekday


def simulate_temporal_inequality_on_tree(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    start_date: Tuple[int, int, int],
    days: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    For each day in a window, compute the hybrid metric and return the series.
    This demonstrates how the tree‑based inequality evolves with temporal context.
    """
    year, month, day = start_date
    base_date = dt.date(year, month, day)
    results = []
    for i in range(days):
        cur_date = base_date + dt.timedelta(days=i)
        metric = hybrid_tree_gini_doomsday(
            nodes, edges, root, (cur_date.year, cur_date.month, cur_date.day), sigma
        )
        results.append(metric)
    return np.array(results)


if __name__ == "__main__":
    # Simple smoke test with a tiny tree and a date range.
    sample_nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    sample_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    root_node = "A"
    test_date = (2026, 5, 29)

    hybrid_value = hybrid_tree_gini_doomsday(
        sample_nodes, sample_edges, root_node, test_date, sigma=0.5
    )
    print(f"Hybrid metric for {test_date}: {hybrid_value:.6f}")

    series = simulate_temporal_inequality_on_tree(
        sample_nodes,
        sample_edges,
        root_node,
        start_date=(2026, 5, 1),
        days=7,
        sigma=0.5,
    )
    print("7‑day hybrid series:", series)