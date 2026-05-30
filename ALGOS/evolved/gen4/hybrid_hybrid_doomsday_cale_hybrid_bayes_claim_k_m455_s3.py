# DARWIN HAMMER — match 455, survivor 3
# gen: 4
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s0.py (gen1)
# parent_b: hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s0.py (gen3)
# born: 2026-05-29T23:29:02Z

"""
Hybrid Doomsday‑Bayes Tree Metric
---------------------------------
Parent A: `hybrid_doomsday_calendar_gini_coefficient_m49_s0.py`
Parent B: `hybrid_bayes_claim_kernel_hybrid_ternary_route_m84_s0.py`

Mathematical bridge
~~~~~~~~~~~~~~~~~~~
- Parent A provides a *weekday distribution* (counts per weekday) and a
  Gini‑coefficient that quantifies its inequality.
- Parent B supplies a Bayesian update rule and a minimum‑cost tree evaluator.

The fusion treats the weekday distribution as a *categorical prior* over the
seven weekday nodes of a circular graph.  New observations (e.g. a sampled
sequence of weekdays) are incorporated via a Dirichlet‑multinomial Bayesian
update, yielding posterior probabilities for each node.  Those posterior
probabilities weight the edges of the ring‑graph when the tree‑cost is
computed.  The Gini coefficient of the posterior distribution is then used
as an uncertainty‑inflation factor on the tree cost, producing a single hybrid
metric that reflects both distributional inequality and expected routing cost.
"""

from __future__ import annotations
import numpy as np
import math
import random
import sys
import pathlib
import datetime as dt
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A utilities (doomsday calendar + Gini)
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where Monday=0 … Sunday=6."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient for a non‑negative 1‑D array."""
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    # classic Gini formula
    cumulative = np.cumsum(xs)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def weekday_counts(year: int, month: int) -> np.ndarray:
    """Counts of each weekday (0‑6) for the whole month."""
    # number of days in month
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)
    days_in_month = (next_month - dt.date(year, month, 1)).days
    counts = np.zeros(7, dtype=int)
    for day in range(1, days_in_month + 1):
        wd = doomsday(year, month, day)
        counts[wd] += 1
    return counts


# ----------------------------------------------------------------------
# Parent B utilities (Bayesian update + tree cost)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    weight_factor: float = 0.2,
) -> float:
    """
    Minimum‑cost tree cost for an undirected graph.
    Edge contribution = Euclidean length * weight_factor.
    The algorithm performs a depth‑first walk from ``root`` and sums the
    weighted lengths of traversed edges (each edge counted once).
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    total_len = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        total_len += length(nodes[a], nodes[b]) * weight_factor

    visited = set()
    stack = [root]
    cost = 0.0
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in adj[cur]:
            if nxt not in visited:
                cost += length(nodes[cur], nodes[nxt]) * weight_factor
                stack.append(nxt)
    return cost + total_len  # include the global material term


# ----------------------------------------------------------------------
# Hybrid layer
# ----------------------------------------------------------------------
def bayesian_update_counts(
    prior_counts: np.ndarray,
    evidence_counts: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Dirichlet‑multinomial update.
    ``alpha`` is the pseudo‑count added to each category (symmetric prior).
    Returns posterior *probabilities* for the seven weekdays.
    """
    if prior_counts.shape != (7,) or evidence_counts.shape != (7,):
        raise ValueError("Both count vectors must have length 7.")
    # Convert prior counts to a proper prior (add alpha smoothing)
    prior = prior_counts + alpha
    # Incorporate evidence
    posterior = prior + evidence_counts
    total = posterior.sum()
    if total == 0:
        return np.full(7, 1 / 7)
    return posterior / total


def build_circular_weekday_graph(posterior: np.ndarray) -> Tuple[Dict[str, Point], List[Edge]]:
    """
    Constructs a ring graph where each node corresponds to a weekday.
    Node positions lie on the unit circle; edge list connects successive
    weekdays (0‑1‑2‑…‑6‑0).  The posterior vector is attached implicitly
    (used later to weight the tree cost).
    """
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    nodes: Dict[str, Point] = {}
    for i, name in enumerate(names):
        angle = 2 * math.pi * i / 7
        nodes[name] = (math.cos(angle), math.sin(angle))
    edges: List[Edge] = []
    for i in range(7):
        a = names[i]
        b = names[(i + 1) % 7]
        edges.append((a, b))
    return nodes, edges


def hybrid_tree_cost(
    year: int,
    month: int,
    observed_counts: np.ndarray,
    root: str = "Mon",
    alpha: float = 1.0,
    weight_factor: float = 0.2,
) -> float:
    """
    End‑to‑end hybrid metric:
    1. Compute the *prior* weekday distribution for the calendar month.
    2. Bayesian‑update it with ``observed_counts``.
    3. Build a circular weekday graph.
    4. Compute the tree cost using the posterior as a scaling factor.
    5. Inflate the cost by (1 + Gini(posterior)) to reflect inequality.
    """
    prior_counts = weekday_counts(year, month)
    posterior = bayesian_update_counts(prior_counts, observed_counts, alpha)

    nodes, edges = build_circular_weekday_graph(posterior)

    # Base tree cost (ignores posterior)
    base_cost = tree_cost(nodes, edges, root, weight_factor)

    # Modulate edge contribution by posterior probabilities:
    # heavier probability ⇒ lower effective cost (more certain path)
    prob_factor = np.mean(posterior)  # simple scalar proxy
    modulated_cost = base_cost * (1.0 - prob_factor)

    # Gini‑based uncertainty inflation
    gini = gini_coefficient(posterior)
    hybrid_metric = modulated_cost * (1.0 + gini)
    return hybrid_metric


def simulate_random_weekday_counts(num_days: int) -> np.ndarray:
    """Generate a random observation vector of weekday counts."""
    draws = np.random.randint(0, 7, size=num_days)
    counts = np.bincount(draws, minlength=7)
    return counts.astype(int)


def hybrid_demo(year: int, month: int, num_observed_days: int = 30) -> None:
    """
    Demonstrates the hybrid workflow:
    - Simulate random observations.
    - Compute the hybrid tree metric.
    - Print intermediate values for inspection.
    """
    obs = simulate_random_weekday_counts(num_observed_days)
    metric = hybrid_tree_cost(year, month, obs)
    print(f"Month {year}-{month:02d}")
    print(f"Prior weekday counts      : {weekday_counts(year, month)}")
    print(f"Observed weekday counts   : {obs}")
    posterior = bayesian_update_counts(weekday_counts(year, month), obs)
    print(f"Posterior probabilities   : {posterior}")
    print(f"Gini of posterior         : {gini_coefficient(posterior):.4f}")
    print(f"Hybrid tree metric        : {metric:.6f}")


if __name__ == "__main__":
    # Smoke test – uses June 2022 and a 30‑day random observation set.
    hybrid_demo(2022, 6, num_observed_days=30)