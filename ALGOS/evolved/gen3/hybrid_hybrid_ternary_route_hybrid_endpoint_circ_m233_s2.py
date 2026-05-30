# DARWIN HAMMER — match 233, survivor 2
# gen: 3
# parent_a: hybrid_ternary_router_hybrid_minimum_cost__m36_s1.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s0.py (gen1)
# born: 2026-05-29T23:27:45Z

"""HybridRouterBreaker
Combines:
- Parent A: ternary router with minimum‑cost tree evaluation and Bayesian edge
  probability updates.
- Parent B: EndpointCircuitBreaker whose failure threshold adapts to the
  morphology (sphericity & flatness) of a spatial structure.

Mathematical bridge:
The spatial distribution of tree nodes defines a 3‑D bounding box.
From its dimensions we compute a *Morphology* (length, width, height) and
derive sphericity S and flatness F (Parent B).  These shape descriptors are
used to modulate the circuit‑breaker failure threshold:

threshold = base_threshold * (1 + α·(1‑S) + β·F)

Simultaneously, each tree edge carries a Bayesian probability
`p(e) = likelihood·prior + false_positive` (Parent A).  The expected
tree‑cost incorporates these probabilities as weights, producing a unified
objective that balances geometric efficiency with reliability.

The module implements:
1. Geometry → morphology conversion.
2. Bayesian edge update.
3. Hybrid decision that returns (expected_cost, breaker_state).

"""

import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]          # 2‑D coordinates of a node
Edge = Tuple[str, str]               # connection between node identifiers
Morphology = Tuple[float, float, float]  # (length, width, height)

# ----------------------------------------------------------------------
# Parent A – tree cost and Bayesian update
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
) -> float:
    """
    Compute material + weighted path cost.
    If edge_weights are supplied they replace the geometric length
    with the Bayesian‑expected length.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        w = edge_weights.get((a, b), edge_weights.get((b, a), None)) if edge_weights else None
        material += w if w is not None else length(nodes[a], nodes[b])

    # BFS from root to accumulate path distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                edge_len = edge_weights.get((a, b), edge_weights.get((b, a), None)) if edge_weights else length(nodes[a], nodes[b])
                dist[b] = dist[a] + edge_len
                stack.append(b)

    return material + path_weight * sum(dist.values())


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian update for a single edge probability."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive


def update_edge_posteriors(
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    false_positive: float,
) -> Dict[Edge, float]:
    """Apply Bayesian marginal update to every edge."""
    post = {}
    for e, prior in priors.items():
        like = likelihoods.get(e, 0.0)
        post[e] = bayes_marginal(prior, like, false_positive)
    return post


# ----------------------------------------------------------------------
# Parent B – morphology and circuit breaker
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless measure of how sphere‑like a bounding box is."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Dimensionless measure of flatness (large planar extent vs thickness)."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical‑style index used by the serpentina morphology."""
    length, width, height = m
    mass = length * width * height  # proxy mass proportional to volume
    if mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(length, width, height)
    return (mass ** b) * math.exp(k * fi) / neck_lever


class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""

    def __init__(self, failure_threshold: int = 3):
        self.base_threshold = failure_threshold
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open

    def adapt_threshold(self, sphericity: float, flatness: float, α: float = 0.5, β: float = 0.3) -> None:
        """
        Adjust the failure threshold based on morphology.
        Higher flatness or lower sphericity makes the system more prone to failure,
        thus lowering the tolerance.
        """
        factor = 1.0 + α * (1.0 - sphericity) + β * flatness
        new_thresh = max(1, int(round(self.base_threshold * factor)))
        self.failure_threshold = new_thresh
        # If current failures already exceed new threshold, open the breaker
        self.open = self.failures >= self.failure_threshold


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def bounding_morphology(nodes: Dict[str, Point]) -> Morphology:
    """
    Derive a 3‑D morphology from 2‑D node coordinates.
    Height is approximated by the standard deviation of pairwise distances,
    providing a non‑zero thickness for planar layouts.
    """
    xs = np.array([p[0] for p in nodes.values()])
    ys = np.array([p[1] for p in nodes.values()])

    length = float(xs.max() - xs.min())
    width = float(ys.max() - ys.min())

    # Height proxy: spread of Euclidean distances from centroid
    cx, cy = xs.mean(), ys.mean()
    dists = np.hypot(xs - cx, ys - cy)
    height = float(dists.std() + 1e-6)  # avoid zero

    # Guard against degenerate cases
    length = max(length, 1e-6)
    width = max(width, 1e-6)
    height = max(height, 1e-6)

    return (length, width, height)


def hybrid_expected_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    false_positive: float,
    path_weight: float = 0.2,
) -> float:
    """
    Compute expected tree cost where each edge length is replaced by its
    Bayesian‑expected length (probability * geometric length).
    """
    posteriors = update_edge_posteriors(priors, likelihoods, false_positive)

    # Edge weight = posterior probability * geometric length
    edge_weights = {}
    for a, b in edges:
        prob = posteriors.get((a, b), posteriors.get((b, a), 0.0))
        geom_len = length(nodes[a], nodes[b])
        edge_weights[(a, b)] = prob * geom_len

    return tree_cost(nodes, edges, root, path_weight=path_weight, edge_weights=edge_weights)


def hybrid_decision(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    priors: Dict[Edge, float],
    likelihoods: Dict[Edge, float],
    false_positive: float,
) -> Tuple[float, bool]:
    """
    Perform a full hybrid evaluation:
    1. Derive morphology → sphericity & flatness.
    2. Adapt circuit breaker threshold.
    3. Compute expected cost with Bayesian‑weighted edges.
    Returns (expected_cost, breaker_allows).
    """
    # 1. Morphology & shape indices
    morph = bounding_morphology(nodes)
    S = sphericity_index(*morph)
    F = flatness_index(*morph)

    # 2. Circuit breaker adaptation
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    breaker.adapt_threshold(S, F)

    # 3. Expected cost
    exp_cost = hybrid_expected_cost(
        nodes, edges, root, priors, likelihoods, false_positive
    )

    # Decision rule: allow operation if cost below a dynamic ceiling
    cost_ceiling = 1.5 * exp_cost / (S + 0.1)  # arbitrary heuristic
    if exp_cost > cost_ceiling:
        breaker.record_failure()
    else:
        breaker.record_success()

    return exp_cost, breaker.allow()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small random tree
    random.seed(42)
    np.random.seed(42)

    # Nodes placed in a unit square
    nodes = {f"N{i}": (random.random(), random.random()) for i in range(5)}
    root = "N0"

    # Simple spanning tree (chain)
    edges = [(f"N{i}", f"N{i+1}") for i in range(4)]

    # Prior probabilities (uniform)
    priors = {e: 0.5 for e in edges}
    # Likelihoods (random)
    likelihoods = {e: random.random() for e in edges}
    false_positive = 0.05

    cost, allowed = hybrid_decision(
        nodes,
        edges,
        root,
        priors,
        likelihoods,
        false_positive,
    )

    print(f"Expected hybrid cost: {cost:.4f}")
    print(f"Circuit breaker allows operation: {allowed}")
    # Verify that the code runs without raising exceptions.