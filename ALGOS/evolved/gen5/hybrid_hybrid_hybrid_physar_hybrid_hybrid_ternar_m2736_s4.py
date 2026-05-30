# DARWIN HAMMER — match 2736, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

"""Hybrid Physarum‑Bayesian Tree Router
=====================================

Parent A: ``hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py`` – 
Physarum‑inspired conductance dynamics where *flux* (q) drives updates of
*conductance* (G) and is interpreted as a reward signal for a contextual bandit.

Parent B: ``hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py`` – 
A ternary router that evaluates a spanning tree by its geometric material
cost and a Bayesian‑updated *edge prior* that captures uncertainty.

**Mathematical bridge**

Both parents treat edges as carriers of a scalar quantity:

* Physarum: `q = G / L * (p_a - p_b)` (flux on an edge)
* Bayesian router: `prior` – a probability attached to an edge

We fuse them by letting the *flux* act as an *evidence* (likelihood) for the
Bayesian update of the edge prior, while the *prior* modulates the effective
cost of the edge together with its conductance.  The hybrid expected edge
cost is therefore


C_edge = L / G * (1 - prior)


where a high conductance (low resistance) and a high prior (high confidence)
both reduce the cost.  The combined system updates conductances with the
Physarum rule and priors with Bayes’ theorem in a single iteration.

The module provides three core functions that demonstrate this hybrid
operation and a small smoke‑test under ``__main__``.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric helpers (from Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Physarum‑bandit primitives (from Parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Flux `q = G / L * (p_a - p_b)` on a single edge."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Physarum conductance dynamics:  G ← G + gain·q·dt − decay·G·dt."""
    return conductance + gain * q * dt - decay * conductance * dt


# ----------------------------------------------------------------------
# Bayesian primitives (from Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = likelihood·prior + false_positive·(1−prior)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior = prior·likelihood / P(E)."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_edge_cost(edge_len: float, conductance: float, prior: float) -> float:
    """
    Expected cost of a single edge merging Physarum resistance and Bayesian
    confidence:

        C = L / G * (1 - prior)

    A high conductance (low resistance) and a high prior (high confidence)
    both reduce the cost.
    """
    if conductance <= 0.0:
        # avoid division by zero – treat as very high resistance
        return float('inf')
    return edge_len / conductance * (1.0 - prior)


def hybrid_tree_cost(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     edge_attrs: Dict[Edge, Dict[str, float]],
                     path_weight: float = 0.2) -> float:
    """
    Total cost of the spanning tree:

        material_cost = Σ hybrid_edge_cost
        path_cost     = path_weight * Σ distance_from_root

    The distance from the root is computed on the undirected graph.
    """
    # material cost
    material = 0.0
    for a, b in edges:
        attr = edge_attrs[(a, b)]
        L = length(nodes[a], nodes[b])
        material += hybrid_edge_cost(L, attr["conductance"], attr["prior"])

    # distance from root (breadth‑first traversal)
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist: Dict[str, float] = {root: 0.0}
    stack: List[str] = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = path_weight * sum(dist.values())
    return material + path_cost


def hybrid_iteration(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     pressures: Dict[str, float],
                     edge_attrs: Dict[Edge, Dict[str, float]],
                     dt: float = 1.0,
                     gain: float = 1.0,
                     decay: float = 0.05,
                     false_positive: float = 0.01) -> Tuple[float, Dict[Edge, Dict[str, float]]]:
    """
    Perform one hybrid update step:

    1. Compute flux on each edge using current conductance and node pressures.
    2. Update conductance with the Physarum rule.
    3. Interpret the *normalized absolute flux* as a likelihood for the edge.
    4. Update the edge prior with Bayes' theorem.
    5. Return the new total tree cost and the updated edge attributes.
    """
    # 1‑2: flux & conductance update
    max_abs_flux = 0.0
    fluxes: Dict[Edge, float] = {}
    for a, b in edges:
        G = edge_attrs[(a, b)]["conductance"]
        L = length(nodes[a], nodes[b])
        q = flux(G, L, pressures[a], pressures[b])
        fluxes[(a, b)] = q
        max_abs_flux = max(max_abs_flux, abs(q))

    # 3‑4: Bayesian prior update (likelihood derived from flux)
    for a, b in edges:
        q = fluxes[(a, b)]
        # Normalized magnitude gives a value in [0,1]; add a tiny epsilon to avoid 0.
        likelihood = min(1.0, max(0.0, abs(q) / (max_abs_flux + 1e-12)))
        prior = edge_attrs[(a, b)]["prior"]
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal)
        edge_attrs[(a, b)]["prior"] = posterior

        # Conductance update (Physarum dynamics)
        G_new = update_conductance(edge_attrs[(a, b)]["conductance"], q, dt, gain, decay)
        edge_attrs[(a, b)]["conductance"] = max(G_new, 0.0)   # keep non‑negative

    # 5: evaluate cost
    total_cost = hybrid_tree_cost(nodes, edges, root, edge_attrs)
    return total_cost, edge_attrs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangle graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, math.sqrt(3) / 2),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    # Initialise pressures (randomly) and edge attributes
    random.seed(42)
    pressures = {n: random.uniform(-1.0, 1.0) for n in nodes}
    edge_attrs: Dict[Edge, Dict[str, float]] = {}
    for a, b in edges:
        edge_attrs[(a, b)] = {
            "conductance": random.uniform(0.5, 1.5),   # G_0
            "prior": random.uniform(0.2, 0.8),        # initial confidence
        }

    print("Initial pressures:", pressures)
    print("Initial edge attributes:", edge_attrs)

    # Run a few hybrid iterations
    for step in range(5):
        cost, edge_attrs = hybrid_iteration(
            nodes,
            edges,
            root,
            pressures,
            edge_attrs,
            dt=0.5,
            gain=0.8,
            decay=0.03,
            false_positive=0.02,
        )
        print(f"\nStep {step + 1}")
        print(f"  Total hybrid tree cost = {cost:.4f}")
        print(f"  Updated edge attributes = {edge_attrs}")

    print("\nSmoke test completed without errors.")