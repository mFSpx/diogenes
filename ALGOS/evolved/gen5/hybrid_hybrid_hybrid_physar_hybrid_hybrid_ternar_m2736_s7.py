# DARWIN HAMMER — match 2736, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_bandit_m204_s0.py (gen4)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_ternar_m439_s0.py (gen4)
# born: 2026-05-29T23:44:00Z

import math
import random
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
UEdge = FrozenSet[str]  # unordered edge for dictionary keys

# ----------------------------------------------------------------------
# Geometry helpers (Parent B)
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Physarum‑bandit primitives (Parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Flux `q = G / L * (p_a - p_b)` on a single edge."""
    if edge_length <= 0.0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float,
                       dt: float = 1.0,
                       gain: float = 1.0,
                       decay: float = 0.05,
                       prior: float = 0.5) -> float:
    """
    Physarum conductance dynamics enriched with Bayesian confidence.

    The gain is modulated by `prior` so that edges we are confident about
    adapt faster, while uncertain edges evolve more cautiously.
    """
    effective_gain = gain * (0.5 + 0.5 * prior)   # linear interpolation [0.5*gain, gain]
    return conductance + effective_gain * q * dt - decay * conductance * dt

# ----------------------------------------------------------------------
# Bayesian primitives (Parent B)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float,
                   false_positive: float,
                   true_negative: float = 0.01) -> float:
    """
    Compute the marginal probability P(E).

    The model assumes two error channels:
        - false positive  : edge is good but evidence suggests otherwise
        - true negative   : edge is bad and evidence also suggests bad
    """
    if not (0.0 <= prior <= 1.0):
        raise ValueError("prior must be in [0,1]")
    if not (0.0 <= likelihood <= 1.0):
        raise ValueError("likelihood must be in [0,1]")
    if not (0.0 <= false_positive <= 1.0):
        raise ValueError("false_positive must be in [0,1]")
    if not (0.0 <= true_negative <= 1.0):
        raise ValueError("true_negative must be in [0,1]")

    # P(E) = likelihood·prior + false_positive·(1−prior)
    # plus a small contribution from true_negative to keep denominator >0
    return likelihood * prior + false_positive * (1.0 - prior) + 1e-12 * true_negative

def bayes_update(prior: float, likelihood: float,
                 false_positive: float,
                 true_negative: float = 0.01) -> float:
    """
    Perform a Bayesian update using the evidence derived from flux.
    """
    marginal = bayes_marginal(prior, likelihood, false_positive, true_negative)
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def _make_uedge(a: str, b: str) -> UEdge:
    """Canonical unordered edge key."""
    return frozenset((a, b))

def hybrid_edge_cost(edge_len: float,
                     conductance: float,
                     prior: float) -> float:
    """
    Expected cost of a single edge merging Physarum resistance and Bayesian
    confidence.

        C = (L / G) * (1 - prior)

    The term (1‑prior) penalises edges we are uncertain about.
    """
    if conductance <= 0.0:
        return float('inf')
    return edge_len / conductance * (1.0 - prior)

def hybrid_tree_cost(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     edge_attrs: Dict[UEdge, Dict[str, float]],
                     path_weight: float = 0.2) -> float:
    """
    Total cost of a spanning tree.

    material_cost = Σ hybrid_edge_cost
    path_cost     = path_weight * Σ distance_from_root
    """
    # material cost
    material = 0.0
    for a, b in edges:
        uedge = _make_uedge(a, b)
        attr = edge_attrs[uedge]
        L = length(nodes[a], nodes[b])
        material += hybrid_edge_cost(L, attr["conductance"], attr["prior"])

    # BFS to compute distances from root
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist: Dict[str, float] = {root: 0.0}
    q: deque[str] = deque([root])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                q.append(nxt)

    path_cost = path_weight * sum(dist.values())
    return material + path_cost

def hybrid_iteration(nodes: Dict[str, Point],
                     edges: List[Edge],
                     root: str,
                     pressures: Dict[str, float],
                     edge_attrs: Dict[UEdge, Dict[str, float]],
                     dt: float = 1.0,
                     gain: float = 1.0,
                     decay: float = 0.05,
                     false_positive: float = 0.01,
                     true_negative: float = 0.01) -> Tuple[float, Dict[UEdge, Dict[str, float]]]:
    """
    Perform one hybrid update step:

    1. Compute flux on each edge using current conductance and node pressures.
    2. Update conductance with the Physarum rule, modulated by the current prior.
    3. Convert the *normalized absolute flux* into a likelihood.
    4. Update the edge prior with Bayes' theorem.
    5. Return the new total tree cost and a **new** edge‑attribute mapping.
    """
    # ------------------------------------------------------------------
    # 1. Flux computation
    # ------------------------------------------------------------------
    fluxes: Dict[UEdge, float] = {}
    max_abs_flux = 0.0
    for a, b in edges:
        uedge = _make_uedge(a, b)
        G = edge_attrs[uedge]["conductance"]
        L = length(nodes[a], nodes[b])
        q_val = flux(G, L, pressures[a], pressures[b])
        fluxes[uedge] = q_val
        max_abs_flux = max(max_abs_flux, abs(q_val))

    # ------------------------------------------------------------------
    # 2‑4. Bayesian‑Physarum joint update
    # ------------------------------------------------------------------
    new_attrs: Dict[UEdge, Dict[str, float]] = {}
    for a, b in edges:
        uedge = _make_uedge(a, b)
        q_val = fluxes[uedge]
        prior = edge_attrs[uedge]["prior"]
        # Likelihood from normalized flux magnitude; epsilon avoids division by zero.
        norm = max_abs_flux if max_abs_flux > 0 else 1.0
        likelihood = min(1.0, max(0.0, abs(q_val) / (norm + 1e-12)))

        # Bayesian update
        posterior = bayes_update(prior, likelihood, false_positive, true_negative)

        # Conductance update, now aware of posterior confidence
        G_new = update_conductance(edge_attrs[uedge]["conductance"],
                                   q_val,
                                   dt=dt,
                                   gain=gain,
                                   decay=decay,
                                   prior=posterior)

        new_attrs[uedge] = {
            "conductance": max(G_new, 0.0),   # enforce non‑negative conductance
            "prior": posterior
        }

    # ------------------------------------------------------------------
    # 5. Cost evaluation
    # ------------------------------------------------------------------
    total_cost = hybrid_tree_cost(nodes, edges, root, new_attrs, path_weight=0.2)
    return total_cost, new_attrs

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

    # Initialise pressures randomly
    random.seed(42)
    pressures = {n: random.uniform(-1.0, 1.0) for n in nodes}

    # Initialise edge attributes (conductance >0, prior in [0,1])
    edge_attrs: Dict[UEdge, Dict[str, float]] = {}
    for a, b in edges:
        uedge = _make_uedge(a, b)
        edge_attrs[uedge] = {
            "conductance": random.uniform(0.1, 1.0),
            "prior": random.uniform(0.3, 0.7)
        }

    # Run a few hybrid iterations and print progress
    for it in range(1, 11):
        cost, edge_attrs = hybrid_iteration(
            nodes,
            edges,
            root,
            pressures,
            edge_attrs,
            dt=0.5,
            gain=0.8,
            decay=0.02,
            false_positive=0.02,
            true_negative=0.01
        )
        print(f"Iter {it:02d} | Total cost: {cost:.4f}")
        for a, b in edges:
            uedge = _make_uedge(a, b)
            attr = edge_attrs[uedge]
            print(f"  Edge {a}-{b}: G={attr['conductance']:.3f}, prior={attr['prior']:.3f}")
        print("-" * 40)