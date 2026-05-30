# DARWIN HAMMER — match 3164, survivor 1
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s4.py (gen2)
# parent_b: hybrid_minimum_cost_tree_bayes_update_m6_s0.py (gen1)
# born: 2026-05-29T23:48:12Z

"""Hybrid Fisher‑SSIM Bayesian Tree Routing
================================================

This module fuses two previously independent algorithms:

* **Parent A** – ``hybrid_fisher_localization_hybrid_ternary_route_m40_s4.py``  
  Provides a *Fisher information* score derived from a Gaussian beam and a
  structural‑similarity (SSIM) metric on textual signals. Their weighted sum
  ``hybrid_metric`` evaluates the quality of a packet given an angle, beam
  parameters and a reference text.

* **Parent B** – ``hybrid_minimum_cost_tree_bayes_update_m6_s0.py``  
  Builds a minimum‑cost spanning tree where each edge weight is re‑weighted by a
  Bayesian update that mixes a prior probability with a likelihood and a false‑
  positive rate.

**Mathematical Bridge**

For every *node* (packet) we treat the Fisher‑derived Gaussian intensity as a
*prior* probability `π_i = gaussian_beam(θ_i, μ, σ)`.  
For an *edge* `(i, j)` we use the *hybrid metric* as a *likelihood* `ℓ_{ij}`:

ℓ_{ij} = 0.5 * ( hybrid_metric(θ_i, μ, σ, text_i, ref) +
                hybrid_metric(θ_j, μ, σ, text_j, ref) )

A constant false‑positive probability `ϕ_{ij}` is assumed (default 0.1).  
The Bayesian marginal and posterior are:


m_{ij} = π_i * ℓ_{ij} + ϕ_{ij} * (1 - π_i)
w_{ij} = bayes_update(π_i, ℓ_{ij}, m_{ij})          # posterior


The *effective edge cost* combines the Euclidean distance `d_{ij}` between node
positions with the posterior weight:


c_{ij} = d_{ij} * (1 - w_{ij}) + λ * w_{ij}


where `λ` (``path_weight``) penalises highly probable edges to keep the tree
balanced.  The total tree cost is the sum of `c_{ij}` over the edges of a
minimum‑spanning tree (MST).  This creates a unified system where angular
localisation, textual similarity and Bayesian evidence jointly shape the routing
topology.

The public API offers three representative functions:

1. ``hybrid_metric`` – combines Fisher information and SSIM.
2. ``compute_edge_cost`` – builds the Bayesian‑adjusted edge cost.
3. ``hybrid_mst_cost`` – builds the MST using Prim's algorithm and returns its
   total cost.

A lightweight smoke test is provided under ``if __name__ == "__main__"``,
demonstrating a complete end‑to‑end run without external dependencies.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher / SSIM utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity at angle ``theta``."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index for two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def text_to_signal(text: str) -> List[float]:
    """Map a string to a numeric signal (ASCII codes)."""
    return [float(ord(ch)) for ch in text]


def hybrid_metric(theta: float, center: float, width: float,
                  packet_text: str, reference_text: str,
                  alpha: float = 0.5) -> float:
    """
    Weighted combination of Fisher information and SSIM.
    ``alpha`` controls the trade‑off (0 → pure SSIM, 1 → pure Fisher).
    """
    f = fisher_score(theta, center, width)
    s = ssim(text_to_signal(packet_text), text_to_signal(reference_text))
    return alpha * f + (1.0 - alpha) * s


# ----------------------------------------------------------------------
# Parent B – Bayesian minimum‑cost tree utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·π + FP·(1‑π)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability P(H|E) = π·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("marginal must be > 0")
    return prior * likelihood / marginal


# ----------------------------------------------------------------------
# Hybrid layer – edge cost that mixes geometry, Fisher‑SSIM and Bayesian update
# ----------------------------------------------------------------------
def compute_edge_cost(node_a: Dict[str, Any],
                      node_b: Dict[str, Any],
                      center: float,
                      width: float,
                      reference_text: str,
                      false_positive: float = 0.1,
                      alpha: float = 0.5,
                      path_weight: float = 0.2) -> float:
    """
    Produce the Bayesian‑adjusted cost for edge (a, b).

    Parameters
    ----------
    node_a, node_b
        Dictionaries containing at least ``'id'``, ``'pos'`` (Point),
        ``'angle'`` (float) and ``'text'`` (str).
    center, width
        Gaussian beam parameters used for priors and Fisher score.
    reference_text
        Global reference string for SSIM comparison.
    false_positive
        Fixed false‑positive probability for the edge.
    alpha
        Weight for the Fisher component inside ``hybrid_metric``.
    path_weight
        Scalar λ that penalises highly probable edges.

    Returns
    -------
    float
        Effective cost ``c_ij`` as described in the module docstring.
    """
    # Prior from Gaussian intensity of node a (could also use node b – symmetric)
    prior_a = gaussian_beam(node_a["angle"], center, width)

    # Likelihood as the average hybrid metric of the two endpoints
    h_a = hybrid_metric(node_a["angle"], center, width,
                        node_a["text"], reference_text, alpha)
    h_b = hybrid_metric(node_b["angle"], center, width,
                        node_b["text"], reference_text, alpha)
    likelihood = 0.5 * (h_a + h_b)

    # Bayesian marginal and posterior
    marginal = bayes_marginal(prior_a, likelihood, false_positive)
    posterior = bayes_update(prior_a, likelihood, marginal)

    # Geometric distance
    dist = length(node_a["pos"], node_b["pos"])

    # Blend distance with posterior weight
    cost = dist * (1.0 - posterior) + path_weight * posterior
    return cost


def prim_mst(nodes: Dict[str, Dict[str, Any]],
             edge_cost_func) -> Tuple[List[Edge], float]:
    """
    Prim's algorithm for a minimum‑spanning tree.

    Parameters
    ----------
    nodes
        Mapping from node identifier to node dictionary (must contain ``'pos'``).
    edge_cost_func
        Callable ``f(id_i, id_j) -> float`` returning the cost of edge (i, j).

    Returns
    -------
    (mst_edges, total_cost)
        List of edges forming the MST and the summed cost.
    """
    if not nodes:
        return [], 0.0

    visited = set()
    start = next(iter(nodes))
    visited.add(start)

    # Candidate edges: (cost, from, to)
    candidates = []
    for nid in nodes:
        if nid == start:
            continue
        c = edge_cost_func(start, nid)
        candidates.append((c, start, nid))

    mst_edges: List[Edge] = []
    total_cost = 0.0

    while len(visited) < len(nodes):
        # Select cheapest candidate that connects to an unvisited node
        candidates.sort(key=lambda x: x[0])
        while candidates:
            cost, frm, to = candidates.pop(0)
            if to not in visited:
                break
        else:
            raise RuntimeError("Graph is not fully connected")

        visited.add(to)
        mst_edges.append((frm, to))
        total_cost += cost

        # Add new frontier edges
        for nid in nodes:
            if nid in visited:
                continue
            c = edge_cost_func(to, nid)
            candidates.append((c, to, nid))

    return mst_edges, total_cost


def hybrid_mst_cost(nodes: Dict[str, Dict[str, Any]],
                    center: float,
                    width: float,
                    reference_text: str,
                    false_positive: float = 0.1,
                    alpha: float = 0.5,
                    path_weight: float = 0.2) -> Tuple[List[Edge], float]:
    """
    Build a minimum‑cost spanning tree whose edge weights are the Bayesian‑adjusted
    hybrid costs.

    Returns
    -------
    (edges, total_cost)
        The MST edge list and its total cost.
    """
    # Closure that captures the static parameters
    def edge_cost(i: str, j: str) -> float:
        return compute_edge_cost(
            nodes[i], nodes[j],
            center=center,
            width=width,
            reference_text=reference_text,
            false_positive=false_positive,
            alpha=alpha,
            path_weight=path_weight,
        )

    return prim_mst(nodes, edge_cost)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic scenario with four packets
    random.seed(42)

    # Global parameters
    CENTER = 0.0          # beam centre
    WIDTH = 1.0           # beam width
    REF_TEXT = "reference payload"

    # Construct nodes
    nodes = {
        "A": {"id": "A",
              "pos": (0.0, 0.0),
              "angle": -0.5,
              "text": "hello world"},
        "B": {"id": "B",
              "pos": (1.0, 0.2),
              "angle": 0.2,
              "text": "hallo welt"},
        "C": {"id": "C",
              "pos": (0.5, 1.5),
              "angle": 0.8,
              "text": "bonjour le monde"},
        "D": {"id": "D",
              "pos": (1.2, 1.0),
              "angle": -0.3,
              "text": "hola mundo"},
    }

    edges, total = hybrid_mst_cost(
        nodes,
        center=CENTER,
        width=WIDTH,
        reference_text=REF_TEXT,
        false_positive=0.1,
        alpha=0.6,
        path_weight=0.3,
    )

    print("MST edges:", edges)
    print("Total hybrid cost:", total)
    sys.exit(0)