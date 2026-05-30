# DARWIN HAMMER — match 375, survivor 1
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py (gen2)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s3.py (gen2)
# born: 2026-05-29T23:28:30Z

"""Hybrid Fisher‑SSIM Bayesian Routing & Minimum‑Cost Tree
======================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A – hybrid_fisher_localization_hybrid_ternary_route_m40_s1.py**  
  Provides a Gaussian beam model, a Fisher‑information score derived from the
  beam, and a Structural‑Similarity‑Index‑Measure (SSIM) for comparing
  textual surfaces.

* **Parent B – hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s3.py**  
  Uses the same Gaussian/Fisher utilities to build Gaussian priors for the
  edges of a graph, updates those priors with temporal evidence via a
  Bayesian rule, and finally evaluates a Minimum‑Cost Spanning Tree (MST).

**Mathematical Bridge**

Both parents rely on the Gaussian beam `G(θ) = exp[-½((θ‑c)/w)²]`.  
The Fisher information for the mean of this Gaussian is  


I(θ) = (∂G/∂θ)² / G = ((θ‑c)² / w⁴) * G(θ)


In Parent B the Fisher score `I(θ)` is interpreted as a *precision* (the
inverse variance) of a Gaussian prior on a graph edge.  A Bayesian update
adds a new Gaussian likelihood derived from a fresh timestamp `θ'`, yielding an
updated precision `I_new = I_old + I(θ')`.  The corresponding variance
`σ² = 1 / I_new` is turned into an edge weight `w = σ`.

Parent A supplies an SSIM similarity `S(x, y)` between two text vectors.
We use this similarity as a *routing bias*: the effective edge weight used in
the MST is multiplied by `1 / (ε + S)`, i.e. more similar packets receive a
lower cost path.

The hybrid algorithm therefore:

1. Computes a Fisher precision for each edge from its current timestamp.
2. Updates the edge precision with the packet’s timestamp (Bayesian step).
3. Derives a variance‑based edge weight.
4. Modulates the weight by the SSIM similarity between the packet text and a
   reference text.
5. Runs a Prim‑style MST to obtain the minimum‑cost routing tree for the
   packet.

The three public functions below illustrate this pipeline."""
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared Gaussian / Fisher utilities (both parents)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ) with centre `center` and standard‑deviation `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    Algebraically equivalent to I = (∂G/∂θ)² / G.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Bayesian edge‑prior update (Parent B)
# ----------------------------------------------------------------------
def update_edge_precisions(
    edge_precisions: Dict[Tuple[str, str], float],
    timestamps: Dict[Tuple[str, str], float],
    center: float,
    width: float,
) -> None:
    """
    In‑place Bayesian update of edge precisions.

    For each edge (u, v) we have a prior precision `p`.  A new observation
    timestamp `t` yields a Fisher precision `I(t)`.  The posterior precision is

        p' = p + I(t)

    The function mutates ``edge_precisions``.
    """
    for edge, prior_prec in edge_precisions.items():
        theta = timestamps.get(edge, center)  # fallback to centre if missing
        fisher_prec = fisher_score(theta, center, width)
        edge_precisions[edge] = prior_prec + fisher_prec


# ----------------------------------------------------------------------
# Minimum‑Cost Spanning Tree (Prim) with SSIM bias (Hybrid core)
# ----------------------------------------------------------------------
def prim_mst(
    nodes: Iterable[str],
    base_weights: Dict[Tuple[str, str], float],
    similarity_bias: float,
) -> Tuple[float, List[Tuple[str, str]]]:
    """
    Compute a Minimum‑Cost Spanning Tree using Prim's algorithm.

    Edge cost = base_weight / (ε + similarity_bias)
    where ``similarity_bias`` is a scalar in [0, 1] derived from SSIM.
    Returns total cost and list of edges in the MST.
    """
    eps = 1e-12
    nodes = list(nodes)
    if not nodes:
        return 0.0, []

    visited = {nodes[0]}
    edges_mst: List[Tuple[str, str]] = []
    total_cost = 0.0

    while len(visited) < len(nodes):
        min_edge = None
        min_cost = math.inf
        for (u, v), w in base_weights.items():
            if (u in visited) ^ (v in visited):  # exactly one endpoint visited
                cost = w / (eps + similarity_bias)
                if cost < min_cost:
                    min_cost = cost
                    min_edge = (u, v)
        if min_edge is None:
            raise RuntimeError("Graph is not connected")
        edges_mst.append(min_edge)
        total_cost += min_cost
        visited.update(min_edge)

    return total_cost, edges_mst


# ----------------------------------------------------------------------
# High‑level hybrid routing function (demonstrates the fusion)
# ----------------------------------------------------------------------
def hybrid_route(
    packet: dict,
    reference_text: str,
    graph_edges: List[Tuple[str, str]],
    edge_timestamps: Dict[Tuple[str, str], float],
    center: float = 0.0,
    width: float = 1.0,
) -> dict:
    """
    Route a packet through a graph using a Fisher‑Bayesian updated MST
    weighted by SSIM similarity.

    Steps
    -----
    1. Extract textual surface from ``packet`` and build numeric vectors.
    2. Compute SSIM similarity `S` with ``reference_text``.
    3. Initialise edge precisions with a small prior (1e-3) and update them
       with the packet's timestamp (if present) via ``update_edge_precisions``.
    4. Convert precisions to variances → base weights `w = 1 / sqrt(precision)`.
    5. Run ``prim_mst`` with the SSIM bias to obtain a routing tree.
    6. Return a dict containing the similarity, updated precisions and the MST.
    """
    # 1. Textual vectors (simple ordinal encoding)
    pkt_text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    # Pad to equal length
    max_len = max(len(pkt_text), len(reference_text))
    pkt_vec = np.array([ord(c) for c in pkt_text.ljust(max_len)], dtype=float)
    ref_vec = np.array([ord(c) for c in reference_text.ljust(max_len)], dtype=float)

    # 2. SSIM similarity
    similarity = ssim(pkt_vec, ref_vec)

    # 3. Initialise precisions
    edge_precisions: Dict[Tuple[str, str], float] = {
        tuple(sorted(e)): 1e-3 for e in graph_edges
    }

    # Use packet timestamp if available; otherwise fall back to centre.
    pkt_timestamp = float(packet.get("timestamp", center))

    # Build a temporary timestamps dict that injects the packet timestamp
    # on all incident edges (simulating that the packet traverses each edge).
    temp_timestamps = {
        edge: pkt_timestamp for edge in edge_precisions.keys()
    }
    # Merge with any externally supplied timestamps (they have higher priority)
    temp_timestamps.update(edge_timestamps)

    # 4. Bayesian update of precisions
    update_edge_precisions(edge_precisions, temp_timestamps, center, width)

    # 5. Convert precision -> variance -> weight
    base_weights: Dict[Tuple[str, str], float] = {}
    for edge, prec in edge_precisions.items():
        # Guard against zero precision
        if prec <= 0:
            variance = 1e6
        else:
            variance = 1.0 / prec
        base_weights[edge] = math.sqrt(variance)  # weight proportional to σ

    # 6. MST with SSIM bias
    nodes = {n for e in graph_edges for n in e}
    total_cost, mst_edges = prim_mst(nodes, base_weights, similarity)

    return {
        "similarity": similarity,
        "edge_precisions": edge_precisions,
        "mst_total_cost": total_cost,
        "mst_edges": mst_edges,
    }


# ----------------------------------------------------------------------
# Additional helper demonstrating pure Bayesian edge update (Parent B)
# ----------------------------------------------------------------------
def bayesian_edge_update_demo():
    """
    Small demonstration of edge‑precision updates without routing.
    Shows how timestamps modify Gaussian priors.
    """
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    timestamps = {("A", "B"): 0.2, ("B", "C"): -0.1, ("C", "A"): 0.5}
    precisions = {tuple(sorted(e)): 0.01 for e in edges}
    update_edge_precisions(precisions, timestamps, center=0.0, width=1.0)
    return precisions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    graph = [("node1", "node2"), ("node2", "node3"), ("node3", "node1")]
    # External timestamps for some edges (simulating historic measurements)
    external_ts = {("node1", "node2"): -0.3, ("node2", "node3"): 0.1}

    sample_packet = {
        "text_surface": "Hello world!",
        "timestamp": 0.05,
        "source": "sensor_A",
    }

    result = hybrid_route(
        packet=sample_packet,
        reference_text="Hello there!",
        graph_edges=graph,
        edge_timestamps=external_ts,
        center=0.0,
        width=1.0,
    )

    print("SSIM similarity :", result["similarity"])
    print("Updated edge precisions :", result["edge_precisions"])
    print("MST total cost :", result["mst_total_cost"])
    print("MST edges :", result["mst_edges"])

    # Run the pure Bayesian demo
    print("Pure Bayesian edge precisions :", bayesian_edge_update_demo())