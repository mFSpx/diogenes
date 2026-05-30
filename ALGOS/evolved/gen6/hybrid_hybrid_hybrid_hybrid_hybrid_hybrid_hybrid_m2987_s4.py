# DARWIN HAMMER — match 2987, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1182_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_ternary_route_m812_s1.py (gen4)
# born: 2026-05-29T23:47:13Z

"""Hybrid algorithm combining Physarum‑style conductance dynamics with
deterministic Bayesian feature extraction and ternary minimum‑cost routing.

Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_m1182_s0.py (Physarum flux + bandit
  propensity update)
- hybrid_hybrid_hybrid_bayes__hybrid_ternary_route_m812_s1.py (deterministic
  feature vector extraction + Euclidean‑weighted MST routing)

Mathematical bridge:
Both parents rely on Euclidean distance.  In the hybrid we treat each object
*i* as a composite state **sᵢ = (pᵢ, vᵢ)** where *pᵢ*∈ℝ² is a spatial point
and *vᵢ*∈ℝⁿ is a deterministic feature (master) vector.  The edge cost used for
routing is a weighted sum of spatial and feature distances

 c_{ij}=α‖pᵢ−pⱼ‖₂+β‖vᵢ−vⱼ‖₂ .

After constructing a minimum‑cost spanning tree on these costs we run a
Physarum‑style flux computation on the tree edges.  Conductance of an edge
*e* is updated by merging the bandit‑propensity term *q = propensity·reward*
with the absolute flux |Φ| :

 g_{new}=max(0, g+Δt·(γ·(|q|+|Φ|)−δ·g))

where γ is a gain factor and δ a decay factor.  This fuses the two core
topologies into a single unified system."""


import hashlib
import json
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Deterministic feature extraction (parent B)
# ----------------------------------------------------------------------
def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit integer hash for *text* using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_master_vector(text: str, dim: int = 8) -> np.ndarray:
    """
    Produce a reproducible pseudo‑random feature vector of length *dim*.
    The vector is normalised to unit length.
    """
    seed = _deterministic_hash(text)
    rng = np.random.RandomState(seed % (2**32))
    vec = rng.randn(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------
# Physarum / bandit primitives (parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_bandit_update(conductance: float, propensity: float, reward: float,
                         dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)


# ----------------------------------------------------------------------
# Composite data structures
# ----------------------------------------------------------------------
Point = Tuple[float, float]


@dataclass
class Node:
    node_id: str
    point: Point                     # spatial coordinate pᵢ ∈ ℝ²
    feature: np.ndarray              # master vector vᵢ ∈ ℝⁿ
    pressure: float = 0.0            # scalar pressure used for flux


@dataclass
class Edge:
    node_a: Node
    node_b: Node
    length: float                    # Euclidean distance between points
    conductance: float = 1.0
    propensity: float = 0.5          # bandit propensity associated with this edge
    reward: float = 0.0              # most recent reward signal


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_hybrid_edge_cost(node_i: Node, node_j: Node,
                            alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Weighted sum of spatial Euclidean distance and feature‑space distance.
    c_{ij}=α‖pᵢ−pⱼ‖₂+β‖vᵢ−vⱼ‖₂ .
    """
    spatial = math.hypot(node_i.point[0] - node_j.point[0],
                        node_i.point[1] - node_j.point[1])
    feature = np.linalg.norm(node_i.feature - node_j.feature)
    return alpha * spatial + beta * feature


def build_hybrid_mst(nodes: List[Node],
                    alpha: float = 1.0,
                    beta: float = 1.0) -> List[Edge]:
    """
    Construct a Minimum Spanning Tree on the complete graph defined by the
    hybrid edge cost.  Kruskal's algorithm with a simple union‑find structure
    is used (O(E log E) where E = n(n‑1)/2).
    Returns the list of edges belonging to the MST.
    """
    # Helper union‑find
    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> bool:
        xr, yr = find(x), find(y)
        if xr == yr:
            return False
        if rank[xr] < rank[yr]:
            parent[xr] = yr
        elif rank[xr] > rank[yr]:
            parent[yr] = xr
        else:
            parent[yr] = xr
            rank[xr] += 1
        return True

    # Initialise union‑find
    for node in nodes:
        parent[node.node_id] = node.node_id
        rank[node.node_id] = 0

    # Generate all possible edges with their hybrid cost
    edge_candidates: List[Tuple[float, Edge]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n_i, n_j = nodes[i], nodes[j]
            length = math.hypot(n_i.point[0] - n_j.point[0],
                                n_i.point[1] - n_j.point[1])
            e = Edge(node_a=n_i, node_b=n_j, length=length)
            cost = compute_hybrid_edge_cost(n_i, n_j, alpha, beta)
            edge_candidates.append((cost, e))

    # Sort by cost and pick edges that connect new components
    edge_candidates.sort(key=lambda x: x[0])
    mst: List[Edge] = []
    for cost, edge in edge_candidates:
        if union(edge.node_a.node_id, edge.node_b.node_id):
            mst.append(edge)
        if len(mst) == len(nodes) - 1:
            break
    return mst


def hybrid_conductance_update(edge: Edge,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05,
                              gamma: float = 1.0) -> None:
    """
    Update the conductance of *edge* by merging the bandit‑propensity term
    and the absolute Physarum flux.

    g_{new}=max(0, g+Δt·(γ·(|q|+|Φ|)−δ·g))

    where q = propensity·reward and Φ = flux(conductance, length, pressure_a,
    pressure_b).
    """
    q = edge.propensity * edge.reward
    phi = flux(edge.conductance, edge.length, edge.node_a.pressure, edge.node_b.pressure)
    combined = gamma * (abs(q) + abs(phi))
    edge.conductance = max(0.0, edge.conductance + dt * (gain * combined - decay * edge.conductance))


def hybrid_route_and_update(nodes: List[Node],
                            alpha: float = 1.0,
                            beta: float = 1.0,
                            dt: float = 1.0,
                            gain: float = 1.0,
                            decay: float = 0.05,
                            gamma: float = 1.0) -> List[Edge]:
    """
    End‑to‑end pipeline:
    1. Build a hybrid MST using spatial + feature costs.
    2. Assign random pressures to nodes (simulating a source/sink configuration).
    3. For each MST edge, draw a random reward and propensity, then update its
       conductance with the fused Physarum‑bandit rule.
    Returns the list of updated edges.
    """
    mst_edges = build_hybrid_mst(nodes, alpha, beta)

    # Random pressure assignment (one source at max pressure, one sink at min)
    max_pressure = random.uniform(5.0, 10.0)
    min_pressure = random.uniform(0.0, 2.0)
    source_node = random.choice(nodes)
    sink_node = random.choice([n for n in nodes if n != source_node])
    for n in nodes:
        n.pressure = max_pressure if n is source_node else (min_pressure if n is sink_node else random.uniform(min_pressure, max_pressure))

    # Update each edge
    for e in mst_edges:
        e.propensity = random.random()          # ∈[0,1]
        e.reward = random.uniform(-1.0, 1.0)     # could be negative
        hybrid_conductance_update(e, dt, gain, decay, gamma)

    return mst_edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic dataset
    texts = [
        "alpha", "bravo", "charlie", "delta", "echo"
    ]
    nodes: List[Node] = []
    for idx, txt in enumerate(texts):
        pt: Point = (random.uniform(0, 100), random.uniform(0, 100))
        fv = extract_master_vector(txt, dim=8)
        nodes.append(Node(node_id=f"n{idx}", point=pt, feature=fv))

    # Run the hybrid pipeline
    updated_edges = hybrid_route_and_update(
        nodes,
        alpha=0.7,
        beta=0.3,
        dt=0.5,
        gain=1.2,
        decay=0.04,
        gamma=0.9
    )

    # Simple verification output (no external libraries)
    print("MST edges after hybrid update:")
    for e in updated_edges:
        print(f"{e.node_a.node_id} – {e.node_b.node_id} | conductance={e.conductance:.4f} | reward={e.reward:.3f} | propensity={e.propensity:.3f}")
    sys.exit(0)