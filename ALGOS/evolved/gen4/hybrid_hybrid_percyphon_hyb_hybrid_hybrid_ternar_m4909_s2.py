# DARWIN HAMMER — match 4909, survivor 2
# gen: 4
# parent_a: hybrid_percyphon_hybrid_endpoint_circ_m45_s0.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_ternary_route_m98_s1.py (gen3)
# born: 2026-05-29T23:58:41Z

"""Hybrid module combining Percyphon procedural slot generation (Parent A) with
graph length‑matrix routing (Parent B). The mathematical bridge is a
morphology‑derived scaling factor

    w = sphericity_index(length, width, height) * flatness_index(length, width, height)

which is used to (i) adjust the ternary offset of generated ProceduralSlot
instances and (ii) weight the Euclidean edge lengths in a routing graph.
An additional righting‑time index provides a small perturbation to the
edge weights, creating a unified system where the physical shape of an
entity influences both its internal procedural attributes and its external
network routing behaviour.
"""

import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (from Parent A)
# ----------------------------------------------------------------------


def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][
        int(h[10:12], 16) % 6
    ]
    return name, alias, persona


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Simple physics‑inspired proxy for the time required for a serpentina‑type
    morphology to self‑right. The formula mixes mass, flatness and a few
    tunable constants.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (k * m.mass * neck_lever) / (b * fi + 1e-12)


# ----------------------------------------------------------------------
# Shared utilities (from Parent B)
# ----------------------------------------------------------------------


Point = Tuple[float, float]
Edge = Tuple[str, str]


def euclidean_length(a: Point, b: Point) -> float:
    """Straight‑line distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge]
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Edge]]:
    """
    Return a symmetric matrix L where L[i, j] = Euclidean length if (i, j) is an edge,
    otherwise 0. Also return the ordered list of edge index pairs matching the
    non‑zero entries of L (used for vectorised prior updates) and edge list.
    """
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)
    edge_idx: List[Tuple[int, int]] = []
    edge_list: List[Edge] = []

    for a, b in edges:
        i, j = idx_map[a], idx_map[b]
        length = euclidean_length(nodes[a], nodes[b])
        L[i, j] = L[j, i] = length
        edge_idx.append((i, j))
        edge_list.append((a, b))

    return L, edge_idx, edge_list


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge between the two parents
# ----------------------------------------------------------------------


def morphology_factor(m: Morphology) -> float:
    """Combined shape factor w = sphericity * flatness."""
    s = sphericity_index(m.length, m.width, m.height)
    f = flatness_index(m.length, m.width, m.height)
    return s * f


def generate_procedural_slots(
    seed: str, count: int, morph: Morphology
) -> List[ProceduralSlot]:
    """
    Create `count` ProceduralSlot objects. The ternary_offset of each slot is
    scaled by the morphology factor, linking physical shape to the procedural
    attribute space.
    """
    w = morphology_factor(morph)
    slots: List[ProceduralSlot] = []
    for i in range(count):
        name, alias, persona = _slot_name(seed, i)
        uuid = _uuid_from_sha256(f"{seed}:{i}")
        # ternary_offset is an integer in {0,1,2} multiplied by the weight
        ternary_offset = int(round(w * (i % 3)))
        slots.append(
            ProceduralSlot(
                slot_index=i,
                name=name,
                alias=alias,
                persona=persona,
                uuid=uuid,
                ternary_offset=ternary_offset,
            )
        )
    return slots


def build_weighted_length_matrix(
    nodes: Dict[str, Point], edges: List[Edge], morph: Morphology
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Edge]]:
    """
    Produce a length matrix where each edge length is multiplied by the
    morphology factor and a small perturbation derived from the righting‑time
    index. This fuses the shape analysis of Parent A with the routing graph of
    Parent B.
    """
    L, edge_idx, edge_list = build_length_matrix(nodes, edges)
    w = morphology_factor(morph)
    rt = righting_time_index(morph)
    # The righting‑time term is kept small (≤1 % of the weight) to act as a
    # gentle modifier rather than dominate the geometry.
    modifier = 1.0 + 0.01 * rt
    weighted_L = L * w * modifier
    return weighted_L, edge_idx, edge_list


def hybrid_shortest_path(
    start: str,
    end: str,
    nodes: Dict[str, Point],
    edges: List[Edge],
    morph: Morphology,
) -> Tuple[List[str], float]:
    """
    Compute a shortest path between `start` and `end` on the weighted graph.
    The edge weights incorporate the morphology factor, demonstrating a
    concrete hybrid operation.
    Returns a tuple (path, total_weight). If no path exists, the path list is
    empty and the weight is math.inf.
    """
    weighted_L, edge_idx, edge_list = build_weighted_length_matrix(nodes, edges, morph)

    # Build adjacency list from the weighted matrix
    idx_map = {name: i for i, name in enumerate(sorted(nodes))}
    adj: Dict[str, List[Tuple[str, float]]] = {name: [] for name in nodes}
    for (i, j), (a, b) in zip(edge_idx, edge_list):
        w = weighted_L[i, j]
        if w > 0:
            adj[a].append((b, w))
            adj[b].append((a, w))

    # Dijkstra's algorithm
    import heapq

    dist: Dict[str, float] = {name: math.inf for name in nodes}
    prev: Dict[str, str] = {}
    dist[start] = 0.0
    heap: List[Tuple[float, str]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    # Reconstruct path
    path: List[str] = []
    cur = end
    while cur in prev:
        path.append(cur)
        cur = prev[cur]
    if cur == start:
        path.append(start)
        path.reverse()
    else:
        path = []  # unreachable

    return path, dist[end]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a simple morphology
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=12.0)

    # Generate procedural slots
    slots = generate_procedural_slots(seed="demo", count=5, morph=morph)
    print("Procedural Slots:")
    for s in slots:
        print(s.as_dict())

    # Define a tiny graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]

    # Compute weighted matrix
    weighted_L, _, _ = build_weighted_length_matrix(nodes, edges, morph)
    print("\nWeighted Length Matrix:")
    print(weighted_L)

    # Find shortest path A -> D using hybrid weights
    path, total = hybrid_shortest_path("A", "D", nodes, edges, morph)
    print("\nHybrid shortest path from A to D:")
    print("Path:", path)
    print("Total weighted length:", total)