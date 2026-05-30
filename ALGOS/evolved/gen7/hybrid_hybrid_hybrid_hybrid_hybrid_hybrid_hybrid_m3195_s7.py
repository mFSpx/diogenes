# DARWIN HAMMER — match 3195, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s2.py (gen6)
# born: 2026-05-29T23:48:36Z

"""Hybrid algorithm merging DARWIN HAMMER privacy risk curvature (Parent A) with
Pheromone‑based geometric algebra dynamics (Parent B).

Mathematical bridge:
- The reconstruction risk score *r_i* from Parent A becomes the scalar (grade‑0)
  component of a 3‑D geometric‑algebra multivector.
- Ollivier‑Ricci curvature *κ_i* computed on the same graph supplies the
  e1‑component.
- Aggregated pheromone signal values attached to a node supply the e2‑component,
  while the pheromone decay factor (derived from half‑life and curvature) feeds
  the e3‑component.
Thus a single multivector encodes privacy, geometric, and adaptive‑signal
information, enabling downstream operations (e.g. VRAM scheduling) to act on
the fused representation."""
import sys
import math
import random
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


@dataclass
class Node:
    """Graph node carrying RAM capacity and privacy risk."""
    id: int
    ram_mb: int
    risk_score: float = 0.0  # filled later


# ----------------------------------------------------------------------
# Parent‑A utilities (privacy & curvature)
# ----------------------------------------------------------------------


def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Risk score r = clamp(uqi / total_records, 0, 1)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def build_adjacency(nodes: List[Node],
                    ram_threshold: float = 0.2) -> Dict[int, List[int]]:
    """
    Simple undirected adjacency based on relative RAM similarity.
    Two nodes are neighbours if |ram_i - ram_j| / max(ram_i, ram_j) <= threshold.
    """
    adj: Dict[int, List[int]] = {n.id: [] for n in nodes}
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i >= j:
                continue
            if max(ni.ram_mb, nj.ram_mb) == 0:
                continue
            rel_diff = abs(ni.ram_mb - nj.ram_mb) / max(ni.ram_mb, nj.ram_mb)
            if rel_diff <= ram_threshold:
                adj[ni.id].append(nj.id)
                adj[nj.id].append(ni.id)
    return adj


def compute_ollivier_ricci_curvature(nodes: List[Node],
                                    adj: Dict[int, List[int]],
                                    alpha: float = 0.5) -> Dict[int, float]:
    """
    Hybrid curvature κ(i) = 1 - Σ_j μ_i(j) / deg(i)
    where μ_i(j) = α·r_i·δ_{i=j} + (1-α)·r_i·(1/deg(i))·Σ_{u∈N(i)}δ_{u=j}
    """
    curvature: Dict[int, float] = {}
    for node in nodes:
        nid = node.id
        neighbors = adj.get(nid, [])
        deg = max(len(neighbors), 1)  # avoid division by zero
        r = node.risk_score
        # transport mass to self
        mu_self = alpha * r
        # transport mass to each neighbor
        mu_neighbor = (1.0 - alpha) * r / deg
        total_mass = mu_self + deg * mu_neighbor
        # Ollivier‑Ricci formula (simplified)
        curvature[nid] = 1.0 - (total_mass / deg)
    return curvature


# ----------------------------------------------------------------------
# Parent‑B utilities (pheromones & geometric algebra)
# ----------------------------------------------------------------------


class PheromoneEntry:
    """Container for a pheromone signal with exponential decay."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = datetime.utcnow()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.utcnow()


class Multivector3D:
    """
    3‑D geometric algebra multivector with eight components:
    [s, e1, e2, e3, e12, e13, e23, e123]
    """
    __slots__ = ("components",)

    def __init__(self,
                 s: float = 0.0,
                 e1: float = 0.0,
                 e2: float = 0.0,
                 e3: float = 0.0,
                 e12: float = 0.0,
                 e13: float = 0.0,
                 e23: float = 0.0,
                 e123: float = 0.0):
        self.components = np.array([s, e1, e2, e3, e12, e13, e23, e123],
                                   dtype=float)

    def __add__(self, other: "Multivector3D") -> "Multivector3D":
        out = Multivector3D()
        out.components = self.components + other.components
        return out

    def __sub__(self, other: "Multivector3D") -> "Multivector3D":
        out = Multivector3D()
        out.components = self.components - other.components
        return out

    def __mul__(self, scalar: float) -> "Multivector3D":
        out = Multivector3D()
        out.components = self.components * scalar
        return out

    __rmul__ = __mul__

    def geometric_product(self, other: "Multivector3D") -> "Multivector3D":
        """
        Very lightweight implementation: only scalar‑scalar and scalar‑vector
        interactions are needed for the hybrid use‑case.
        """
        a = self.components
        b = other.components
        # scalar part (grade‑0)
        s = a[0] * b[0]
        # vector part (grade‑1) – treat e1/e2/e3 as independent axes
        e1 = a[0] * b[1] + a[1] * b[0]
        e2 = a[0] * b[2] + a[2] * b[0]
        e3 = a[0] * b[3] + a[3] * b[0]
        out = Multivector3D(s, e1, e2, e3)
        return out

    def __repr__(self) -> str:
        labels = ["s", "e1", "e2", "e3", "e12", "e13", "e23", "e123"]
        parts = [f"{lbl}:{val:.3g}" for lbl, val in zip(labels, self.components) if abs(val) > 1e-12]
        return f"Multivector3D({', '.join(parts)})"


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def aggregate_pheromones_by_node(nodes: List[Node],
                                pheromones: Iterable[PheromoneEntry]) -> Dict[int, List[PheromoneEntry]]:
    """
    Group pheromone entries by the integer part of their ``surface_key``.
    The convention is that ``surface_key`` encodes the node id as a string.
    """
    mapping: Dict[int, List[PheromoneEntry]] = {n.id: [] for n in nodes}
    for p in pheromones:
        try:
            nid = int(p.surface_key)
        except ValueError:
            continue
        if nid in mapping:
            mapping[nid].append(p)
    return mapping


def hybrid_multivector_representation(nodes: List[Node],
                                       adj: Dict[int, List[int]],
                                       pheromones: Iterable[PheromoneEntry],
                                       alpha: float = 0.5) -> Dict[int, Multivector3D]:
    """
    Produce a multivector per node:
        s   = risk_score (privacy)
        e1  = Ollivier‑Ricci curvature (geometric)
        e2  = sum of current pheromone signal values (adaptive)
        e3  = average decay factor of those pheromones (temporal)
    """
    # 1. Compute curvature using risk‑aware transport
    curvature = compute_ollivier_ricci_curvature(nodes, adj, alpha)

    # 2. Group pheromones
    ph_by_node = aggregate_pheromones_by_node(nodes, pheromones)

    out: Dict[int, Multivector3D] = {}
    for node in nodes:
        nid = node.id
        curv = curvature.get(nid, 0.0)
        ph_list = ph_by_node.get(nid, [])
        # apply decay before aggregation
        total_signal = 0.0
        total_decay = 0.0
        for ph in ph_list:
            ph.apply_decay()
            total_signal += ph.signal_value
            total_decay += ph.decay_factor()
        avg_decay = (total_decay / len(ph_list)) if ph_list else 0.0

        mv = Multivector3D(s=node.risk_score,
                           e1=curv,
                           e2=total_signal,
                           e3=avg_decay)
        out[nid] = mv
    return out


def update_pheromones_from_curvature(pheromones: Iterable[PheromoneEntry],
                                    curvature_map: Dict[int, float],
                                    influence: float = 0.1) -> None:
    """
    Modulate each pheromone's ``signal_value`` by a factor that depends on the
    curvature of its associated node:
        new_value = old_value * (1 + influence * κ_i)
    """
    for ph in pheromones:
        try:
            nid = int(ph.surface_key)
        except ValueError:
            continue
        κ = curvature_map.get(nid, 0.0)
        ph.signal_value *= (1.0 + influence * κ)


def privacy_aware_vram_schedule(nodes: List[Node],
                                curvature: Dict[int, float]) -> List[Tuple[int, int]]:
    """
    Return a list of (node_id, allocated_ram_mb) sorted by a priority score:
        priority = risk_score * (1 + κ)
    Higher priority gets larger RAM chunks (simple greedy allocation).
    """
    # compute priority
    priority = []
    for n in nodes:
        κ = curvature.get(n.id, 0.0)
        score = n.risk_score * (1.0 + κ)
        priority.append((score, n))

    # sort descending by score
    priority.sort(key=lambda x: x[0], reverse=True)

    # allocate RAM proportionally (total RAM = sum of all node RAM)
    total_ram = sum(n.ram_mb for n in nodes)
    allocations: List[Tuple[int, int]] = []
    for _, node in priority:
        # simple heuristic: allocate node.ram_mb * (score / max_score)
        max_score = priority[0][0] if priority else 1.0
        alloc = int(node.ram_mb * (node.risk_score * (1.0 + curvature.get(node.id, 0.0)) / max_score))
        allocations.append((node.id, max(1, alloc)))  # at least 1 MB
    return allocations


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # 1️⃣ Create synthetic nodes
    random.seed(42)
    nodes = []
    for i in range(5):
        ram = random.choice([1024, 2048, 4096])
        # fake privacy data
        uq = random.randint(0, 100)
        tot = 200
        risk = reconstruction_risk_score(uq, tot)
        nodes.append(Node(id=i, ram_mb=ram, risk_score=risk))

    # 2️⃣ Build adjacency based on RAM similarity
    adjacency = build_adjacency(nodes, ram_threshold=0.25)

    # 3️⃣ Create some pheromones attached to nodes (surface_key = node id)
    pheromones = [
        PheromoneEntry(surface_key=str(n.id), signal_kind="privacy",
                       signal_value=random.uniform(0.1, 1.0), half_life_seconds=30)
        for n in nodes for _ in range(2)  # two per node
    ]

    # 4️⃣ Hybrid multivector construction
    mv_map = hybrid_multivector_representation(nodes, adjacency, pheromones, alpha=0.6)

    # 5️⃣ Show results
    print("=== Multivector per node ===")
    for nid, mv in mv_map.items():
        print(f"Node {nid}: {mv}")

    # 6️⃣ Update pheromones using curvature
    curvature_vals = compute_ollivier_ricci_curvature(nodes, adjacency, alpha=0.6)
    update_pheromones_from_curvature(pheromones, curvature_vals, influence=0.2)

    # 7️⃣ VRAM schedule
    schedule = privacy_aware_vram_schedule(nodes, curvature_vals)
    print("\n=== VRAM allocation (node_id, allocated_mb) ===")
    for nid, alloc in schedule:
        print(f"{nid}: {alloc} MB")