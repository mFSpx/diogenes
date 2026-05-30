# DARWIN HAMMER — match 5667, survivor 4
# gen: 7
# parent_a: hybrid_possum_filter_hybrid_semantic_neig_m209_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m1724_s0.py (gen6)
# born: 2026-05-30T00:04:04Z

"""Hybrid Morphology‑Spatial‑Physarum Network
Parents:
- hybrid_possum_filter_hybrid_semantic_neig_m209_s1.py (spatial diversity + recovery priority)
- hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m1724_s0.py (Morphology → pressure, Shapley‑weighted conductance, Physarum dynamics)

Mathematical bridge:
1. Each Entity provides a spatial score `h(i,j)` (Parent A) that quantifies
   diversity between two nodes.
2. The same pair obtains a conductance `g_ij` derived from Shapley‑weighted
   morphology importance (Parent B).  
   The final edge conductance is the product  

        G_ij = h(i,j) · g_ij

   thus coupling spatial‑semantic similarity with game‑theoretic morphology
   attribution.
3. Pressures are the sphericity index of each node (Parent B).  Physarum‑type
   flow updates use the hybrid conductances `G_ij`.  Edge failures are
   monitored by a circuit‑breaker that opens when conductance repeatedly falls
   below a tolerance.
"""

from __future__ import annotations
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Set, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Entity:
    """Physical or logical entity used by the hybrid algorithm."""
    id: str
    lat: float
    lon: float
    category: str
    morphology: Morphology
    recovery_priority: float = 0.0   # from Parent A semantic‑morphology system
    address_signature: str = ""      # unused but kept for compatibility

# ----------------------------------------------------------------------
# Parent A utilities (spatial diversity & morphology indices)
# ----------------------------------------------------------------------
def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def sphericity_index(length: float, width: float, height: float) -> float:
    """Scalar pressure proxy used by the Physarum dynamics."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical robustness metric (unused in the final hybrid but retained)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

# ----------------------------------------------------------------------
# Parent B utilities (hypervector encoding, Shapley weighting, circuit breaker)
# ----------------------------------------------------------------------
def morphology_to_hypervector(m: Morphology, dim: int = 10_000) -> np.ndarray:
    """Bind morphology dimensions into a binary hypervector (+1 / -1)."""
    rng = np.random.default_rng(seed=int(m.length * 1e3) ^ int(m.width * 1e3))
    # Simple deterministic random vector per morphology
    hv = rng.choice([-1, 1], size=dim)
    # Modulate by normalized feature magnitudes
    feats = np.array([m.length, m.width, m.height, m.mass])
    if feats.max() == 0:
        scale = np.ones_like(feats)
    else:
        scale = feats / feats.max()
    # Broadcast scaling onto the hypervector (first quarter for length, etc.)
    chunk = dim // 4
    for i, s in enumerate(scale):
        hv[i * chunk:(i + 1) * chunk] = hv[i * chunk:(i + 1) * chunk] * (1 if s >= 0.5 else -1)
    return hv


def shapley_weights(m: Morphology) -> Dict[str, float]:
    """
    Very light‑weight Shapley‑like attribution:
    each geometric feature contributes proportionally to its marginal increase
    over the mean of the other features.
    """
    feats = {
        "length": m.length,
        "width": m.width,
        "height": m.height,
        "mass": m.mass,
    }
    base = sum(feats.values()) / len(feats)
    contributions = {k: max(v - base, 0.0) for k, v in feats.items()}
    total = sum(contributions.values())
    if total == 0:
        # fallback to uniform attribution
        return {k: 1.0 / len(feats) for k in feats}
    return {k: v / total for k, v in contributions.items()}


class EndpointCircuitBreaker:
    """Tracks edge failures and opens the edge after a threshold."""
    def __init__(self, failure_threshold: int = 3, tolerance: float = 1e-4):
        self.failure_threshold = failure_threshold
        self.tolerance = tolerance
        self.failures = 0
        self.open = False

    def record(self, conductance: float) -> None:
        if conductance < self.tolerance:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open = True
        else:
            self.failures = 0  # reset on successful conductance


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_spatial_score(ei: Entity, ej: Entity, alpha: float,
                         max_distance: float) -> float:
    """
    Convex combination of normalized spatial similarity and recovery priority.
    Mirrors Parent A's `h(i,j)`.
    """
    d = haversine_distance((ei.lat, ei.lon), (ej.lat, ej.lon))
    spatial_sim = 1.0 - min(d / max_distance, 1.0)  # in [0,1]
    priority_sim = (ei.recovery_priority + ej.recovery_priority) / 2.0
    return alpha * spatial_sim + (1.0 - alpha) * priority_sim


def shapley_weighted_conductance(ei: Entity, ej: Entity) -> float:
    """
    Base conductance derived from Shapley attributions of the two endpoints.
    Conductance is the geometric mean of the weighted sums.
    """
    wi = shapley_weights(ei.morphology)
    wj = shapley_weights(ej.morphology)
    # Weighted sum of features (length, width, height, mass)
    sum_i = sum(wi[k] * getattr(ei.morphology, k) for k in wi)
    sum_j = sum(wj[k] * getattr(ej.morphology, k) for k in wj)
    # Ensure positivity
    sum_i = max(sum_i, 1e-6)
    sum_j = max(sum_j, 1e-6)
    return math.sqrt(sum_i * sum_j)


def hybrid_edge_conductance(ei: Entity, ej: Entity,
                            alpha: float, max_distance: float) -> float:
    """
    Final edge conductance G_ij = h(i,j) * g_ij.
    """
    h = hybrid_spatial_score(ei, ej, alpha, max_distance)
    g = shapley_weighted_conductance(ei, ej)
    return h * g


def build_hybrid_network(entities: List[Entity],
                         alpha: float = 0.6,
                         max_distance: float = 2_000_000.0) -> Tuple[np.ndarray,
                                                                     np.ndarray,
                                                                     List[Tuple[int, int]],
                                                                     List[EndpointCircuitBreaker]]:
    """
    Returns:
        pressures   – (N,) array of node pressures (sphericity index)
        conductances – (E,) array of edge conductances G_ij
        edges        – list of (i,j) index tuples
        breakers     – list of circuit breakers, one per edge
    """
    n = len(entities)
    pressures = np.empty(n, dtype=float)
    for idx, e in enumerate(entities):
        pressures[idx] = sphericity_index(e.morphology.length,
                                          e.morphology.width,
                                          e.morphology.height)

    edges: List[Tuple[int, int]] = []
    conductances: List[float] = []
    breakers: List[EndpointCircuitBreaker] = []

    for i in range(n):
        for j in range(i + 1, n):
            G = hybrid_edge_conductance(entities[i], entities[j],
                                        alpha, max_distance)
            edges.append((i, j))
            conductances.append(G)
            breakers.append(EndpointCircuitBreaker())

    return pressures, np.array(conductances, dtype=float), edges, breakers


def physarum_shapley_step(pressures: np.ndarray,
                          conductances: np.ndarray,
                          edges: List[Tuple[int, int]],
                          breakers: List[EndpointCircuitBreaker],
                          dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    One Physarum update:
        f_ij = G_ij * (p_i - p_j)          (flow on edge)
        p_i ← p_i + dt * ( Σ_in f - Σ_out f )
    Edge conductances are decayed proportionally to absolute flow magnitude,
    mimicking resource consumption, and monitored by circuit breakers.
    Returns updated pressures and conductances.
    """
    n = pressures.shape[0]
    flow = np.zeros(len(edges), dtype=float)

    # Compute flows
    for idx, (i, j) in enumerate(edges):
        if breakers[idx].open:
            flow[idx] = 0.0
            continue
        diff = pressures[i] - pressures[j]
        flow[idx] = conductances[idx] * diff

    # Divergence per node
    divergence = np.zeros(n, dtype=float)
    for idx, (i, j) in enumerate(edges):
        f = flow[idx]
        divergence[i] -= f
        divergence[j] += f

    # Pressure update (Euler step)
    pressures = pressures + dt * divergence

    # Conductance adaptation
    for idx in range(len(conductances)):
        # Decay proportional to absolute flow; ensure positivity
        decay = dt * 0.01 * abs(flow[idx])
        conductances[idx] = max(conductances[idx] - decay, 1e-6)
        breakers[idx].record(conductances[idx])

    return pressures, conductances


def encode_entities_hypervectors(entities: List[Entity],
                                 dim: int = 10_000) -> np.ndarray:
    """
    Returns an (N, dim) matrix where each row is the hypervector encoding of an entity.
    """
    hv_matrix = np.empty((len(entities), dim), dtype=int)
    for i, e in enumerate(entities):
        hv_matrix[i] = morphology_to_hypervector(e.morphology, dim)
    return hv_matrix


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small synthetic population
    random.seed(42)
    np.random.seed(42)

    def rand_morph() -> Morphology:
        return Morphology(
            length=random.uniform(0.5, 5.0),
            width=random.uniform(0.5, 5.0),
            height=random.uniform(0.5, 5.0),
            mass=random.uniform(1.0, 20.0)
        )

    entities = [
        Entity(
            id=f"E{i}",
            lat=random.uniform(-45, 45),
            lon=random.uniform(-90, 90),
            category="test",
            morphology=rand_morph(),
            recovery_priority=random.uniform(0.0, 1.0),
            address_signature=f"sig{i}"
        )
        for i in range(6)
    ]

    # Build network
    pressures, conductances, edges, breakers = build_hybrid_network(entities)

    # Run a few Physarum‑Shapley iterations
    for step in range(5):
        pressures, conductances = physarum_shapley_step(
            pressures, conductances, edges, breakers, dt=0.2
        )
        print(f"Step {step+1}: pressures={pressures.round(3)}")
        open_edges = sum(b.open for b in breakers)
        print(f"          open edges={open_edges}/{len(breakers)}")

    # Encode entities as hypervectors (demonstrates third required function)
    hv_matrix = encode_entities_hypervectors(entities, dim=2000)
    print(f"Hypervector shape: {hv_matrix.shape}, sample sum={hv_matrix[0].sum()}")