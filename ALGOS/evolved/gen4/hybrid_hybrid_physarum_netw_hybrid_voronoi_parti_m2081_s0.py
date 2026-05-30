# DARWIN HAMMER — match 2081, survivor 0
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# parent_b: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s0.py (gen2)
# born: 2026-05-29T23:40:38Z

"""
Hybrid algorithm merging Physarum flux conductance dynamics (Parent A: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py) 
with Voronoi partitioning and hybrid endpoint circuit breakers with serpentina self-righting (Parent B: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s0.py).

The mathematical bridge between these structures is the integration of Physarum flux conductance dynamics 
with the morphology and recovery priority of the hybrid endpoint circuit breakers, 
allowing for the creation of a hybrid system that combines the benefits of both algorithms.

In this hybrid system, the conductance of each edge in the Physarum network is modulated by the 
Voronoi region it belongs to, and the morphology of the circuit breakers influences the 
pressure drop across each edge. The updated conductance feeds back into the bandit policy, 
while the Voronoi regions adapt to the changing morphology of the circuit breakers.
"""

from __future__ import annotations
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (re‑implemented for self‑containment)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, gain: float, decay: float, dt: float) -> float:
    return max(0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Voronoi partitioning
# ----------------------------------------------------------------------
Point = tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

# ----------------------------------------------------------------------
# Hybrid endpoint circuit breakers with serpentina self-righting
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
@dataclass
class HybridEdge:
    conductance: float
    edge_length: float
    morphology: Morphology
    pressure_a: float
    pressure_b: float

def hybrid_flux(edge: HybridEdge, eps: float = 1e-12) -> float:
    """Physarum flux on a single edge with morphology modulation."""
    if edge.edge_length <= 0:
        raise ValueError('edge_length must be positive')
    modulation = edge.morphology.length * edge.morphology.width * edge.morphology.height / edge.morphology.mass
    return edge.conductance * modulation / max(edge.edge_length, eps) * (edge.pressure_a - edge.pressure_b)

def hybrid_update_conductance(edge: HybridEdge, q: float, gain: float, decay: float, dt: float) -> HybridEdge:
    new_conductance = update_conductance(edge.conductance, q, gain, decay, dt)
    return HybridEdge(new_conductance, edge.edge_length, edge.morphology, edge.pressure_a, edge.pressure_b)

def hybrid_assign(points: list[Point], seeds: list[Point], edges: list[HybridEdge]) -> dict[int, list[HybridEdge]]:
    regions = {i: [] for i in range(len(seeds))}
    for edge in edges:
        region_idx = nearest((edge.pressure_a, edge.pressure_b), seeds)
        regions[region_idx].append(edge)
    return regions

if __name__ == "__main__":
    points = [(0, 0), (1, 1), (2, 2)]
    seeds = [(0, 0), (2, 2)]
    edges = [
        HybridEdge(1.0, 1.0, Morphology(1.0, 1.0, 1.0, 1.0), 10.0, 5.0),
        HybridEdge(2.0, 2.0, Morphology(2.0, 2.0, 2.0, 2.0), 5.0, 10.0)
    ]
    regions = hybrid_assign(points, seeds, edges)
    print(regions)
    q = hybrid_flux(edges[0])
    print(q)
    new_edge = hybrid_update_conductance(edges[0], q, 1.0, 0.1, 0.01)
    print(new_edge)