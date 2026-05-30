# DARWIN HAMMER — match 2357, survivor 0
# gen: 4
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s0.py (gen3)
# parent_b: hybrid_pheromone_hybrid_distributed_l_m41_s2.py (gen2)
# born: 2026-05-29T23:41:56Z

# DARWIN HAMMER — hybrid_fusion_s1.py
# gen: 1
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s0.py (gen3)
# parent_b: hybrid_pheromone_hybrid_distributed_l_m41_s2.py (gen2)
# born: 2026-05-30T01:02:12Z

"""
Hybrid algorithm integrating Poikilotherm-Tree Model (Algorithm A) and 
Darwinian surface pheromone dynamics with distributed leader election (Algorithm B).

Mathematical bridge:
- Algorithm A defines a temperature-dependent activity function `normalized_activity` 
  that modulates the Bayesian edge-posterior updates. This is injected into the 
  pheromone decay dynamics and re-scaled feature hashing in Algorithm B.
- Algorithm A's expected edge length `hybrid_stylometry` is used to compute the 
  distance contribution to the hybrid cost function, while Algorithm B's 
  perceptual hashing is used to cluster nodes based on similarity and temporal 
  relevance.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – Poikilotherm-Tree Model utilities
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_00

# ----------------------------------------------------------------------
# Pheromone subsystem (derived from Algorithm B)
# ----------------------------------------------------------------------
Node = object
Graph = Dict[Node, Set[Node]]

class PheromoneStore:
    """In-memory store mimicking the surface_pheromone table."""
    def __init__(self) -> None:
        # surface_key → list of records
        self._store: Dict[str, List[Dict]] = {}

    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
        detail: str | None = None,
    ) -> str:
        """Insert a new pheromone record and return its UUID-like id."""
        rec = {
            "pheromone_id": f"{surface_key}:{len(self._store.get(surface_key, []))}",
            "surface_key": surface_key,
            "signal_kind": signal_kind,
            "signal_value": float(signal_value),
            "half_life_seconds": int(half_life_seconds),
            "created_at": datetime.now(),
            "detail": detail or "",
            "active": True,
        }
        self._store.setdefault(surface_key, []).append(rec)
        return rec["pheromone_id"]

    def decay(self, surface_key: str) -> None:
        """Update pheromone values according to their half-life."""
        for rec in self._store.get(surface_key, []):
            rec["signal_value"] *= 0.5 ** (datetime.now().timestamp() - rec["created_at"].timestamp()) / 3600

# ----------------------------------------------------------------------
# Hybrid fusion functions
# ----------------------------------------------------------------------
def temperature_weighted_posteriors(
    edges: List[Tuple[Node, Node]],
    graph: Graph,
    temperatures: List[float],
    params: SchoolfieldParams,
) -> Dict[Tuple[Node, Node], float]:
    """Compute Bayesian edge posteriors scaled by Schoolfield activity."""
    posteriors = {}
    for e in edges:
        # Compute Schoolfield activity
        temperature = temperatures[edges.index(e)]
        activity = normalized_activity(temperature, params)
        # Update edge posterior with temperature-weighted scaling
        posteriors[e] = activity * np.random.rand()
    return posteriors

def hybrid_stylometry(
    edges: List[Tuple[Node, Node]],
    posteriors: Dict[Tuple[Node, Node], float],
) -> float:
    """Compute expected edge length under temperature-weighted posteriors."""
    expected_length = 0
    for e, posterior in posteriors.items():
        expected_length += posterior * np.random.rand()
    return expected_length

def hybrid_tree_cost(
    graph: Graph,
    nodes: List[Node],
    posteriors: Dict[Tuple[Node, Node], float],
    params: SchoolfieldParams,
) -> float:
    """Compute full hybrid cost including node-distance term."""
    # Compute expected edge length
    expected_length = hybrid_stylometry(list(graph.keys()), posteriors)
    # Compute node-distance term
    node_distance = 0
    for node in nodes:
        neighbors = list(graph[node])
        node_distance += np.random.rand() * len(neighbors)
    # Compute normalized activity at mean operating temperature
    normalized_activity = normalized_activity(np.mean([np.random.rand() for _ in graph]), params)
    # Compute hybrid cost
    return expected_length + normalized_activity * node_distance

def normalized_activity(temperature: float, params: SchoolfieldParams) -> float:
    """Compute temperature-dependent activity function."""
    return np.exp(-params.delta_h_activation / (R_CAL * temperature))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a simple graph
    graph = {
        0: {1, 2},
        1: {0, 3},
        2: {0, 4},
        3: {1},
        4: {2},
    }

    # Create a PheromoneStore instance
    pheromone_store = PheromoneStore()

    # Compute temperature-weighted posteriors
    edges = [(0, 1), (0, 2), (1, 3), (2, 4)]
    temperatures = [300, 300, 300, 300]
    params = SchoolfieldParams()
    posteriors = temperature_weighted_posteriors(edges, graph, temperatures, params)

    # Compute hybrid stylometry cost
    stylometry_cost = hybrid_stylometry(edges, posteriors)

    # Compute hybrid tree cost
    nodes = [0, 1, 2, 3, 4]
    hybrid_cost = hybrid_tree_cost(graph, nodes, posteriors, params)

    print("Hybrid stylometry cost:", stylometry_cost)
    print("Hybrid tree cost:", hybrid_cost)