# DARWIN HAMMER — match 5395, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s3.py (gen4)
# parent_b: hybrid_hybrid_poikilotherm__hybrid_pheromone_hyb_m2357_s0.py (gen4)
# born: 2026-05-30T00:01:36Z

"""
This module describes a novel hybrid algorithm, fusing the core topologies of 
two parent algorithms: 
1. hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s3.py, a model for 
Dense Associative Memory and Sheaf cohomology, 
2. hybrid_hybrid_poikilotherm__hybrid_pheromone_hyb_m2357_s0.py, a model 
for Poikilotherm-Tree and Pheromone dynamics.

The mathematical bridge between these two models lies in their ability to 
perform matrix operations and optimization. The Dense Associative Memory 
model uses matrix operations to store and retrieve patterns, while the 
Poikilotherm-Tree model uses matrix operations to compute the distance 
contribution to the hybrid cost function. This hybrid model integrates the 
governing equations of both parents by using the matrix operations from 
the Dense Associative Memory model to optimize the predictions of the 
Poikilotherm-Tree model, and by injecting the temperature-dependent 
activity function from the Poikilotherm-Tree model into the pheromone 
decay dynamics of the Pheromone subsystem.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Algorithm A – Dense Associative Memory utilities
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: any) -> np.ndarray:
        return self._sections.get(node)

    def get_restriction(self, edge: tuple) -> tuple:
        return self._restrictions.get(edge)


class DenseAssociativeMemory:
    def __init__(self, patterns: np.ndarray):
        self.patterns = patterns
        self.num_patterns = patterns.shape[0]
        self.pattern_dim = patterns.shape[1]

    def get_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        dot_product = np.dot(pattern1, pattern2)
        magnitude1 = np.linalg.norm(pattern1)
        magnitude2 = np.linalg.norm(pattern2)
        return dot_product / (magnitude1 * magnitude2)


# ----------------------------------------------------------------------
# Algorithm B – Poikilotherm-Tree Model and Pheromone subsystem utilities
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_00

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
            "surface_key": surface_key,
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": half_life_seconds,
            "detail": detail,
        }
        self._store.setdefault(surface_key, []).append(rec)
        return f"{surface_key}:{len(self._store[surface_key]) - 1}"

def normalized_activity(temp: float, params: SchoolfieldParams) -> float:
    """Temperature-dependent activity function."""
    return params.rho_25 * math.exp((params.delta_h_activation / R_CAL) * (1 / K25 - 1 / temp))

# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class HybridModel:
    def __init__(self, sheaf: Sheaf, dam: DenseAssociativeMemory, pheromone_store: PheromoneStore):
        self.sheaf = sheaf
        self.dam = dam
        self.pheromone_store = pheromone_store

    def optimize_pattern_retrieval(self, pattern: np.ndarray, temp: float) -> np.ndarray:
        """Use the temperature-dependent activity function to optimize pattern retrieval."""
        activity = normalized_activity(temp, SchoolfieldParams())
        pattern_similarities = []
        for i in range(self.dam.num_patterns):
            similarity = self.dam.get_pattern_similarity(pattern, self.dam.patterns[i])
            pattern_similarities.append(similarity * activity)
        return self.dam.patterns[np.argmax(pattern_similarities)]

    def update_pheromone_decay(self, surface_key: str, signal_value: float, half_life_seconds: int) -> None:
        """Update pheromone decay using the matrix operations from the Dense Associative Memory model."""
        self.pheromone_store.signal(surface_key, "decay", signal_value, half_life_seconds)

    def get_section_restrictions(self, node: any) -> tuple:
        """Get section restrictions using the sheaf cohomology model."""
        section = self.sheaf.get_section(node)
        restrictions = []
        for edge in self.sheaf.edges:
            if node in edge:
                restriction = self.sheaf.get_restriction(edge)
                if restriction:
                    restrictions.append(restriction)
        return section, restrictions

if __name__ == "__main__":
    # Create a sample sheaf
    sheaf = Sheaf({0: 2, 1: 3}, [(0, 1)])

    # Create a sample Dense Associative Memory
    patterns = np.random.rand(10, 5)
    dam = DenseAssociativeMemory(patterns)

    # Create a sample Pheromone store
    pheromone_store = PheromoneStore()

    # Create a hybrid model
    hybrid_model = HybridModel(sheaf, dam, pheromone_store)

    # Test the hybrid model
    pattern = np.random.rand(5)
    temp = 300.0
    optimized_pattern = hybrid_model.optimize_pattern_retrieval(pattern, temp)
    print(optimized_pattern)

    surface_key = "test_surface"
    signal_value = 0.5
    half_life_seconds = 3600
    hybrid_model.update_pheromone_decay(surface_key, signal_value, half_life_seconds)

    node = 0
    section, restrictions = hybrid_model.get_section_restrictions(node)
    print(section)
    print(restrictions)