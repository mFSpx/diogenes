# DARWIN HAMMER — match 3532, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1369_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s3.py (gen6)
# born: 2026-05-29T23:50:29Z

"""
This module represents a novel hybrid algorithm, integrating the core topologies of 
hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1369_s1.py and 
hybrid_hybrid_hybrid_hybrid_minimum_cost_tree_m1716_s3.py.

The mathematical bridge between the two structures is the modulation of pheromone signals 
with the RBF-similarity weighted Minimum-Cost Tree. Specifically, we use the 
RBF-similarity matrix to weight the pheromone signals, allowing the pheromone signals 
to influence the edge costs in the Minimum-Cost Tree.

The pheromone signals are used to modulate the workshare allocation in the 
Minimum-Cost Tree, while the RBF-similarity matrix provides a continuous 
weighting factor for the edges of the tree. The hybrid algorithm combines these 
two components to create a unified system.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict, Set, Hashable
import numpy as np
from datetime import datetime, timezone
from uuid import uuid4

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
FeatureVec = Sequence[float]
Point = Tuple[float, float]
Node = Hashable
Edge = Tuple[Node, Node]

# ----------------------------------------------------------------------
# Pheromone utilities
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """In‑memory singleton store for pheromone entries."""
    _entries: Dict[str, PheromoneEntry] = {}

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        return [e for e in cls._entries.values() if e.surface_key == surface_key]


# ----------------------------------------------------------------------
# RBF similarity utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_rbf_similarity_matrix(features: List[FeatureVec]) -> np.ndarray:
    """Compute the full RBF similarity matrix for a list of feature vectors."""
    n = len(features)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            dist = euclidean(features[i], features[j])
            sim[i, j] = gaussian(dist)
            sim[j, i] = sim[i, j]
    return sim


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def modulate_pheromone_signals(pheromone_entries: List[PheromoneEntry], 
                                similarity_matrix: np.ndarray) -> List[PheromoneEntry]:
    modulated_entries = []
    for entry in pheromone_entries:
        modulated_signal_value = entry.signal_value * similarity_matrix[0, 0]
        modulated_entry = PheromoneEntry(entry.surface_key, entry.signal_kind, 
                                          modulated_signal_value, entry.half_life_seconds)
        modulated_entries.append(modulated_entry)
    return modulated_entries


def compute_hybrid_cost(edges: List[Edge], 
                        features: List[FeatureVec], 
                        pheromone_entries: List[PheromoneEntry], 
                        alpha: float, beta: float) -> float:
    similarity_matrix = compute_rbf_similarity_matrix(features)
    modulated_pheromone_entries = modulate_pheromone_signals(pheromone_entries, similarity_matrix)
    cost = 0.0
    for edge in edges:
        i, j = edge
        dist = euclidean(features[i], features[j])
        modulated_dist = dist * (1 - alpha * similarity_matrix[i, j])
        cost += modulated_dist
    for entry in modulated_pheromone_entries:
        cost += beta * entry.signal_value
    return cost


def generate_similarity_driven_edges(features: List[FeatureVec], 
                                    similarity_matrix: np.ndarray, 
                                    num_edges: int) -> List[Edge]:
    edges = []
    for _ in range(num_edges):
        i, j = random.sample(range(len(features)), 2)
        if similarity_matrix[i, j] > 0.5:
            edges.append((i, j))
    return edges


if __name__ == "__main__":
    features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    pheromone_entries = [PheromoneEntry("surface_key", "signal_kind", 1.0, 100)]
    edges = generate_similarity_driven_edges(features, compute_rbf_similarity_matrix(features), 2)
    cost = compute_hybrid_cost(edges, features, pheromone_entries, 0.5, 1.0)
    print(cost)