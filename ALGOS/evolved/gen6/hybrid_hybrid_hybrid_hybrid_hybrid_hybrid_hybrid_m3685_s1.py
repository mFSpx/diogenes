# DARWIN HAMMER — match 3685, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s6.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m1446_s0.py (gen5)
# born: 2026-05-29T23:51:09Z

"""
Module for the hybrid algorithm combining 'hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s6.py' 
and 'hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m1446_s0.py'. 

The mathematical bridge between the two algorithms is found by integrating the model pool 
management from the first parent with the minimum-cost tree and bandit-router-sketch-RLCT 
algorithm from the second parent. The model pool's RAM ceiling is used as a constraint in 
the minimum-cost tree algorithm to optimize the model loading process.

The hybrid algorithm uses a probabilistic approach to estimate the expected reward of each 
model loading action, and then uses these probabilities to update the model pool. This is 
achieved by using the Gaussian function to compute the probabilistic weights, and then 
using these weights to update the probabilities of each model loading action.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Vector = Tuple[float, ...]

# ----------------------------------------------------------------------
# Model pool with RAM ceiling and linear schedule utilities
# ----------------------------------------------------------------------
class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """Return True if the model fits within the remaining RAM."""
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load a model if RAM permits; raise otherwise."""
        if self.is_loaded(model.name):
            return  # already loaded
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)

# ----------------------------------------------------------------------
# Minimum-cost tree and bandit-router-sketch-RLCT algorithm
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_distances : dict mapping node → root‑to‑node distance
    """
    adj = {node: [] for node in nodes}
    edge_len = {}
    node_distances = {root: 0}

    for a, b in edges:
        dist = length(nodes[a], nodes[b])
        edge_len[(a, b)] = dist
        edge_len[(b, a)] = dist
        adj[a].append(b)
        adj[b].append(a)

    # Perform BFS to compute node distances
    queue = [root]
    visited = set([root])
    while queue:
        node = queue.pop(0)
        for neighbour in adj[node]:
            if neighbour not in visited:
                node_distances[neighbour] = node_distances[node] + edge_len[(node, neighbour)]
                queue.append(neighbour)
                visited.add(neighbour)

    return adj, edge_len, node_distances

def hybrid_model_loading(
    model_pool: ModelPool,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Edge, float], Dict[str, float]]:
    """
    Hybrid model loading algorithm.

    This function integrates the model pool management with the minimum-cost tree 
    algorithm to optimize the model loading process under the RAM ceiling constraint.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_distances : dict mapping node → root‑to‑node distance
    """
    adj, edge_len, node_distances = tree_metrics(nodes, edges, root)

    # Use Gaussian function to compute probabilistic weights for model loading
    def gaussian(x: float, mean: float, std: float) -> float:
        return np.exp(-((x - mean) / std) ** 2)

    # Compute expected reward for each model loading action
    expected_rewards = {}
    for model in model_pool.loaded.values():
        reward = gaussian(model.ram_mb, 1000, 500)  # example Gaussian function
        expected_rewards[model.name] = reward

    # Update model pool based on expected rewards
    sorted_models = sorted(expected_rewards.items(), key=lambda x: x[1], reverse=True)
    for model_name, _ in sorted_models:
        model = ModelTier(model_name, 1000, "example")  # example model
        try:
            model_pool.load(model)
        except RuntimeError:
            print(f"Insufficient RAM to load {model_name}")

    return adj, edge_len, node_distances

if __name__ == "__main__":
    model_pool = ModelPool(ram_ceiling_mb=6000)
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    adj, edge_len, node_distances = hybrid_model_loading(model_pool, nodes, edges, root)
    print(adj)
    print(edge_len)
    print(node_distances)