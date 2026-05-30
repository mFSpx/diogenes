# DARWIN HAMMER — match 974, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_krampu_m9_s5.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s2.py (gen3)
# born: 2026-05-29T23:32:00Z

"""
Hybrid Bandit-Router with Graph Curvature and Ternary Route Endpoint Circuit Breaker
===============================================================================
Parents:
- hybrid_bandit_router_honeybee_store_m9_s5.py (Bandit action selection + Honeybee store dynamics + graph-matrix operations for Ollivier-Ricci curvature)
- hybrid_ternary_route_hybrid_endpoint_circ_m233_s2.py (Ternary router with minimum-cost tree evaluation and Bayesian edge probability updates + EndpointCircuitBreaker)

Mathematical Bridge:
This fusion combines the bandit action selection and graph curvature estimation with the ternary router and endpoint circuit breaker.
The bandit action selection is used to choose the next node in the route, and the graph curvature estimation is used to modulate the weights of the edges in the route.
The ternary router is used to evaluate the minimum-cost tree, and the endpoint circuit breaker is used to adapt the failure threshold based on the morphology of the spatial structure.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Bandit / Store components
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the global reward statistics."""
    _POLICY.clear()

def update_policy(updates: List["BanditUpdate"]) -> None:
    """Accumulate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action: str) -> float:
    """Mean reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    action_id: str
    reward: float

# ----------------------------------------------------------------------
# Graph Curvature components
# ----------------------------------------------------------------------
def graph_curvature(adjacency: np.ndarray) -> np.ndarray:
    """Compute the graph curvature."""
    n_nodes = adjacency.shape[0]
    curvature = np.zeros(n_nodes)
    for i in range(n_nodes):
        neighbors = np.where(adjacency[i, :] > 0)[0]
        curvature[i] = np.sum(adjacency[i, neighbors]) / len(neighbors)
    return curvature

def update_adjacency(adjacency: np.ndarray, node: int, delta: float) -> np.ndarray:
    """Update the adjacency matrix based on the net resource flow."""
    adjacency[node, :] += delta
    adjacency[:, node] += delta
    return adjacency

# ----------------------------------------------------------------------
# Ternary Router components
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]
Morphology = Tuple[float, float, float]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
) -> float:
    """
    Compute material + weighted path cost.
    If edge_weights are supplied they replace the geometric length
    with the Bayesian-expected length
    """
    total_cost = 0.0
    for edge in edges:
        if edge_weights is None:
            total_cost += length(nodes[edge[0]], nodes[edge[1]])
        else:
            total_cost += edge_weights.get(edge, 0.0)
    return total_cost

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def hybrid_select_action(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
) -> str:
    """Select the next node in the route using the bandit action selection."""
    actions = list(nodes.keys())
    rewards = [_reward(action) for action in actions]
    best_action = np.argmax(rewards)
    return actions[best_action]

def hybrid_update_adjacency(
    adjacency: np.ndarray,
    node: int,
    delta: float,
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
) -> np.ndarray:
    """Update the adjacency matrix based on the net resource flow and the morphology of the spatial structure."""
    morphology = compute_morphology(nodes, edges)
    threshold = compute_threshold(morphology)
    if delta > threshold:
        adjacency = update_adjacency(adjacency, node, delta)
    return adjacency

def compute_morphology(nodes: Dict[str, Point], edges: List[Edge]) -> Morphology:
    """Compute the morphology of the spatial structure."""
    x_coords = [node[0] for node in nodes.values()]
    y_coords = [node[1] for node in nodes.values()]
    length = max(x_coords) - min(x_coords)
    width = max(y_coords) - min(y_coords)
    height = 0.0  # Assume 2D structure
    return (length, width, height)

def compute_threshold(morphology: Morphology) -> float:
    """Compute the failure threshold based on the morphology of the spatial structure."""
    length, width, height = morphology
    sphericity = length / (length + width + height)
    flatness = width / (length + width + height)
    alpha = 0.5
    beta = 0.2
    base_threshold = 0.1
    threshold = base_threshold * (1 + alpha * (1 - sphericity) + beta * flatness)
    return threshold

if __name__ == "__main__":
    nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (0.0, 1.0)}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"
    path_weight = 0.2
    edge_weights = None
    adjacency = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    node = 0
    delta = 0.5
    selected_action = hybrid_select_action(nodes, edges, root, path_weight, edge_weights)
    updated_adjacency = hybrid_update_adjacency(adjacency, node, delta, nodes, edges, root, path_weight, edge_weights)
    print("Selected Action:", selected_action)
    print("Updated Adjacency:")
    print(updated_adjacency)