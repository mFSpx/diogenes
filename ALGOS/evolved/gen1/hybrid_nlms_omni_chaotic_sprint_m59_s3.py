# DARWIN HAMMER — match 59, survivor 3
# gen: 1
# parent_a: nlms.py (gen0)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:24:16Z

from __future__ import annotations

import math
import random
import sys
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core NLMS utilities (Algorithm A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error `target – w·x`.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error


# ----------------------------------------------------------------------
# Graph‑generation & seismic propagation (part of Algorithm B)
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, int]  # (src, dst, impedance)


def generate_synthetic_graph(num_nodes: int, avg_degree: int = 3) -> Tuple[Dict[NodeId, List[Tuple[NodeId, int]]], np.ndarray]:
    """
    Create a random undirected graph with integer impedances and a random
    feature matrix Φ (shape: num_nodes × feature_dim).

    Returns
    -------
    adjacency : dict
        Mapping node → list of (neighbor, impedance).
    features : np.ndarray
        Random features for each node (dtype float64).
    """
    random.seed(42)
    nodes = [f"n{i}" for i in range(num_nodes)]
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]] = {n: [] for n in nodes}
    edges: List[Edge] = []

    # Ensure connectivity with a simple chain
    for i in range(num_nodes - 1):
        impedance = random.choice([1, 5, 10, 20])
        edges.append((nodes[i], nodes[i + 1], impedance))

    # Add extra random edges
    extra_edges = num_nodes * avg_degree // 2 - (num_nodes - 1)
    while extra_edges > 0:
        a, b = random.sample(nodes, 2)
        if any(nb == b for nb, _ in adjacency[a]):
            continue
        impedance = random.choice([1, 5, 10, 20])
        edges.append((a, b, impedance))
        extra_edges -= 1

    # Populate adjacency list (undirected)
    for u, v, imp in edges:
        adjacency[u].append((v, imp))
        adjacency[v].append((u, imp))

    # Random feature matrix (feature_dim = 4)
    feature_dim = 4
    features = np.random.randn(num_nodes, feature_dim)

    return adjacency, features


def seismic_wavefront_velocities(
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    root: NodeId,
    max_visits: int = 10_000,
) -> Dict[NodeId, float]:
    """
    Perform a BFS‑style propagation that accumulates “stress” as the sum of
    impedances along the path.  The velocity at a node is defined as
    `v = 1 / max(stress, 1)` (mirroring Algorithm B).

    Returns
    -------
    velocities : dict
        Mapping node → velocity.
    """
    visited: set[NodeId] = set()
    velocities: Dict[NodeId, float] = {}
    queue: deque[Tuple[NodeId, int]] = deque([(root, 0)])
    visits = 0

    while queue and visits < max_visits:
        node, stress = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        velocities[node] = 1.0 / max(float(stress), 1.0)
        visits += 1
        for neighbor, impedance in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append((neighbor, stress + impedance))

    return velocities


# ----------------------------------------------------------------------
# Fluidic triage (Algorithm B) – used as an auxiliary feature source
# ----------------------------------------------------------------------
def fluidic_pressure_cells(event_stream: List[Dict]) -> Counter:
    """
    Count pressure per “voronoi_cell_id”.  Each event contributes a friction
    increment based on its epistemic flag.

    Returns
    -------
    Counter mapping cell_id → accumulated pressure.
    """
    pressure = Counter()
    for ev in event_stream:
        cell_id = ev.get("voronoi_cell_id", "default_theater")
        ep_flag = ev.get("epistemic_flag", "SURE_MAYBE")
        inc = 10 if ep_flag == "BULLSHIT" else (2 if ep_flag == "POSSIBLE" else 0)
        pressure[cell_id] += inc
    return pressure


# ----------------------------------------------------------------------
# Hybrid training step – the mathematical fusion
# ----------------------------------------------------------------------
def construct_input_vector(
    node: NodeId,
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    features: np.ndarray,
    node_index: Dict[NodeId, int],
) -> np.ndarray:
    """
    Construct input vector `x_i = Σ_j impedance_ij * φ_j`.

    Parameters
    ----------
    node : NodeId
        Node identifier.
    adjacency : dict
        Graph adjacency with impedances.
    features : np.ndarray
        Node feature matrix Φ (num_nodes × feature_dim).
    node_index : dict
        Mapping node identifier → row index in `features`.

    Returns
    -------
    x_i : np.ndarray
        Input vector.
    """
    x_i = np.zeros(features.shape[1])
    for neighbor, impedance in adjacency[node]:
        x_i += impedance * features[node_index[neighbor]]
    return x_i


def hybrid_train_step(
    weights: np.ndarray,
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    features: np.ndarray,
    node_index: Dict[NodeId, int],
    root: NodeId,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[NodeId, float]]:
    """
    One hybrid iteration:
      1. Run seismic propagation → target velocities `v_i`.
      2. For each visited node construct input `x_i = Σ_j impedance_ij * φ_j`.
      3. Predict `ŷ_i = w·x_i` via NLMS.
      4. Accumulate NLMS updates over all nodes (batch‑style) and return the
         new weight vector.

    Parameters
    ----------
    weights : np.ndarray
        Current global weight vector (shape: feature_dim,).
    adjacency : dict
        Graph adjacency with impedances.
    features : np.ndarray
        Node feature matrix Φ (num_nodes × feature_dim).
    node_index : dict
        Mapping node identifier → row index in `features`.
    root : str
        Identifier of the wavefront source node.
    mu, eps : float
        NLMS hyper‑parameters.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    velocities : dict
        Wavefront velocities computed in step 1 (for inspection).
    """
    # 1. Propagation
    velocities = seismic_wavefront_velocities(adjacency, root)

    # 2‑4. NLMS batch update
    grad_sum = np.zeros_like(weights)
    total_error = 0.0
    count = 0

    for node, velocity in velocities.items():
        x_i = construct_input_vector(node, adjacency, features, node_index)
        prediction = nlms_predict(weights, x_i)
        error = velocity - prediction
        total_error += abs(error)
        count += 1
        delta = mu * error * x_i / (np.dot(x_i, x_i) + eps)
        grad_sum += delta

    new_weights = weights + grad_sum
    return new_weights, velocities


def main():
    num_nodes = 100
    avg_degree = 3
    adjacency, features = generate_synthetic_graph(num_nodes, avg_degree)
    node_index = {node: i for i, node in enumerate(adjacency)}
    weights = np.random.rand(features.shape[1])

    for _ in range(10):
        weights, velocities = hybrid_train_step(weights, adjacency, features, node_index, "n0")

    print("Final weights:", weights)


if __name__ == "__main__":
    main()