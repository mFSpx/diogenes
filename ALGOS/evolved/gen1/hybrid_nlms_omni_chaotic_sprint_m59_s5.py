# DARWIN HAMMER — match 59, survivor 5
# gen: 1
# parent_a: nlms.py (gen0)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:24:16Z

from __future__ import annotations

import math
import random
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


def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    targets: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a *batch* NLMS update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape: (d,)).
    X : np.ndarray
        Input matrix where each row is an input vector x_i (shape: (N, d)).
    targets : np.ndarray
        Desired scalar outputs v_i (shape: (N,)).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    errors : np.ndarray
        Prediction errors for each sample.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")

    # Predictions and errors
    preds = X @ weights
    errors = targets - preds

    # Normalized step for each sample
    powers = np.sum(X * X, axis=1) + eps  # shape (N,)
    steps = (mu * errors / powers)[:, None] * X   # shape (N, d)

    # Aggregate the per‑sample steps
    delta_w = steps.sum(axis=0)
    new_weights = weights + delta_w
    return new_weights, errors


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
    np.random.seed(42)

    nodes = [f"n{i}" for i in range(num_nodes)]
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]] = {n: [] for n in nodes}
    edges: List[Edge] = []

    # Ensure connectivity with a simple chain
    for i in range(num_nodes - 1):
        impedance = random.choice([1, 5, 10, 20])
        edges.append((nodes[i], nodes[i + 1], impedance))

    # Add extra random edges
    extra_edges = max(num_nodes * avg_degree // 2 - (num_nodes - 1), 0)
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
    `v = 1 / max(stress, 1)`.

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
# Fluidic triage (Algorithm B) – auxiliary feature source
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
# Helper: build impedance‑weighted neighbourhood vectors
# ----------------------------------------------------------------------
def build_input_vectors(
    adjacency: Dict[NodeId, List[Tuple[NodeId, int]]],
    features: np.ndarray,
    node_index: Dict[NodeId, int],
) -> Tuple[np.ndarray, List[NodeId]]:
    """
    For every node that appears in ``adjacency`` compute

        x_i = Σ_j impedance_ij · φ_j

    where φ_j is the feature vector of neighbour ``j``.
    Returns a matrix X (rows correspond to nodes in the same order as
    ``node_list``) and the list of node identifiers.
    """
    dim = features.shape[1]
    X_rows: List[np.ndarray] = []
    node_list: List[NodeId] = []

    for node, nbrs in adjacency.items():
        vec = np.zeros(dim, dtype=float)
        for nbr, imp in nbrs:
            idx = node_index.get(nbr)
            if idx is None:
                continue
            vec += imp * features[idx]
        X_rows.append(vec)
        node_list.append(node)

    X = np.vstack(X_rows) if X_rows else np.empty((0, dim))
    return X, node_list


# ----------------------------------------------------------------------
# Hybrid training step – the mathematical fusion (improved)
# ----------------------------------------------------------------------
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

    1. Run seismic propagation → target velocities ``v_i``.
    2. Build impedance‑weighted input vectors ``x_i`` for every visited node.
    3. Perform a *batch* NLMS update that minimises the mean‑square error
       between the predicted velocities ``ŷ_i = w·x_i`` and the true
       velocities ``v_i``.
    4. Return the updated weight vector and the velocity map (for inspection).

    Parameters
    ----------
    weights : np.ndarray
        Current global weight vector (shape: (feature_dim,)).
    adjacency : dict
        Graph adjacency with impedances.
    features : np.ndarray
        Node feature matrix Φ (num_nodes × feature_dim).
    node_index : dict
        Mapping node identifier → row index in ``features``.
    root : str
        Identifier of the wavefront source node.
    mu, eps : float
        NLMS hyper‑parameters.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    velocities : dict
        Wavefront velocities computed in step 1.
    """
    # 1. Propagation (targets)
    velocities = seismic_wavefront_velocities(adjacency, root)

    # 2. Build input vectors only for nodes that received a velocity
    #    (unvisited nodes are ignored because they have no target).
    visited_nodes = list(velocities.keys())
    sub_adj = {n: adjacency[n] for n in visited_nodes}
    X, node_order = build_input_vectors(sub_adj, features, node_index)

    if X.shape[0] == 0:
        # No visited nodes – nothing to learn.
        return weights.copy(), velocities

    # Align targets with the order of rows in X
    targets = np.array([velocities[n] for n in node_order], dtype=float)

    # 3. Batch NLMS update
    new_weights, _ = nlms_batch_update(weights, X, targets, mu=mu, eps=eps)

    return new_weights, velocities


# ----------------------------------------------------------------------
# Example demo (kept minimal for clarity)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic graph
    adjacency, feats = generate_synthetic_graph(num_nodes=50, avg_degree=4)

    # Mapping from node id to feature‑matrix row index
    node_to_idx = {f"n{i}": i for i in range(feats.shape[0])}

    # Initialise NLMS weight vector (same dimension as node features)
    w = np.zeros(feats.shape[1], dtype=float)

    # Run a few hybrid training iterations
    root_node = "n0"
    for epoch in range(5):
        w, vel_map = hybrid_train_step(
            weights=w,
            adjacency=adjacency,
            features=feats,
            node_index=node_to_idx,
            root=root_node,
            mu=0.3,
            eps=1e-8,
        )
        mse = np.mean([(v - nlms_predict(w, sum(
            imp * feats[node_to_idx[nbr]]
            for nbr, imp in adjacency[node]
        ))) ** 2 for node, v in vel_map.items() if node in adjacency])
        print(f"Epoch {epoch + 1}: weight norm = {np.linalg.norm(w):.4f}, MSE ≈ {mse:.6f}")