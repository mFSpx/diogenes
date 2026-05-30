# DARWIN HAMMER — match 59, survivor 2
# gen: 1
# parent_a: nlms.py (gen0)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:24:16Z

"""Hybrid NLMS‑Graph Engine
================================
This module fuses two distinct parent algorithms:

* **Algorithm A – nlms.py** – a normalized least‑mean‑squares adaptive filter that
  updates a weight vector `w` using the rule  
  `w ← w + μ·e·x / (‖x‖²+ε)` where `e = target – w·x`.

* **Algorithm B – omni_chaotic_sprint.py** – a graph‑propagation engine that
  (a) builds an adjacency structure with impedance‑weighted edges,
  (b) performs a breadth‑first “seismic” wave that yields a *wavefront velocity*
      `v_i = 1 / max(stress_i,1)` for each visited node, and
  (c) extracts fluidic‑pressure statistics from an event stream.

**Mathematical Bridge**

For every node `i` we construct an *input vector*  


x_i = Σ_j  (impedance_ij) · φ_j


where `φ_j` is the feature vector of neighbor `j`.  
The NLMS predictor `ŷ_i = w·x_i` attempts to model the wavefront velocity
`v_i`.  The NLMS error `e_i = v_i – ŷ_i` is then used to adapt the global
weight vector `w`.  In this way the adaptive filter learns to map the
impedance‑weighted neighbourhood composition to the observed propagation
speed, tightly coupling the two parent topologies into a single hybrid
system.

The module provides a self‑contained demo that generates a synthetic graph,
runs the seismic propagation, builds feature matrices, and performs NLMS
updates.
"""

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

    for node, v_target in velocities.items():
        idx = node_index[node]
        # Build neighbourhood‑weighted input vector x_i
        x_i = np.zeros_like(weights)
        for neighbor, impedance in adjacency.get(node, []):
            n_idx = node_index[neighbor]
            x_i += impedance * features[n_idx]
        # NLMS update for this sample (but we accumulate the delta)
        y_pred = nlms_predict(weights, x_i)
        error = v_target - y_pred
        power = float(x_i @ x_i) + eps
        delta = mu * error * x_i / power
        grad_sum += delta
        total_error += abs(error)
        count += 1

    if count == 0:
        return weights, velocities

    new_weights = weights + grad_sum / count
    return new_weights, velocities


# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    NUM_NODES = 50
    FEATURE_DIM = 4

    # 1. Synthetic graph & features
    adjacency, features = generate_synthetic_graph(NUM_NODES)
    node_ids = list(adjacency.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    # 2. Initialise NLMS weight vector (random)
    rng = np.random.default_rng(0)
    weights = rng.standard_normal(FEATURE_DIM)

    # 3. Choose a root node for the seismic wave
    root_node = node_ids[0]

    # 4. Run a few hybrid training iterations
    print("Starting hybrid NLMS‑Graph training demo")
    for epoch in range(5):
        weights, velocities = hybrid_train_step(
            weights,
            adjacency,
            features,
            node_index,
            root_node,
            mu=0.7,
        )
        avg_vel = sum(velocities.values()) / len(velocities)
        print(f"Epoch {epoch+1:02d}: avg velocity={avg_vel:.4f}, weight norm={np.linalg.norm(weights):.4f}")

    # 5. Fluidic triage example (independent but shows integration)
    synthetic_events = [
        {"voronoi_cell_id": f"cell{random.randint(0,5)}", "epistemic_flag": random.choice(["BULLSHIT", "POSSIBLE", "SURE_MAYBE"])}
        for _ in range(30)
    ]
    pressure = fluidic_pressure_cells(synthetic_events)
    high_pressure = [c for c, p in pressure.items() if p > 50]
    print(f"Fluidic triage: {len(pressure)} cells tracked, high‑pressure cells: {high_pressure}")

if __name__ == "__main__":
    _demo()