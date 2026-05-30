# DARWIN HAMMER — match 4458, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_ternar_m55_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s3.py (gen4)
# born: 2026-05-29T23:55:50Z

"""Hybrid Krampus‑Ollivier Regret‑Weighted Graph Analyzer

This module fuses the two parent algorithms:

* **Parent A** – builds a geometric graph from a high‑dimensional matrix,
  computes an Ollivier‑Ricci curvature per node and augments the 3‑D
  brain‑map coordinates with that curvature.

* **Parent B** – creates a set of ``MathAction`` objects, computes
  regret‑weighted probabilities for them and uses those probabilities to
  prune a structure.

**Mathematical bridge** – we interpret each node of the graph as an
``MathAction``.  The curvature of a node quantifies local geometric
tightness, while the regret‑weighted probability quantifies the
decision‑theoretic desirability of the corresponding action.  By
multiplying the two scalars we obtain a *combined importance weight* that
drives both the pruning of edges (graph‑level) and the final embedding
into a 3‑D space (brain‑map level).  This creates a single unified system
that simultaneously respects geometric consistency and regret‑aware
decision making.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """A decision action with an expected value used for regret weighting."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Core functions (Hybrid operations)
# ----------------------------------------------------------------------
def hybrid_build_adj(matrix: np.ndarray, distance_threshold: float = 1.0) -> List[Tuple[int, int]]:
    """
    Build an undirected adjacency list from rows of ``matrix``.
    Two nodes are connected if their Euclidean distance is below
    ``distance_threshold``.
    """
    num_nodes = matrix.shape[0]
    adj: List[Tuple[int, int]] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.linalg.norm(matrix[i] - matrix[j]) < distance_threshold:
                adj.append((i, j))
    return adj


def hybrid_node_curvature(adj: List[Tuple[int, int]], matrix: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for each node.

    For each neighbor ``j`` of node ``i`` we compute
        κ(i, j) = 1 - d(i, j) / (d_i + d_j) / 2
    where ``d_i`` is the average distance from ``i`` to its neighbors.
    The node curvature is the mean of κ(i, j) over all neighbors.
    """
    num_nodes = matrix.shape[0]
    curvature = np.zeros(num_nodes, dtype=float)

    # Pre‑compute neighbor lists and average distances
    neighbor_dict = {i: [] for i in range(num_nodes)}
    for i, j in adj:
        neighbor_dict[i].append(j)
        neighbor_dict[j].append(i)

    avg_dist = np.zeros(num_nodes, dtype=float)
    for i, neigh in neighbor_dict.items():
        if neigh:
            dists = [np.linalg.norm(matrix[i] - matrix[j]) for j in neigh]
            avg_dist[i] = sum(dists) / len(dists)

    for i, neigh in neighbor_dict.items():
        if not neigh:
            curvature[i] = 0.0
            continue
        kappa_sum = 0.0
        for j in neigh:
            dij = np.linalg.norm(matrix[i] - matrix[j])
            denom = (avg_dist[i] + avg_dist[j]) / 2.0
            # Guard against division by zero
            if denom == 0.0:
                kappa = 0.0
            else:
                kappa = 1.0 - dij / denom
            kappa_sum += kappa
        curvature[i] = kappa_sum / len(neigh)
    return curvature


def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """
    Compute regret‑weighted probabilities for a list of actions.

    The raw score for an action is its ``expected_value``.  Regret weighting
    applies a softmax with temperature proportional to the action's ``risk``
    (higher risk → flatter distribution).  The resulting vector sums to 1.
    """
    if not actions:
        return np.array([])

    raw = np.array([a.expected_value for a in actions], dtype=float)
    # Temperature = 1 + mean risk (ensures >0)
    temperatures = np.array([1.0 + a.risk for a in actions], dtype=float)
    scaled = raw / temperatures
    # Numerical stability for softmax
    max_scaled = np.max(scaled)
    exp_vals = np.exp(scaled - max_scaled)
    probs = exp_vals / exp_vals.sum()
    return probs


def hybrid_combined_weights(curvature: np.ndarray, regret_probs: np.ndarray) -> np.ndarray:
    """
    Element‑wise product of curvature and regret‑weighted probability.
    The result is normalised to unit sum so it can be interpreted as a
    probability distribution over nodes.
    """
    if curvature.size != regret_probs.size:
        raise ValueError("Curvature and regret probability vectors must have the same length.")
    combined = curvature * regret_probs
    total = combined.sum()
    return combined / total if total > 0 else combined


def hybrid_prune_adj_by_weight(
    adj: List[Tuple[int, int]],
    combined_weights: np.ndarray,
    keep_ratio: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Prune edges based on the average combined weight of their incident nodes.
    Only the top ``keep_ratio`` fraction of edges (by weight) are retained.
    """
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1].")

    # Compute edge scores
    edge_scores = []
    for i, j in adj:
        score = (combined_weights[i] + combined_weights[j]) / 2.0
        edge_scores.append((score, (i, j)))

    # Determine cutoff
    edge_scores.sort(key=lambda x: x[0], reverse=True)
    keep_count = max(1, int(len(edge_scores) * keep_ratio))
    pruned = [pair for _, pair in edge_scores[:keep_count]]
    return pruned


def hybrid_brain_xyz(matrix: np.ndarray, combined_weights: np.ndarray) -> np.ndarray:
    """
    Produce a 3‑D embedding from the high‑dimensional ``matrix``.
    Each row is projected onto its first three components and then
    scaled by the corresponding combined weight.
    """
    if matrix.shape[0] != combined_weights.size:
        raise ValueError("Number of rows in matrix must match size of combined_weights.")

    # Simple linear projection: take first three dimensions (or pad with zeros)
    proj = np.zeros((matrix.shape[0], 3), dtype=float)
    dim = min(3, matrix.shape[1])
    proj[:, :dim] = matrix[:, :dim]

    # Weight each point
    weighted = proj * combined_weights[:, np.newaxis]
    return weighted


# ----------------------------------------------------------------------
# Helper utilities (mirroring small parts of Parent B)
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Return the current UTC timestamp as ISO‑8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create a random high‑dimensional dataset (10 nodes, 5 dimensions)
    np.random.seed(42)
    data = np.random.rand(10, 5)

    # 2. Build adjacency based on Euclidean distance
    adjacency = hybrid_build_adj(data, distance_threshold=0.8)

    # 3. Compute curvature per node
    curv = hybrid_node_curvature(adjacency, data)

    # 4. Create a MathAction for each node with random expected values and risk
    actions = [
        MathAction(id=f"node_{i}", expected_value=random.uniform(0, 10), risk=random.uniform(0, 2))
        for i in range(data.shape[0])
    ]

    # 5. Regret‑weighted probabilities
    regret_probs = calculate_regret_weighted_probabilities(actions)

    # 6. Combine curvature and regret information
    combined = hybrid_combined_weights(curv, regret_probs)

    # 7. Prune the graph using the combined weights
    pruned_adj = hybrid_prune_adj_by_weight(adjacency, combined, keep_ratio=0.6)

    # 8. Produce a final 3‑D embedding
    embedding = hybrid_brain_xyz(data, combined)

    # Simple sanity prints
    print("Timestamp:", utc_now())
    print("Original adjacency size:", len(adjacency))
    print("Pruned adjacency size:", len(pruned_adj))
    print("Curvature vector:", curv.round(3))
    print("Regret probabilities:", regret_probs.round(3))
    print("Combined weights (sum≈1):", combined.round(3), "sum=", combined.sum())
    print("3‑D embedding (first 3 points):\n", embedding[:3])
    sys.exit(0)