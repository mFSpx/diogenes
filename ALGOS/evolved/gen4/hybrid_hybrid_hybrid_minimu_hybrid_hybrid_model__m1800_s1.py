# DARWIN HAMMER — match 1800, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s0.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s2.py (gen3)
# born: 2026-05-29T23:38:51Z

"""Hybrid Minimum‑Cost Tree Bayesian‑Gradient Algorithm

This module fuses two parent algorithms:

* **Parent A** – deterministic tree utilities (edge lengths, root‑to‑node distances)
* **Parent B** – linear transform loss, gradient, and XGBoost‑style split gain

**Mathematical bridge** – the bridge is a *tree‑regularized loss* that adds a
quadratic penalty on the Euclidean distance between weight vectors attached to
adjacent nodes of the tree.  The penalty uses the same edge‑length metric from
Parent A, while the base loss and its gradient come from Parent B.  The combined
gradient is then used to update the weight matrix in a single hybrid step.

The resulting system can be interpreted as a Bayesian‑inspired minimum‑cost
tree where the “cost” of an edge is the squared difference of the model’s
parameters, weighted by the geometric edge length.  This integrates the
probabilistic gradient machinery of Parent B with the deterministic topology
of Parent A."""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A – deterministic tree utilities
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
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj: Dict[str, List[str]] = defaultdict(list)
    edge_len: Dict[Edge, float] = {}
    node_dist: Dict[str, float] = {}

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        l = length(nodes[a], nodes[b])
        edge_len[(a, b)] = l
        edge_len[(b, a)] = l

    # BFS to compute distances from root
    queue = deque([(root, 0.0)])
    visited = set()
    while queue:
        node, dist = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        node_dist[node] = dist
        for neighbor in adj[node]:
            if neighbor not in visited:
                queue.append((neighbor, dist + edge_len[(node, neighbor)]))

    return dict(adj), edge_len, node_dist

# ----------------------------------------------------------------------
# Parent B – linear transform utilities (TTT = “thin‑to‑thin” matrix)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> float:
    """Mean‑squared‑error loss for a linear map W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Gradient of the MSE loss w.r.t. W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2.0 * np.outer(residual, x)

# ----------------------------------------------------------------------
# Hybrid components – the mathematical bridge
# ----------------------------------------------------------------------
def tree_weight_regularizer(
    W: np.ndarray,
    node_list: List[str],
    node_idx: Dict[str, int],
    edge_len: Dict[Edge, float],
    lambda_reg: float = 0.1,
) -> float:
    """
    Quadratic regularizer that penalises differences between weight vectors of
    adjacent nodes, scaled by the geometric edge length.

    Σ_{(i,j)∈E} λ·ℓ_{ij}·‖W_{:,i} – W_{:,j}‖²
    """
    reg = 0.0
    for (a, b), l in edge_len.items():
        # Count each undirected edge once
        if a > b:
            continue
        i, j = node_idx[a], node_idx[b]
        diff = W[:, i] - W[:, j]
        reg += lambda_reg * l * (diff @ diff)
    return reg

def hybrid_total_loss(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    node_list: List[str],
    node_idx: Dict[str, int],
    edge_len: Dict[Edge, float],
    lambda_reg: float = 0.1,
) -> float:
    """
    Combined loss = MSE loss (Parent B) + tree‑regularizer (Parent A).
    """
    base = ttt_loss(W, x, target)
    reg = tree_weight_regularizer(W, node_list, node_idx, edge_len, lambda_reg)
    return base + reg

def hybrid_gradient(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    node_list: List[str],
    node_idx: Dict[str, int],
    edge_len: Dict[Edge, float],
    lambda_reg: float = 0.1,
) -> np.ndarray:
    """
    Gradient of the hybrid total loss.

    ∇_W L = ∇_W MSE  +  λ·Σ_{(i,j)} ℓ_{ij}·2·(e_i – e_j)·(W_{:,i} – W_{:,j})ᵀ
    where e_i is a basis vector selecting column i.
    """
    grad = ttt_grad(W, x, target)  # shape (d_out, d_in)

    # Add regularizer gradient
    d_out, d_in = W.shape
    reg_grad = np.zeros_like(W)
    for (a, b), l in edge_len.items():
        if a > b:
            continue
        i, j = node_idx[a], node_idx[j] = node_idx[b]
        diff = W[:, i] - W[:, j]          # (d_out,)
        contribution = 2.0 * lambda_reg * l * diff[:, None]  # (d_out,1)
        reg_grad[:, i] += contribution[:, 0]
        reg_grad[:, j] -= contribution[:, 0]

    return grad + reg_grad

def hybrid_update_step(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    node_list: List[str],
    node_idx: Dict[str, int],
    edge_len: Dict[Edge, float],
    lr: float = 0.01,
    lambda_reg: float = 0.1,
) -> np.ndarray:
    """
    Perform a single gradient‑descent step on the hybrid loss.
    Returns the updated weight matrix.
    """
    grad = hybrid_gradient(W, x, target, node_list, node_idx, edge_len, lambda_reg)
    return W - lr * grad

# ----------------------------------------------------------------------
# Utility to build node index mapping from a node list
# ----------------------------------------------------------------------
def build_node_index(nodes: Dict[str, Point]) -> Tuple[List[str], Dict[str, int]]:
    """Return (ordered_node_list, mapping node → column index)."""
    ordered = list(nodes.keys())
    idx = {name: i for i, name in enumerate(ordered)}
    return ordered, idx

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 0.866),  # equilateral triangle
    }
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    root = "A"

    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    node_list, node_idx = build_node_index(nodes)

    # Initialise a weight matrix whose columns correspond to nodes
    d_in = 4
    d_out = 4
    W = init_ttt(d_in, d_out, scale=0.05, seed=42)

    # Random input / target
    x = np.random.rand(d_in)
    target = np.random.rand(d_out)

    # Compute initial loss
    loss_before = hybrid_total_loss(
        W, x, target, node_list, node_idx, edge_len, lambda_reg=0.2
    )
    print(f"Initial hybrid loss: {loss_before:.6f}")

    # One update step
    W_new = hybrid_update_step(
        W, x, target, node_list, node_idx, edge_len, lr=0.01, lambda_reg=0.2
    )

    # Loss after update
    loss_after = hybrid_total_loss(
        W_new, x, target, node_list, node_idx, edge_len, lambda_reg=0.2
    )
    print(f"Hybrid loss after one step: {loss_after:.6f}")

    # Verify that loss decreased (not guaranteed for arbitrary lr, but should for small lr)
    if loss_after < loss_before:
        print("Loss decreased – hybrid step successful.")
    else:
        print("Loss did not decrease – consider reducing the learning rate.")