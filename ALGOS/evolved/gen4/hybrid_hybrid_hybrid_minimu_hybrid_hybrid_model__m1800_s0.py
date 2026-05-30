# DARWIN HAMMER — match 1800, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s0.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s2.py (gen3)
# born: 2026-05-29T23:38:51Z

"""
Module for the hybrid minimum-cost tree Bayesian bandit-router algorithm with ternary tree loss and logistic regression.
This module combines the minimum-cost tree Bayesian update algorithm from 'hybrid_minimum_cost_tree_bayes_update_m6_s2.py'
and the hybrid bandit-router and sketch-RLCT algorithm with ternary tree loss and logistic regression from 'hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s2.py'
by finding a mathematical interface between their structures. The minimum-cost tree Bayesian update algorithm uses
a deterministic cost function with probabilistic weights, while the hybrid bandit-router and sketch-RLCT algorithm
uses a Count-Min sketch to estimate the empirical log-likelihood sum and the effective number of activation patterns.
The mathematical bridge between the two algorithms is the use of ternary tree loss to compute the expected reward
of each action and logistic regression to update the probabilistic weights.
"""

import numpy as np
import math
import random
import sys
import pathlib

# Types
Point = tuple[float, float]
Edge = tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(
    nodes: dict[str, Point],
    edges: list[Edge],
    root: str,
) -> tuple[dict[str, list[str]], dict[Edge, float], dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered a, b) → length
    node_dist : dict mapping node → root‑to‑node distance
    """
    adj = {}
    edge_len = {}
    node_dist = {}

    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
        edge_len[(b, a)] = length(nodes[b], nodes[a])

    # Compute root-to-node distances using BFS
    queue = [(root, 0)]
    visited = set()
    while queue:
        node, dist = queue.pop(0)
        if node not in visited:
            visited.add(node)
            node_dist[node] = dist
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

    return adj, edge_len, node_dist

def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def hybrid_prune(W: np.ndarray, x: np.ndarray, target: np.ndarray = None, reg_lambda: float = 1.0, gamma: float = 0.0, delta: float = 0.001) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = -float(g) / (float(h) + float(reg_lambda))
    split_g = 0.5 * (
        (g ** 2) / (h + reg_lambda)
        - (g ** 2) / (h + reg_lambda)
    )
    if abs(split_g) < delta:
        return 0.0
    else:
        return split_g

def hybrid_operation(nodes: dict[str, Point], edges: list[Edge], root: str, d_in: int, scale: float = 0.01, seed: int = 0) -> float:
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    W = init_ttt(d_in, scale=scale, seed=seed)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    loss = ttt_loss(W, x, target)
    prune_result = hybrid_prune(W, x, target)
    return loss, prune_result

def hybrid_tree_update(nodes: dict[str, Point], edges: list[Edge], root: str, d_in: int, scale: float = 0.01, seed: int = 0) -> float:
    adj, edge_len, node_dist = tree_metrics(nodes, edges, root)
    W = init_ttt(d_in, scale=scale, seed=seed)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    loss = ttt_loss(W, x, target)
    for node in nodes:
        x = np.random.rand(d_in)
        target = np.random.rand(d_in)
        loss += ttt_loss(W, x, target)
    return loss

def hybrid_ttt_update(W: np.ndarray, x: np.ndarray, target: np.ndarray = None, reg_lambda: float = 1.0, gamma: float = 0.0, delta: float = 0.001) -> float:
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = -float(g) / (float(h) + float(reg_lambda))
    split_g = 0.5 * (
        (g ** 2) / (h + reg_lambda)
        - (g ** 2) / (h + reg_lambda)
    )
    if abs(split_g) < delta:
        return 0.0
    else:
        return split_g

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 2),
    }
    edges = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'A'),
    ]
    root = 'A'
    d_in = 10
    scale = 0.01
    seed = 0
    loss, prune_result = hybrid_operation(nodes, edges, root, d_in, scale, seed)
    print(f"Loss: {loss}, Prune Result: {prune_result}")
    loss = hybrid_tree_update(nodes, edges, root, d_in, scale, seed)
    print(f"Tree Loss: {loss}")
    W = init_ttt(d_in, scale=scale, seed=seed)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    result = hybrid_ttt_update(W, x, target)
    print(f"TTT Update Result: {result}")