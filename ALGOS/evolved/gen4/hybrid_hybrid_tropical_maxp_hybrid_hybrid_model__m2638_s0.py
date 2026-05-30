# DARWIN HAMMER — match 2638, survivor 0
# gen: 4
# parent_a: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s0.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1.py (gen3)
# born: 2026-05-29T23:43:17Z

"""
This module integrates the hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s0 and 
hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s1 algorithms into a single 
hybrid system. The bridge between the two structures is the concept of applying 
tropical max-plus algebra to the decision-making process of the hybrid model, 
specifically in the computation of the optimal leaf weights and split gains.

The mathematical interface is found by representing the tropical max-plus operations 
as a non-linear transformation of the traditional arithmetic operations, and then 
applying this transformation to the gradients and Hessians in the hybrid model.

The resulting hybrid system combines the strengths of both parent algorithms: 
the ability to efficiently compute minimum-cost trees using Bayesian updates, 
and the flexibility to adapt to changing data distributions using optimal 
leaf weights and split gains.

"""

import numpy as np
import math
import random
import sys
import pathlib

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def length(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(nodes, edges, root):
    adj = {n: [] for n in nodes}
    edge_len = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)
    return adj, edge_len, dist

def bayes_marginal(prior, likelihood, false_positive):
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior, likelihood, marginal):
    return likelihood * prior / marginal

def hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive):
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    updated = bayes_update(prior, likelihood, marginal)
    tropical_dist = [t_add(dist[node], updated) for node in dist]
    return adj, edge_len, dict(zip(dist.keys(), tropical_dist))

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight_tropical(
    gradient_sum: float, 
    hessian_sum: float, 
    reg_lambda: float = 1.0
) -> float:
    return -t_add(gradient_sum, hessian_sum) / (hessian_sum + reg_lambda)

def split_gain_tropical(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0
) -> float:
    return 0.5 * (
        (t_add(left_gradient ** 2, left_hessian)) / (left_hessian + reg_lambda)
        + (t_add(right_gradient ** 2, right_hessian)) / (right_hessian + reg_lambda)
        - (t_add(left_gradient, right_gradient) ** 2) / (left_hessian + right_hessian + reg_lambda)
    ) - gamma

def hybrid_prune_tropical(
    W, 
    x, 
    target=None, 
    reg_lambda=1.0, 
    gamma=0.0, 
    learning_rate=0.1
):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    margin = -residual
    g, h = binary_logistic_grad_hess(1.0, margin)
    optimal_weight = optimal_leaf_weight_tropical(g, h, reg_lambda)
    W_update = W - learning_rate * np.outer(g, x)
    split_g = split_gain_tropical(g, h, g, h, reg_lambda=reg_lambda, gamma=gamma)
    return W_update, split_g

def main():
    nodes = [(0, 0), (1, 0), (1, 1)]
    edges = [(0, 1), (1, 2)]
    root = 0
    prior = 0.5
    likelihood = 0.8
    false_positive = 0.2

    adj, edge_len, dist = hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive)

    d_in = 10
    d_out = 10
    scale = 0.01
    seed = 0
    W = np.random.standard_normal((d_out, d_in)) * scale
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    reg_lambda = 1.0
    gamma = 0.0
    learning_rate = 0.1

    W_updated, split_g = hybrid_prune_tropical(W, x, target, reg_lambda, gamma, learning_rate)

if __name__ == "__main__":
    main()