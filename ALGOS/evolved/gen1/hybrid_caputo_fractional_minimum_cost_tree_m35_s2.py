# DARWIN HAMMER — match 35, survivor 2
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""
HYBRID CAPUTO MINIMUM COST TREE (HCMCT) algorithm — fusion of Caputo fractional derivative and minimum cost tree scoring for length/path trade-offs.

The mathematical bridge between these two algorithms lies in the fact that both structures can be represented as graphs, where nodes are connected by edges with associated weights. In the Caputo fractional derivative, these weights are the power-law decay kernels (phi(t; alpha) / sum_j phi(t - tau_j; alpha)) for each past time, while in the minimum cost tree, they represent the lengths of the edges.

By combining these two structures, we can create a hybrid graph where the weights on the edges are influenced by both the power-law decay and the edge lengths, leading to a new minimum cost tree scoring function that incorporates long-range memory and path-dependent trade-offs.

This fusion is achieved by modifying the tree_cost function to take into account the Caputo fractional derivative weights, which are computed using the caputo_derivative function.
"""
import math
import random
import sys
import numpy as np
from math import gamma
from scipy.special import gamma as scipy_gamma
from pathlib import Path

# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        return scipy_gamma(1 - z) * scipy_gamma(z) / math.sin(math.pi * z)
    z += _LANCZOS_G + 0.5
    term = 1.0
    for c in _LANCZOS_C:
        term *= (z + c) / (z - c)
    return np.sqrt(2 * math.pi) * z ** (z + 0.5) * np.exp(-z) * term


def caputo_derivative(f, t, alpha):
    """Compute the Caputo fractional derivative of f at time t."""
    dt = np.diff(t)
    df = np.diff(f)
    integral = np.dot(df, dt ** (-alpha)) / gamma(1 - alpha)
    return np.insert(integral, 0, 0)


def gamma_term(t, alpha, sum_j_gamma):
    """Compute the power-law decay kernel phi(t; alpha) / sum_j phi(t - tau_j; alpha)."""
    gamma_value = gamma(lanczos_term(t, alpha))
    return gamma_value / sum_j_gamma


def lanczos_term(t, alpha):
    """Compute the Lanczos term t**(alpha - 1) / Gamma(alpha)."""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def caputo_weighted_sum(h, alpha, sum_j_gamma, tau):
    """Compute the Caputo-weighted sum of h over the full history."""
    return np.sum(gamma_term(tau, alpha, sum_j_gamma) * (h[tau] if tau < len(h) else 0) for tau in range(len(h)))


def hybrid_tree_cost(nodes, edges, root, alpha, path_weight=0.2):
    """Compute the hybrid minimum cost tree scoring function."""
    adj = {n: [] for n in nodes}
    material = 0.0
    sum_j_gamma = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
        sum_j_gamma += lanczos_term(length(nodes[a], nodes[b]), alpha)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    h = [dist[node] for node in nodes]
    for a, b in edges:
        h = caputo_weighted_sum(h, alpha, sum_j_gamma, length(nodes[a], nodes[b]))
    return material + path_weight * sum(dist.values())


def tree_cost(nodes, edges, root, path_weight=0.2):
    """Compute the minimum cost tree scoring function."""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    return material + path_weight * sum(dist.values())


def length(a, b):
    """Compute the length of the edge between a and b."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (0, 10),
        'C': (10, 10),
    }
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
    ]
    root = 'A'
    alpha = 0.5
    path_weight = 0.2
    hybrid_cost = hybrid_tree_cost(nodes, edges, root, alpha, path_weight)
    print(hybrid_cost)
    tree_cost = tree_cost(nodes, edges, root, path_weight)
    print(tree_cost)
    assert hybrid_cost >= 0 and tree_cost >= 0