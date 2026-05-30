# DARWIN HAMMER — match 35, survivor 4
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""
This module combines the Caputo fractional derivative from caputo_fractional.py and the minimum-cost tree scoring from minimum_cost_tree.py.
The mathematical bridge between these two structures is the use of the Caputo fractional derivative to model the decay of the tree's edge weights over time.
This allows for a more nuanced and dynamic representation of the tree's structure, taking into account the algebraic decay of the edge weights.
"""

import numpy as np
import math
import random
import sys
import pathlib

__all__ = [
    "gamma_lanczos",
    "caputo_derivative",
    "fractional_decay",
    "fractional_ssm_step",
    "length",
    "tree_cost",
    "fractional_tree_cost",
    "dynamic_tree_cost",
]

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
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x


def caputo_derivative(f, alpha, t):
    """Caputo fractional derivative of f(t) with order alpha."""
    dt = 0.01
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integral = np.trapz(f_tau / (t - tau) ** alpha, tau)
    return integral / gamma_lanczos(1 - alpha)


def fractional_decay(alpha, t):
    """Fractional decay kernel."""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def fractional_ssm_step(alpha, A, B, x_t, h_prev):
    """Fractional SSM step."""
    w_k = []
    for k in range(len(h_prev)):
        w_k.append(fractional_decay(alpha, len(h_prev) - k))
    w_k = np.array(w_k) / np.sum(w_k)
    h_t = np.sum([w_k[k] * (A * h_prev[k] + B * x_t) for k in range(len(h_prev))])
    return h_t


def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(nodes, edges, root, path_weight=0.2):
    """Minimum-cost tree scoring."""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
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


def fractional_tree_cost(alpha, nodes, edges, root, path_weight=0.2):
    """Fractional tree cost."""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    fractional_decay_kernel = [fractional_decay(alpha, dist[n]) for n in nodes]
    return material + path_weight * sum([d * k for d, k in zip(dist.values(), fractional_decay_kernel)])


def dynamic_tree_cost(alpha, nodes, edges, root, path_weight=0.2, t=1.0):
    """Dynamic tree cost."""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    dynamic_decay_kernel = [caputo_derivative(lambda x: x, alpha, dist[n]) for n in nodes]
    return material + path_weight * sum([d * k for d, k in zip(dist.values(), dynamic_decay_kernel)])


if __name__ == "__main__":
    nodes = {"A": (0, 0), "B": (1, 0), "C": (1, 1)}
    edges = [("A", "B"), ("B", "C")]
    root = "A"
    alpha = 0.5
    print(fractional_tree_cost(alpha, nodes, edges, root))
    print(dynamic_tree_cost(alpha, nodes, edges, root))