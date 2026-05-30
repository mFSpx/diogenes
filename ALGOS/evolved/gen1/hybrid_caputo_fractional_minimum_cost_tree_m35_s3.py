# DARWIN HAMMER — match 35, survivor 3
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""
This module fuses the Caputo Fractional Derivative algorithm and the Minimum-Cost Tree scoring algorithm.
The mathematical bridge between the two is found in the concept of weighted decay kernels and path costs.
The Caputo Fractional Derivative algorithm uses a power-law kernel to model algebraically-decaying long-range memory,
while the Minimum-Cost Tree scoring algorithm uses a path cost function to evaluate the trade-offs between length and path.
By combining these two concepts, we can create a hybrid algorithm that uses the Caputo Fractional Derivative to model the decay of path costs over time.
"""

import numpy as np
import math
import random
import sys
import pathlib

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
    """Lanczos approximation of Gamma(z) for z > 0"""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = np.array([_LANCZOS_C[0]])
    for i in range(1, _LANCZOS_G + 1):
        x = np.append(x, _LANCZOS_C[i] / (z + i))
    return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.sum(x)


def caputo_derivative(f, t, alpha):
    """Caputo Fractional Derivative"""
    return 1 / gamma_lanczos(1 - alpha) * np.sum((f[1:] - f[:-1]) / (t[1:] - t[:-1]) ** alpha)


def fractional_decay(t, alpha):
    """Fractional decay kernel"""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def minimum_cost_tree(nodes, edges, root, path_weight=0.2):
    """Minimum-cost tree scoring"""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
                stack.append(b)
    return material + path_weight * sum(dist.values())


def hybrid_tree_cost(nodes, edges, root, alpha, path_weight=0.2):
    """Hybrid tree cost function"""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) * fractional_decay(np.abs(nodes[a][0] - nodes[b][0]), alpha)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) * fractional_decay(np.abs(nodes[a][0] - nodes[b][0]), alpha)
                stack.append(b)
    return material + path_weight * sum(dist.values())


def hybrid_tree_cost_with_caputo(nodes, edges, root, alpha, path_weight=0.2):
    """Hybrid tree cost function with Caputo derivative"""
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) * caputo_derivative(np.array([math.hypot(nodes[c][0] - nodes[d][0], nodes[c][1] - nodes[d][1]) for c, d in edges]), np.array([math.hypot(nodes[c][0] - nodes[d][0], nodes[c][1] - nodes[d][1]) for c, d in edges]), alpha)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1]) * caputo_derivative(np.array([math.hypot(nodes[c][0] - nodes[d][0], nodes[c][1] - nodes[d][1]) for c, d in edges]), np.array([math.hypot(nodes[c][0] - nodes[d][0], nodes[c][1] - nodes[d][1]) for c, d in edges]), alpha)
                stack.append(b)
    return material + path_weight * sum(dist.values())


if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    alpha = 0.5
    print(hybrid_tree_cost(nodes, edges, root, alpha))
    print(hybrid_tree_cost_with_caputo(nodes, edges, root, alpha))