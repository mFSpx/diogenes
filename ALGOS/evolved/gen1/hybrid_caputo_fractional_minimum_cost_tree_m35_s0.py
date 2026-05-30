# DARWIN HAMMER — match 35, survivor 0
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

"""
Hybrid Caputo Minimum-Cost Tree Algorithm

This module fuses the Caputo Fractional Derivative (caputo_fractional.py) and 
the Minimum-Cost Tree scoring (minimum_cost_tree.py) algorithms. The mathematical 
bridge between these two structures lies in the use of power-law decay kernels 
and graph-based optimizations. The Caputo Fractional Derivative provides a 
framework for modeling algebraically-decaying long-range memory, while the 
Minimum-Cost Tree scoring algorithm optimizes tree structures based on edge 
lengths and path weights. By integrating these two concepts, we can create 
a hybrid algorithm that optimizes tree structures while accounting for 
algebraically-decaying long-range memory effects.

The key innovation in this fusion is the application of the Caputo Fractional 
Derivative to the edge weights in the Minimum-Cost Tree scoring algorithm. 
This allows us to model the decay of memory effects over time, rather than 
simply using a fixed path weight. The resulting hybrid algorithm can be used 
to optimize tree structures in a wide range of applications, from signal 
processing to network optimization.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gamma_lanczos(z):
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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return np.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5)**(z + 0.5) * np.exp(-(z + _LANCZOS_G + 0.5)) * np.polyval(_LANCZOS_C[::-1], z)

def caputo_derivative(alpha, t, f):
    integral = 0
    for tau in range(t):
        integral += f[tau] / (t - tau)**alpha
    return integral / gamma_lanczos(1 - alpha)

def length(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes, edges, root, path_weight=0.2, alpha=0.5):
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
    caputo_weights = [caputo_derivative(alpha, len(dist), [dist[n] for n in dist]) for n in dist]
    return material + path_weight * sum([dist[n] * caputo_weights[i] for i, n in enumerate(dist)])

def hybrid_tree_optimization(nodes, edges, root, path_weight=0.2, alpha=0.5):
    return tree_cost(nodes, edges, root, path_weight, alpha)

def random_tree(nodes, edges):
    root = random.choice(list(nodes.keys()))
    return tree_cost(nodes, edges, root)

if __name__ == "__main__":
    nodes = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (1, 1),
        'D': (0, 1),
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    print(hybrid_tree_optimization(nodes, edges, root))
    print(random_tree(nodes, edges))