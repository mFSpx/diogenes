# DARWIN HAMMER — match 3636, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s4.py (gen5)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py (gen2)
# born: 2026-05-29T23:51:08Z

import math
import numpy as np
import random
import sys
import pathlib

GROUPS = ("codex", "groq", "cohere", "local_models")
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Compute the raw (unnormalized) Caputo kernel values for a vector of time indices."""
    if alpha <= 0:
        raise ValueError("Fractional order alpha must be positive.")
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0:
        return 0
    return 1 / (1 + math.exp(-lam * t ** alpha))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def length(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2) -> float:
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

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior / (likelihood * prior + false_positive * (1 - prior))

def hybrid_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2, alpha: float = 0.2, lam: float = 1.0) -> float:
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
    return material + path_weight * sum(dist.values()) * prune_probability(max(dist.values()), lam, alpha) + tree_cost(nodes, edges, root, path_weight)

def hybrid_fisher_score(theta: float, center: float, width: float, eps: float = 1e-12, alpha: float = 0.2, lam: float = 1.0) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity * prune_probability(abs(theta - center), lam, alpha)

def hybrid_tree_cost(nodes: dict, edges: list, root: str, path_weight: float = 0.2, alpha: float = 0.2, lam: float = 1.0) -> float:
    return tree_cost(nodes, edges, root, path_weight) * prune_probability(max(tree_cost(nodes, edges, root, path_weight), 1), lam, alpha)

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (0, 10), 'C': (10, 10)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    print(hybrid_cost(nodes, edges, 'A'))
    print(hybrid_fisher_score(1.0, 0.0, 1.0))
    print(hybrid_tree_cost(nodes, edges, 'A'))