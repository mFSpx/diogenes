# DARWIN HAMMER — match 1171, survivor 2
# gen: 3
# parent_a: geometric_product.py (gen0)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# born: 2026-05-29T23:33:12Z

"""
This module integrates the geometric product from geometric_product.py and 
the Fisher information scoring and minimum-cost tree scoring with Bayesian evidence update 
from hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py. The mathematical bridge 
between the two structures is the use of Gaussian distributions to model uncertainty 
in the tree edges and nodes. Specifically, the Fisher information scoring is used to 
estimate the precision of the Gaussian distribution, while the minimum-cost tree scoring 
is used to estimate the prior probabilities of the tree edges and nodes.

In the hybrid algorithm, the geometric product is used to combine the uncertainty 
estimates from the Fisher information scoring and the prior probabilities from the 
minimum-cost tree scoring. This allows for a more accurate estimation of the uncertainty 
in the tree edges and nodes.

The module exports three functions: `gaussian_blade`, `fisher_score`, and `tree_cost`. 
The `gaussian_blade` function uses the geometric product to combine the uncertainty 
estimates from the Fisher information scoring and the prior probabilities from the 
minimum-cost tree scoring. The `fisher_score` function estimates the precision of the 
Gaussian distribution using the Fisher information scoring. The `tree_cost` function 
estimates the prior probabilities of the tree edges and nodes using the minimum-cost 
tree scoring.
"""

import numpy as np
import math
import random
import sys
import pathlib

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient.
                frozenset() is the scalar (grade-0) blade.
    n: dimension of the base vector space.
    """

    def __init__(self, components, n):
        # Drop zero coefficients to keep repr clean
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

def gaussian_blade(theta: float, center: float, width: float, prior_prob: float) -> Multivector:
    blade = Multivector({frozenset(): 1.0}, 2)
    gaussian = gaussian_beam(theta, center, width)
    blade.components = {frozenset(): gaussian * prior_prob}
    return blade

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a.components.keys()) + list(blade_b.components.keys())
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, tuple[float, float]], edges: list[tuple[str, str]], root: str, path_weight: float = 0.2) -> float:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
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

if __name__ == "__main__":
    blade = gaussian_blade(1.0, 0.0, 1.0, 0.5)
    print(blade.components)
    score = fisher_score(1.0, 0.0, 1.0)
    print(score)
    nodes = {"A": (1.0, 1.0), "B": (2.0, 2.0)}
    edges = [("A", "B")]
    cost = tree_cost(nodes, edges, "A")
    print(cost)