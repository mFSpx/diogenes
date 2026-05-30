# DARWIN HAMMER — match 1171, survivor 0
# gen: 3
# parent_a: geometric_product.py (gen0)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py (gen2)
# born: 2026-05-29T23:33:12Z

"""
This module fuses the Clifford geometric product from geometric_product.py 
and the hybrid Fisher information and minimum-cost tree scoring from hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s1.py.
The mathematical bridge between the two structures is the use of multivectors to model uncertainty in the tree edges and nodes.

The Clifford algebra provides a way to unify different mathematical objects, 
such as scalars, vectors, and multivectors, under a single algebraic structure. 
In this hybrid algorithm, we use multivectors to represent the uncertainty in the tree edges and nodes, 
modeled using Gaussian distributions.

The Fisher information scoring is used to compute the uncertainty in the tree edges and nodes, 
while the minimum-cost tree scoring is used to compute the material cost of the tree.

The resulting hybrid algorithm provides a more comprehensive and accurate model 
for computing the uncertainty and material cost of complex systems.
"""

import math
import numpy as np
from geometric_product import Multivector, _multiply_blades, _blade_sign

Point = tuple[float, float]
Edge = tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> Multivector:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    components = {frozenset(): derivative * derivative / intensity}
    return Multivector(components, 1)

def length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: dict[str, Point], edges: list[Edge], root: str, path_weight: float = 0.2) -> Multivector:
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = Multivector({frozenset(): 0.0}, 1)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_cost = length(nodes[a], nodes[b])
        edge_components = {frozenset(): edge_cost}
        edge_multivector = Multivector(edge_components, 1)
        material = material + edge_multivector
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    path_cost = Multivector({frozenset(): path_weight * sum(dist.values())}, 1)
    return material + path_cost

def hybrid_score(nodes: dict[str, Point], edges: list[Edge], root: str, theta: float, center: float, width: float) -> Multivector:
    fisher_multivector = fisher_score(theta, center, width)
    tree_multivector = tree_cost(nodes, edges, root)
    result_blade, sign = _multiply_blades(fisher_multivector.components, tree_multivector.components)
    result_components = {result_blade: sign * fisher_multivector.components[frozenset()] * tree_multivector.components[frozenset()]}
    return Multivector(result_components, 1)

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    theta = 0.5
    center = 0.5
    width = 0.1
    hybrid_multivector = hybrid_score(nodes, edges, root, theta, center, width)
    print(hybrid_multivector.components)