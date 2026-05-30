# DARWIN HAMMER — match 5636, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_physar_m2637_s0.py (gen6)
# born: 2026-05-30T00:03:39Z

"""
This module integrates the tree cost calculation from the hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s3.py 
and the Fisher information scoring with Physarum network scoring from hybrid_hybrid_hybrid_fisher_hybrid_hybrid_physar_m2637_s0.py.
The mathematical bridge between the two structures lies in the information-theoretic quantities that can be 
derived from both the Fisher information and the Physarum network's flux dynamics, which can be used to inform 
the tree cost calculation by modulating the weight of the path costs based on the Fisher information score.
"""

import math
import numpy as np
import random
import sys
import pathlib

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: dict[str, Point], 
              edges: list[tuple[str, str]], 
              root: str, 
              path_weight: float = 0.2, 
              fisher_center: float = 0.0, 
              fisher_width: float = 1.0) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node, 
        modulated by the Fisher information score
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS/DFS to compute root‑to‑node distances
    dist: dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    # modulate path costs with Fisher information score
    fisher_scores = {n: fisher_score(dist[n], fisher_center, fisher_width) for n in nodes}
    path_cost = sum(dist[n] * fisher_scores[n] for n in nodes)
    return material + path_weight * path_cost

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def physarum_conductance_update(flux: float, discrepancy: float) -> float:
    return flux + discrepancy * np.exp(-flux)

def hybrid_score(theta: float, center: float, width: float, flux: float, discrepancy: float) -> float:
    fisher_term = fisher_score(theta, center, width)
    physarum_term = physarum_conductance_update(flux, discrepancy)
    return fisher_term + physarum_term

def sample_tree() -> (dict[str, Point], list[tuple[str, str]], str):
    nodes = {f"n{i}": Point(random.random(), random.random()) for i in range(5)}
    edges = [(f"n{i}", f"n{j}") for i in range(5) for j in range(i+1, 5)]
    root = "n0"
    return nodes, edges, root

if __name__ == "__main__":
    nodes, edges, root = sample_tree()
    cost = tree_cost(nodes, edges, root)
    print(f"Tree cost: {cost}")

    theta = 0.5
    center = 0.0
    width = 1.0
    flux = 0.1
    discrepancy = 0.01
    score = hybrid_score(theta, center, width, flux, discrepancy)
    print(f"Hybrid score: {score}")