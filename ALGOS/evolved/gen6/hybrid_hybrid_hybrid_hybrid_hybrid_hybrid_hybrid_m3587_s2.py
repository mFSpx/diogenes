# DARWIN HAMMER — match 3587, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2366_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1161_s2.py (gen4)
# born: 2026-05-29T23:50:47Z

"""
Module hybrid_hybrid_fusion_m2366_m1161_s2.py
Fuses parent algorithms:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2366_s2.py (DARWIN HAMMER — match 2366, survivor 2)
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_ternar_m1161_s2.py (DARWIN HAMMER — match 1161, survivor 2)

The mathematical bridge between the two parents lies in their shared use of Fisher score calculations.
The Fisher score is used in parent A to compute the entropy of pheromone probabilities, and in parent B to adjust the cost of a tree.
This module integrates these two applications of the Fisher score into a unified system.

"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

Point = Tuple[float, float]
Edge = Tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def calculate_pheromone_probabilities(surface_key: str, limit: int) -> list:
    pheromones = [random.random() for _ in range(limit)]
    total = sum(pheromones)
    return [p / total for p in pheromones]

def hybrid_fisher_pheromone(surface_key: str, limit: int, center: float, width: float) -> float:
    pheromone_probabilities = calculate_pheromone_probabilities(surface_key, limit)
    fisher_information = [fisher_score(p, center, width) for p in pheromone_probabilities]
    return -sum((p * fi) * math.log(max(p * fi, 1e-12)) for p, fi in zip(pheromone_probabilities, fisher_information) if p * fi > 0)

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_cost(nodes: Dict[str, Point], edges: List[Edge], root: str, path_weight: float = 0.2, 
              fisher_center: float = 0.0, fisher_width: float = 1.0) -> float:
    """Calculate the cost of a tree with fisher score adjustment."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    fisher_scores = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        distance = length(nodes[a], nodes[b])
        fisher_score_value = fisher_score(distance, fisher_center, fisher_width)
        fisher_scores[(a, b)] = fisher_score_value
        material += distance * (1 + fisher_score_value)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b]) * (1 + fisher_scores.get((a, b), fisher_scores.get((b, a), 0)))
    return material

def hybrid_tree_pheromone(nodes: Dict[str, Point], edges: List[Edge], root: str, surface_key: str, limit: int, 
                           fisher_center: float = 0.0, fisher_width: float = 1.0) -> Tuple[float, float]:
    tree_material = tree_cost(nodes, edges, root, fisher_center=fisher_center, fisher_width=fisher_width)
    pheromone_entropy = hybrid_fisher_pheromone(surface_key, limit, fisher_center, fisher_width)
    return tree_material, pheromone_entropy

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1), 'D': (0, 1)}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    surface_key = "test_surface"
    limit = 10
    fisher_center = 0.5
    fisher_width = 1.0
    tree_material, pheromone_entropy = hybrid_tree_pheromone(nodes, edges, 'A', surface_key, limit, fisher_center, fisher_width)
    print(f"Tree material: {tree_material}, Pheromone entropy: {pheromone_entropy}")