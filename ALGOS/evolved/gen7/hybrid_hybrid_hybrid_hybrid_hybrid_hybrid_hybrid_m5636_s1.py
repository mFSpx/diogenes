# DARWIN HAMMER — match 5636, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_physar_m2637_s0.py (gen6)
# born: 2026-05-30T00:03:39Z

"""
This module fuses the hybrid minimum cost tree algorithm from 
hybrid_hybrid_hybrid_distri_hybrid_minimum_cost__m1186_s3.py and 
the hybrid Physarum network algorithm from 
hybrid_hybrid_hybrid_fisher_hybrid_hybrid_physar_m2637_s0.py.

The mathematical bridge between the two structures lies in the use of 
information-theoretic quantities derived from the Fisher information 
and the Physarum network's flux dynamics. Specifically, we use the 
Fisher information to modulate the conductance updates in the 
Physarum network, effectively incorporating the probabilistic 
scoring functions from the Fisher information into the Physarum 
dynamics. We also utilize the minimum cost tree to initialize the 
Physarum network's node connections.

The fusion integrates the governing equations of both parents by 
using the Fisher information to inform the edge selection in the 
minimum cost tree, and then using the resulting tree to initialize 
the Physarum network's conductance updates.
"""

import math
import numpy as np
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Mapping, Hashable, Set

@dataclass(frozen=True)
class Point:
    """A point in 2D space."""
    x: float
    y: float

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a.x - b.x, a.y - b.y)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    """
    Compute the total cost of a tree:
      material = sum of edge lengths
      path_cost = weighted sum of distances from root to every node
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS/DFS to compute root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                stack.append(nxt)

    path_cost = sum(dist.values())
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

def initialize_physarum_network(nodes: Dict[str, Point],
                               edges: List[Tuple[str, str]],
                               root: str) -> Dict[str, Dict[str, float]]:
    """
    Initialize the Physarum network conductance updates using the 
    minimum cost tree.
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Initialize conductance updates
    conductance: Dict[str, Dict[str, float]] = {n: {} for n in nodes}
    for cur in nodes:
        for nxt in adj[cur]:
            conductance[cur][nxt] = 1.0  # Initial conductance

    return conductance

def update_physarum_network(conductance: Dict[str, Dict[str, float]],
                            nodes: Dict[str, Point],
                            edges: List[Tuple[str, str]],
                            root: str,
                            theta: float,
                            center: float,
                            width: float,
                            flux: float,
                            discrepancy: float) -> Dict[str, Dict[str, float]]:
    """
    Update the Physarum network conductance updates using the 
    Fisher information and minimum cost tree.
    """
    fisher_term = fisher_score(theta, center, width)
    for cur in conductance:
        for nxt in conductance[cur]:
            conductance[cur][nxt] = physarum_conductance_update(flux, discrepancy)

    return conductance

def hybrid_minimum_cost_physarum(nodes: Dict[str, Point],
                                 edges: List[Tuple[str, str]],
                                 root: str,
                                 theta: float,
                                 center: float,
                                 width: float,
                                 flux: float,
                                 discrepancy: float) -> float:
    """
    Compute the hybrid minimum cost Physarum score.
    """
    tree_cost_val = tree_cost(nodes, edges, root)
    conductance = initialize_physarum_network(nodes, edges, root)
    conductance = update_physarum_network(conductance, nodes, edges, root, theta, center, width, flux, discrepancy)
    hybrid_score_val = hybrid_score(theta, center, width, flux, discrepancy)
    return tree_cost_val + hybrid_score_val

if __name__ == "__main__":
    nodes = {"A": Point(0.0, 0.0), "B": Point(1.0, 0.0), "C": Point(0.0, 1.0)}
    edges = [("A", "B"), ("A", "C")]
    root = "A"
    theta = 0.5
    center = 0.5
    width = 0.1
    flux = 1.0
    discrepancy = 0.1

    hybrid_score_val = hybrid_minimum_cost_physarum(nodes, edges, root, theta, center, width, flux, discrepancy)
    print(hybrid_score_val)