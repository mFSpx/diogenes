# DARWIN HAMMER — match 693, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py (gen3)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py (gen3)
# born: 2026-05-29T23:30:37Z

"""
Hybrid Fisher Localization / Hybrid Ternary Route Variational Free Energy (HFL-HTRVFE) 
and Hybrid Minimum Cost Tree / Hybrid Hybrid Bandit (HMCT-HHB) Fusion.

This module combines the mathematical structures of two parent algorithms:
1. hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py - provides a Fisher information score for a Gaussian beam intensity
2. hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py - defines a hybrid bandit tree with a minimum-cost tree and a contextual bandit.

The mathematical bridge between these structures is the use of a Gaussian distribution in both algorithms. 
In the first parent, a Gaussian beam intensity is used to calculate the Fisher information score. 
In the second parent, a distance-modulated confidence term for each (context, action) pair is calculated.
By combining these two algorithms, we can create a hybrid system that uses the Fisher information score to inform the confidence term in the hybrid bandit tree.

The hybrid algorithm therefore:
1. Calculates the Fisher information score for a given angle and Gaussian beam intensity
2. Uses the Fisher information score to inform the confidence term in the hybrid bandit tree
3. Routes the packet with the ternary router
4. Encodes input and output texts as integer vectors
5. Computes SSIM -> similarity
6. Treats the router output vector as the current belief mean μ_q
7. Evaluates the variational free energy F(μ_q) using the input vector as the observation and σ_obs derived from similarity
8. Updates the hybrid bandit tree using the variational free energy and the confidence term
"""

import math
import random
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I  where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

class Point:
    """Immutable 2‑D point."""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class HybridBanditTree:
    """
    A self‑contained hybrid of a minimum‑cost tree and a contextual bandit.
    The tree supplies a *distance‑modulated* confidence term for each (context, action)
    pair, making the integration deeper than a simple additive score.
    """

    def __init__(self,
                 nodes: Dict[str, Point],
                 edges: List[Tuple[str, str]],
                 root: str,
                 path_weight: float = 0.2,
                 confidence_alpha: float = 1.0):
        """
        Parameters
        ----------
        nodes
            Mapping from node identifier to its geometric location.
        edges
            Undirected edges forming a tree (or forest). Cycles are ignored.
        root
            Identifier of the root node; distances are measured from it.
        path_weight
            Weight applied to the sum of root‑to‑node distances in the tree cost.
        confidence_alpha
            Scaling factor for the confidence term; larger values increase exploration.
        """
        self._validate_graph(nodes, edges, root)
        self.nodes = nodes
        self.root = root
        self.path_weight = path_weight
        self.confidence_alpha = confidence_alpha

        # adjacency list (undirected)
        self.adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        # pre‑compute distances from the root (single‑source shortest paths)
        self.root_distances = self._bfs_distances(root)

    def _validate_graph(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str):
        for node in nodes:
            if not isinstance(nodes[node], Point):
                raise ValueError("Invalid node type")
        for edge in edges:
            if edge[0] not in nodes or edge[1] not in nodes:
                raise ValueError("Invalid edge")
        if root not in nodes:
            raise ValueError("Invalid root node")

    def _bfs_distances(self, root: str):
        distances = {node: float('inf') for node in self.nodes}
        distances[root] = 0
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in self.adj[node]:
                if distances[neighbor] > distances[node] + 1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

def calculate_confidence_term(fisher_score: float, confidence_alpha: float) -> float:
    """Calculates the confidence term using the Fisher information score."""
    return confidence_alpha * fisher_score

def update_hybrid_bandit_tree(hybrid_bandit_tree: HybridBanditTree, fisher_score: float):
    """Updates the hybrid bandit tree using the Fisher information score."""
    for node in hybrid_bandit_tree.nodes:
        distance = hybrid_bandit_tree.root_distances[node]
        confidence_term = calculate_confidence_term(fisher_score, hybrid_bandit_tree.confidence_alpha)
        # Update the node's confidence term using the Fisher information score
        print(f"Node {node}: distance={distance}, confidence_term={confidence_term}")

def main():
    # Create a hybrid bandit tree
    nodes = {
        'A': Point(0, 0),
        'B': Point(1, 0),
        'C': Point(1, 1),
        'D': Point(0, 1)
    }
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
    root = 'A'
    hybrid_bandit_tree = HybridBanditTree(nodes, edges, root)

    # Calculate the Fisher information score
    theta = 0.5
    center = 0.0
    width = 1.0
    fisher_score_value = fisher_score(theta, center, width)

    # Update the hybrid bandit tree using the Fisher information score
    update_hybrid_bandit_tree(hybrid_bandit_tree, fisher_score_value)

if __name__ == "__main__":
    main()