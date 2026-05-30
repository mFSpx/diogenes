# DARWIN HAMMER — match 693, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py (gen3)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py (gen3)
# born: 2026-05-29T23:30:37Z

import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional

@dataclass(frozen=True)
class Point:
    """Immutable 2‑D point."""
    x: float
    y: float

@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the hybrid policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0  

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

        self.adj: Dict[str, List[str]] = {n: [] for n in nodes}
        for a, b in edges:
            self.adj[a].append(b)
            self.adj[b].append(a)

        self.root_distances = self._bfs_distances(root)

        self._stats: Dict[Tuple[str, str], List[float]] = {}

    def _validate_graph(self, nodes: Dict[str, Point], edges: List[Tuple[str, str]], root: str):
        if root not in nodes:
            raise ValueError("Root node not in nodes")
        for edge in edges:
            if edge[0] not in nodes or edge[1] not in nodes:
                raise ValueError("Edge node not in nodes")

    def _bfs_distances(self, root: str) -> Dict[str, float]:
        distances = {root: 0.0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for neighbor in self.adj[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1.0
                    queue.append(neighbor)
        return distances

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_fisher_free_energy(mu_q: np.ndarray, sigma_obs: float, fisher_info: float) -> float:
    return -0.5 * sigma_obs**2 * (1 + np.log(2 * np.pi * sigma_obs**2)) - fisher_info

def ssim(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    return (2 * mu_x * mu_y + 2 * sigma_xy) / (mu_x**2 + mu_y**2 + sigma_x**2 + sigma_y**2)

def encode_text(text: str) -> np.ndarray:
    return np.array([ord(c) for c in text])

def run_hybrid_bandit_tree(hybrid_bandit_tree: HybridBanditTree, context_id: str, action_id: str) -> float:
    confidence_term = hybrid_bandit_tree.root_distances[context_id] * hybrid_bandit_tree.path_weight
    return confidence_term

def improved_hybrid_algorithm():
    nodes = {'A': Point(0.0, 0.0), 'B': Point(1.0, 0.0), 'C': Point(1.0, 1.0)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    path_weight = 0.2
    confidence_alpha = 1.0
    hybrid_bandit_tree = HybridBanditTree(nodes, edges, root, path_weight, confidence_alpha)

    theta = 0.5
    center = 0.0
    width = 1.0
    fisher_info = fisher_score(theta, center, width)

    text1 = "Hello"
    text2 = "World"
    vector1 = encode_text(text1)
    vector2 = encode_text(text2)
    similarity = ssim(vector1, vector2)
    sigma_obs = 1.0 / (1 + similarity)

    mu_q = vector1
    hybrid_free_energy = hybrid_fisher_free_energy(mu_q, sigma_obs, fisher_info)

    context_id = 'A'
    action_id = 'B'
    confidence_term = run_hybrid_bandit_tree(hybrid_bandit_tree, context_id, action_id)

    print("Fisher information score:", fisher_info)
    print("Variational free energy:", hybrid_free_energy)
    print("Confidence term:", confidence_term)

if __name__ == "__main__":
    improved_hybrid_algorithm()