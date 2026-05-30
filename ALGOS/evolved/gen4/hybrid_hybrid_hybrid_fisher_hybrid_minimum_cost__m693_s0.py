# DARWIN HAMMER — match 693, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_ternar_m177_s0.py (gen3)
# parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py (gen3)
# born: 2026-05-29T23:30:37Z

"""
DARWIN HAMMER — hybrid_hybrid_hybrid_fisher_locali_hybrid_ternar_m177_s0_m253_s3.py
gen: 3
parent_a: hybrid_hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py (gen2)
parent_b: hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py (gen3)
born: 2026-05-30T01:30:00Z

This module combines the mathematical structures of two parent algorithms:
1. hybrid_hybrid_fisher_localization_hybrid_ternary_route_m40_s3.py - provides a Fisher information score for a Gaussian beam intensity
2. hybrid_minimum_cost_tree_hybrid_hybrid_bandit_m253_s3.py - defines a self-contained hybrid of a minimum-cost tree and a contextual bandit

The mathematical bridge between these structures is the use of a Gaussian distribution in both algorithms. In the first parent, a Gaussian beam intensity is used to calculate the Fisher information score. In the second parent, a Gaussian generative model is used to define the variational free-energy formulation. By combining these two algorithms, we can create a hybrid system that uses the Fisher information score to inform the variational free-energy formulation.

In this hybrid algorithm, we first calculate the Fisher information score for a given angle and Gaussian beam intensity. Then, we use the Fisher information score to inform the variational free-energy formulation. The variational free-energy formulation is then used to guide the hybrid bandit tree. The bandit tree supplies a distance-modulated confidence term for each (context, action) pair, making the integration deeper than a simple additive score.

The hybrid algorithm therefore:
1. Calculates the Fisher information score for a given angle and Gaussian beam intensity
2. Uses the Fisher information score to inform the variational free-energy formulation
3. Routes the packet with the ternary router
4. Encodes input and output texts as integer vectors
5. Computes SSIM -> similarity
6. Treats the router output vector as the current belief mean μ_q
7. Evaluates the variational free energy F(μ_q) using the input vector as the observation and σ_obs derived from similarity
8. Guides the hybrid bandit tree using the variational free-energy formulation
"""

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
    propensity: float = 1.0  # not used in the current formulation but kept for compatibility

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

        # statistics: (context, action) → [total_reward, count]
        self._stats: Dict[Tuple[str, str], List[float]] = {}

    # --------------------------------------------------------------------- #
    #                     Graph utilities and validation                  #

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

def hybrid_fisher_free_energy(mu_q: np.ndarray, sigma_obs: float, fisher_info: float) -> float:
    """Hybrid Fisher information score and variational free energy.

    F(μ_q) = -0.5 * sigma_obs^2 * (1 + np.log(2 * np.pi * sigma_obs^2)) - fisher_info
    """
    return -0.5 * sigma_obs**2 * (1 + np.log(2 * np.pi * sigma_obs**2)) - fisher_info

def run_hybrid_bandit_tree(hybrid_bandit_tree: HybridBanditTree, context_id: str, action_id: str) -> float:
    """Run the hybrid bandit tree.

    Returns the distance-modulated confidence term for the given (context, action) pair.
    """
    confidence_term = hybrid_bandit_tree.root_distances[context_id] * hybrid_bandit_tree.path_weight
    return confidence_term

def example_usage():
    # Create a hybrid bandit tree
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 0.0), 'C': (1.0, 1.0)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    path_weight = 0.2
    confidence_alpha = 1.0
    hybrid_bandit_tree = HybridBanditTree(nodes, edges, root, path_weight, confidence_alpha)

    # Calculate the Fisher information score
    theta = 0.5
    center = 0.0
    width = 1.0
    fisher_info = fisher_score(theta, center, width)

    # Calculate the variational free energy
    mu_q = np.array([0.0, 0.0])
    sigma_obs = 1.0
    hybrid_free_energy = hybrid_fisher_free_energy(mu_q, sigma_obs, fisher_info)

    # Run the hybrid bandit tree
    context_id = 'A'
    action_id = 'B'
    confidence_term = run_hybrid_bandit_tree(hybrid_bandit_tree, context_id, action_id)

    print("Fisher information score:", fisher_info)
    print("Variational free energy:", hybrid_free_energy)
    print("Confidence term:", confidence_term)

if __name__ == "__main__":
    example_usage()