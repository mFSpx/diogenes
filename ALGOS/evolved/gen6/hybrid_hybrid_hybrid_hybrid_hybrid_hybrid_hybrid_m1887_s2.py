# DARWIN HAMMER — match 1887, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py (gen5)
# born: 2026-05-29T23:39:41Z

"""
Module docstring:
This module introduces a novel hybrid algorithm, fusing the mathematical structures of 
'hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py' and 
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m522_s2.py'. 
The bridge between the two parents lies in the integration of the cellular sheaf's 
restriction maps with the regret-weighted strategy. Specifically, we use the 
regret-weighted strategy to guide the sheaf's section assignments, and the 
restriction maps to update the regrets.

The mathematical interface is established by interpreting the sheaf sections as 
query vectors in the regret-weighted strategy. The restriction maps of the sheaf 
are then used to update the regrets, effectively creating a sheaf-aware 
regret-weighted strategy.
"""

import numpy as np
import random
import math
import sys
import pathlib

__all__ = [
    "Sheaf",
    "RegretWeightedStrategy",
    "hybrid_energy",
    "hybrid_update_rule",
    "hybrid_retrieve",
]

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a linear restriction map
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and
      ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

class RegretWeightedStrategy:
    def __init__(self, regrets: list):
        self.regrets = regrets

    def compute_strategy(self):
        """
        Convert a list of non-negative regrets into a probability distribution.

        The classic regret-matching rule is used:
            p_i ∝ max(regret_i, 0)
        The probabilities sum to one; if all regrets are zero a uniform distribution
        is returned.
        """
        positive = [max(r, 0.0) for r in self.regrets]
        total = sum(positive)
        if total == 0.0:
            n = len(self.regrets)
            return [1.0 / n] * n
        return [p / total for p in positive]

def hybrid_energy(sheaf: Sheaf, strategy: RegretWeightedStrategy):
    """
    Compute the hybrid energy function.

    This function combines the energy function of the dense associative memory 
    with the restriction maps of the cellular sheaf.
    """
    energy = 0.0
    for edge, (src_map, dst_map) in sheaf._restrictions.items():
        u, v = edge
        src_section = sheaf._sections.get(u, np.zeros(sheaf.node_dims[u]))
        dst_section = sheaf._sections.get(v, np.zeros(sheaf.node_dims[v]))
        energy += np.linalg.norm(src_map @ src_section - dst_map @ dst_section)
    strategy_probabilities = strategy.compute_strategy()
    for i, prob in enumerate(strategy_probabilities):
        energy += prob * np.linalg.norm(sheaf._sections.get(i, np.zeros(sheaf.node_dims[i])))
    return energy

def hybrid_update_rule(sheaf: Sheaf, strategy: RegretWeightedStrategy):
    """
    Update the sheaf sections and regrets using the hybrid energy function.

    This function uses the regret-weighted strategy to guide the sheaf's section 
    assignments, and the restriction maps to update the regrets.
    """
    for edge, (src_map, dst_map) in sheaf._restrictions.items():
        u, v = edge
        src_section = sheaf._sections.get(u, np.zeros(sheaf.node_dims[u]))
        dst_section = sheaf._sections.get(v, np.zeros(sheaf.node_dims[v]))
        sheaf._sections[u] = src_section - 0.1 * (src_map.T @ (src_map @ src_section - dst_map @ dst_section))
        sheaf._sections[v] = dst_section - 0.1 * (dst_map.T @ (src_map @ src_section - dst_map @ dst_section))
    strategy.regrets = [r + 0.1 * np.linalg.norm(sheaf._sections.get(i, np.zeros(sheaf.node_dims[i]))) for i, r in enumerate(strategy.regrets)]

def hybrid_retrieve(sheaf: Sheaf, strategy: RegretWeightedStrategy):
    """
    Retrieve the sheaf sections using the hybrid energy function.

    This function uses the regret-weighted strategy to guide the sheaf's section 
    assignments.
    """
    strategy_probabilities = strategy.compute_strategy()
    retrieved_sections = {}
    for i, prob in enumerate(strategy_probabilities):
        retrieved_sections[i] = prob * sheaf._sections.get(i, np.zeros(sheaf.node_dims[i]))
    return retrieved_sections

if __name__ == "__main__":
    node_dims = {0: 10, 1: 10}
    edges = [(0, 1)]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_restriction((0, 1), np.random.rand(5, 10), np.random.rand(5, 10))
    sheaf.set_section(0, np.random.rand(10))
    sheaf.set_section(1, np.random.rand(10))

    regrets = [1.0, 2.0]
    strategy = RegretWeightedStrategy(regrets)

    print(hybrid_energy(sheaf, strategy))
    hybrid_update_rule(sheaf, strategy)
    print(hybrid_retrieve(sheaf, strategy))