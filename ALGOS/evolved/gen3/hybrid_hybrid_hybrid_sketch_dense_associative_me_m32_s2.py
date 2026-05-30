# DARWIN HAMMER — match 32, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4.py (gen2)
# parent_b: dense_associative_memory.py (gen0)
# born: 2026-05-29T23:25:19Z

"""
Module docstring:
This module introduces a novel hybrid algorithm, fusing the mathematical structures of 
'hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4.py' and 'dense_associative_memory.py'. 
The bridge between the two parents lies in the representation of sheaf sections as vectors 
and the energy function of the dense associative memory. Specifically, we integrate the 
cellular sheaf's restriction maps with the memory matrix of the dense associative memory, 
using the energy function to guide the sheaf's section assignments.

The mathematical interface is established by interpreting the sheaf sections as query 
vectors in the dense associative memory's energy function. The restriction maps of the 
sheaf are then used to update the memory matrix, effectively creating a sheaf-aware 
dense associative memory.
"""

import numpy as np
import random
import math
import sys
import pathlib

__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
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
        """Assign a vector to a node."""
        self._sections[node] = np.asarray(value, dtype=float)


class DenseAssociativeMemory:
    """
    Dense Associative Memory — Modern Hopfield Networks.
    """

    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)
        self.beta = beta

    def _softmax(self, z: np.ndarray):
        """Numerically stable softmax over 1-D array z."""
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _lse(self, z: np.ndarray):
        """log-sum-exp of 1-D array z (numerically stable)."""
        m = z.max()
        return m + np.log(np.exp(z - m).sum())

    def energy(self, xi: np.ndarray):
        """Compute the Dense AM energy E(xi).

        Parameters
        ----------
        xi : array shape (d,)
            Query / current state vector.

        Returns
        -------
        float
            Scalar energy value. Fixed-point attractors are local minima.
        """
        xi = np.asarray(xi, dtype=float)
        scores = self.beta * (self.patterns @ xi)
        lse_term = self._lse(scores) / self.beta
        quadratic_term = 0.5 * xi @ xi
        return -np.log(self._softmax(scores)).sum() + lse_term + quadratic_term


def hybrid_energy(sheaf: Sheaf, dam: DenseAssociativeMemory):
    """Compute the hybrid energy, integrating the sheaf's sections with the dense associative memory."""
    energy_values = []
    for node, section in sheaf._sections.items():
        energy_value = dam.energy(section)
        energy_values.append(energy_value)
    return np.mean(energy_values)


def hybrid_update_rule(sheaf: Sheaf, dam: DenseAssociativeMemory):
    """Update the sheaf's sections using the dense associative memory's update rule."""
    for node, section in sheaf._sections.items():
        updated_section = dam.patterns.T @ dam._softmax(dam.beta * (dam.patterns @ section))
        sheaf.set_section(node, updated_section)


def hybrid_retrieve(sheaf: Sheaf, dam: DenseAssociativeMemory, query: np.ndarray):
    """Retrieve the closest pattern in the dense associative memory using the sheaf's sections."""
    closest_pattern = None
    min_distance = float('inf')
    for node, section in sheaf._sections.items():
        distance = np.linalg.norm(query - section)
        if distance < min_distance:
            min_distance = distance
            closest_pattern = section
    return closest_pattern


if __name__ == "__main__":
    node_dims = {'A': 2, 'B': 2}
    edges = [('A', 'B')]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_restriction(('A', 'B'), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]))
    sheaf.set_section('A', np.array([1, 0]))
    sheaf.set_section('B', np.array([0, 1]))

    patterns = np.array([[1, 0], [0, 1]])
    dam = DenseAssociativeMemory(patterns)

    hybrid_energy_value = hybrid_energy(sheaf, dam)
    hybrid_update_rule(sheaf, dam)
    retrieved_pattern = hybrid_retrieve(sheaf, dam, np.array([1, 0]))

    print("Hybrid Energy:", hybrid_energy_value)
    print("Retrieved Pattern:", retrieved_pattern)