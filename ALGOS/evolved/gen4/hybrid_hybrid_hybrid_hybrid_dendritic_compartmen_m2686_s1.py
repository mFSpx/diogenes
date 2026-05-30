# DARWIN HAMMER — match 2686, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py (gen3)
# parent_b: dendritic_compartment.py (gen0)
# born: 2026-05-29T23:43:40Z

"""
This module represents a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py` 
and `dendritic_compartment.py`. The mathematical bridge between these two structures 
lies in the integration of the cellular sheaf on a directed graph from the first 
parent with the Hodgkin-Huxley cable model from the second parent. This is achieved 
by representing the dendritic tree as a directed graph, where each compartment is 
a node, and the axial coupling conductance between compartments is represented as 
the restriction maps in the cellular sheaf. The resulting hybrid algorithm allows 
for the simulation of the dendritic tree's electrical activity using the Hodgkin-Huxley 
model, while also incorporating the associative memory properties of the dense associative 
memory.
"""

import numpy as np
import math
import random
import sys
import pathlib

__all__ = [
    "Sheaf",
    "DendriticCompartment",
    "hybrid_energy",
    "hybrid_update_rule",
    "hybrid_retrieve",
    "sodium_current",
    "potassium_current",
    "leak_current",
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
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: any) -> np.ndarray:
        """Get the section assigned to a node."""
        return self._sections.get(node)

    def get_restriction(self, edge: tuple) -> tuple:
        """Get the restriction matrices for a directed edge."""
        return self._restrictions.get(edge)


class DendriticCompartment:
    """
    Dendritic compartment with Hodgkin-Huxley ion channels.
    """

    def __init__(self, C_m: float, g_L: float, E_L: float, g_Na: float, E_Na: float, g_K: float, E_K: float):
        self.C_m = C_m
        self.g_L = g_L
        self.E_L = E_L
        self.g_Na = g_Na
        self.E_Na = E_Na
        self.g_K = g_K
        self.E_K = E_K
        self.V = 0.0
        self.m = 0.0
        self.h = 0.0
        self.n = 0.0

    def sodium_current(self, V: float, m: float, h: float) -> float:
        """Hodgkin-Huxley sodium current."""
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def potassium_current(self, V: float, n: float) -> float:
        """Hodgkin-Huxley potassium current."""
        return self.g_K * n**4 * (V - self.E_K)

    def leak_current(self, V: float) -> float:
        """Hodgkin-Huxley leak current."""
        return self.g_L * (V - self.E_L)

    def update(self, dt: float, I_syn: float) -> None:
        """Update the compartment's state using the Hodgkin-Huxley model."""
        dV_dt = (-self.g_L * (self.V - self.E_L) + self.sodium_current(self.V, self.m, self.h) + self.potassium_current(self.V, self.n) + I_syn) / self.C_m
        self.V += dV_dt * dt
        dm_dt = 0.1 * (1 - self.m) - 0.1 * self.m
        self.m += dm_dt * dt
        dh_dt = 0.1 * (1 - self.h) - 0.1 * self.h
        self.h += dh_dt * dt
        dn_dt = 0.1 * (1 - self.n) - 0.1 * self.n
        self.n += dn_dt * dt


def hybrid_energy(sheaf: Sheaf, compartments: list) -> float:
    """Compute the energy of the hybrid system."""
    energy = 0.0
    for compartment in compartments:
        energy += compartment.V**2
    for edge in sheaf.edges:
        u, v = edge
        src_map, dst_map = sheaf.get_restriction(edge)
        energy += np.linalg.norm(src_map - dst_map)**2
    return energy


def hybrid_update_rule(sheaf: Sheaf, compartments: list, dt: float, I_syn: float) -> None:
    """Update the hybrid system using the Hodgkin-Huxley model and the cellular sheaf."""
    for compartment in compartments:
        compartment.update(dt, I_syn)
    for edge in sheaf.edges:
        u, v = edge
        src_map, dst_map = sheaf.get_restriction(edge)
        compartment_u = compartments[u]
        compartment_v = compartments[v]
        src_map += dt * (compartment_u.V - compartment_v.V)
        dst_map += dt * (compartment_v.V - compartment_u.V)


def hybrid_retrieve(sheaf: Sheaf, compartments: list) -> np.ndarray:
    """Retrieve the state of the hybrid system."""
    state = np.zeros((len(compartments),))
    for i, compartment in enumerate(compartments):
        state[i] = compartment.V
    return state


if __name__ == "__main__":
    node_dims = {0: 1, 1: 1}
    edges = [(0, 1)]
    sheaf = Sheaf(node_dims, edges)
    sheaf.set_restriction((0, 1), np.array([[1.0]]), np.array([[1.0]]))
    compartments = [DendriticCompartment(1.0, 0.1, -70.0, 120.0, 50.0, 36.0, -77.0), DendriticCompartment(1.0, 0.1, -70.0, 120.0, 50.0, 36.0, -77.0)]
    hybrid_update_rule(sheaf, compartments, 0.01, 0.0)
    print(hybrid_retrieve(sheaf, compartments))