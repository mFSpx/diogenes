# DARWIN HAMMER — match 4523, survivor 0
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s1.py (gen4)
# parent_b: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py (gen5)
# born: 2026-05-29T23:56:16Z

"""
HYBRID DENDRITIC VORONOI ANALYZER (HDVA) — match 1016, survivor 1
This module fuses the governing equations of Voronoi partitioning from voronoi_partition.py
and the Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer (HD-RW-TD) from
hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py.

The mathematical bridge between these two structures lies in the use of membrane potential
(V) from the dendritic model as a weighting factor to compute the regret-weighted Voronoi regions.

"""

import numpy as np
import math
import random
import sys
import pathlib

class Sheaf:
    def __init__(self, node_dims: dict, edges: list, seeds: list):
        self.node_dims: dict = dict(node_dims)
        self.edges: list = list(edges)
        self._restrictions: dict = {}
        self._sections: dict = {}
        self.seeds: list = seeds

    def set_restriction(
        self,
        edge: tuple,
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
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

    def assign(self, points: list):
        regions = {}
        for point in points:
            node = self.nearest(point, self.seeds)
            if node not in regions:
                regions[node] = []
            regions[node].append(point)
        return regions

    def nearest(self, point: tuple, seeds: list):
        if not seeds:
            raise ValueError('seeds required')
        return min(range(len(seeds)), key=lambda i: self.distance(point, seeds[i]))

    def distance(self, a: tuple, b: tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def regret_weighted_voronoi(self, membrane_potential: float):
        # Use membrane potential as a weighting factor to compute regret-weighted Voronoi regions
        regions = self.assign([(0, 0)])
        for node, points in regions.items():
            weighted_points = [(point[0] * membrane_potential, point[1] * membrane_potential) for point in points]
            self.set_section(node, np.array(weighted_points))
        return self._sections

def calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion, I_syn):
    """Calculate membrane potential.

    C_m * dV_i/dt = -g_L*(V_i - E_L) + I_ion(V_i) + I_syn(t)

    Parameters
    ----------
    C_m:
        Membrane capacitance (uF/cm^2)
    g_L:
        Passive leak conductance (mS/cm^2)
    E_L:
        Leak reversal potential (mV)
    V_i:
        Membrane potential of compartment i (mV)
    I_ion:
        Nonlinear ion current (mA/cm^2)
    I_syn:
        Synaptic current (mA/cm^2)
    """
    return -g_L * (V_i - E_L) + I_ion(V_i) + I_syn

def regret_weighted_ternary_decision(
    membrane_potential: float,
    action_values: list,
    regret_weights: list,
):
    # Use membrane potential to calculate regret-weighted probabilities
    probabilities = [p * membrane_potential for p in regret_weights]
    # Map probabilities onto a ternary alphabet
    ternary_actions = [np.argmax(prob) for prob in probabilities]
    return ternary_actions

def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

if __name__ == "__main__":
    # Smoke test
    C_m = 1.0  # Membrane capacitance (uF/cm^2)
    g_L = 0.1  # Passive leak conductance (mS/cm^2)
    E_L = -70.0  # Leak reversal potential (mV)
    V_i = -65.0  # Membrane potential of compartment i (mV)
    I_ion = lambda V: sodium_current(V, 0.5, 0.5, g_Na=120.0, E_Na=50.0)
    I_syn = 0.0  # Synaptic current (mA/cm^2)
    membrane_potential = calculate_membrane_potential(C_m, g_L, E_L, V_i, I_ion(V_i), I_syn)
    sheaf = Sheaf(node_dims={0: 2}, edges=[(0, 1)], seeds=[(0, 0)])
    regions = sheaf.regret_weighted_voronoi(membrane_potential)
    print(regions)