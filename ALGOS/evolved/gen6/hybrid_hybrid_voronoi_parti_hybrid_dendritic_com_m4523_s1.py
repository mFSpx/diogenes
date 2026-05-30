# DARWIN HAMMER — match 4523, survivor 1
# gen: 6
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s1.py (gen4)
# parent_b: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py (gen5)
# born: 2026-05-29T23:56:16Z

"""
Hybrid Voronoi-Dendritic Regret-Weighted Analyzer (HVD-RWA)

This module fuses the governing equations of two parent algorithms:
- Hybrid Voronoi Partitioning and Cellular Sheaf Theory (HVP-CT) from `hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s1.py`
- Hybrid Dendritic Regret-Weighted Ternary-Decision Analyzer (HD-RW-TD) from `hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s3.py`

The mathematical bridge between the two parents is established by using the Voronoi regions from the HVP-CT algorithm as inputs to calculate regret-weighted probabilities in the HD-RW-TD algorithm. Specifically, the Voronoi regions are used to define the input to the dendritic model, which generates membrane potentials that are then used to compute regret-weighted probabilities.

"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ---------------------------------------------------------------------------
# Voronoi model utilities
# ---------------------------------------------------------------------------
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

def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

# ---------------------------------------------------------------------------
# Dendritic model utilities
# ---------------------------------------------------------------------------
def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current.

    I_Na = g_Na * m^3 * h * (V - E_Na)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    """
    return g_Na * m**3 * h * (V - E_Na)

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
        Nonlinear 
    """
    return -g_L*(V_i - E_L) + I_ion + I_syn

def hybrid_vorono_dendritic_regret_weighted_analyzer(sheaf: Sheaf, points: list, C_m: float, g_L: float, E_L: float, V_i: float, I_syn: float):
    regions = sheaf.assign(points)
    regret_weights = []
    for region in regions.values():
        centroid = np.mean(region, axis=0)
        V = calculate_membrane_potential(C_m, g_L, E_L, V_i, sodium_current(V_i, 0.5, 0.5), I_syn)
        regret_weight = _softmax(np.array([V, centroid[0], centroid[1]]))
        regret_weights.append(regret_weight)
    return regret_weights

def regret_weighted_ternary_decision(regret_weights: list):
    ternary_decisions = []
    for regret_weight in regret_weights:
        decision = np.argmax(regret_weight)
        ternary_decisions.append(decision)
    return ternary_decisions

def main():
    sheaf = Sheaf({0: 2, 1: 2}, [(0, 1)], [(0, 0), (1, 1)])
    points = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)]
    C_m = 1.0
    g_L = 0.1
    E_L = -70.0
    V_i = -70.0
    I_syn = 10.0
    regret_weights = hybrid_vorono_dendritic_regret_weighted_analyzer(sheaf, points, C_m, g_L, E_L, V_i, I_syn)
    ternary_decisions = regret_weighted_ternary_decision(regret_weights)
    print(ternary_decisions)

if __name__ == "__main__":
    main()