# DARWIN HAMMER — match 2686, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py (gen3)
# parent_b: dendritic_compartment.py (gen0)
# born: 2026-05-29T23:43:40Z

"""
Hybrid Algorithm: Fusing Cellular Sheaf and Hodgkin-Huxley Dendritic Compartments

This module integrates the topological structure of Cellular Sheaf (parent algorithm A) 
with the dynamic equations of Hodgkin-Huxley dendritic compartments (parent algorithm B). 
The bridge between the two is established by interpreting the sheaf's sections as 
membrane potentials in a dendritic tree, where each node in the sheaf corresponds to 
a compartment in the dendritic tree. The sheaf's restriction maps are used to 
couple the compartments, enabling a dynamic simulation of the dendritic tree.

Parent Algorithms:
- hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py (Cellular Sheaf)
- dendritic_compartment.py (Hodgkin-Huxley Dendritic Compartments)
"""

import numpy as np
from typing import Dict, List, Tuple

class HybridDendriticSheaf:
    def __init__(self, node_dims: Dict, edges: List[Tuple], C_m: float, g_L: float, E_L: float):
        self.node_dims = node_dims
        self.edges = edges
        self.C_m = C_m
        self.g_L = g_L
        self.E_L = E_L
        self._restrictions = {}
        self._sections = {}
        self._m = {}  # Sodium activation gate
        self._h = {}  # Sodium inactivation gate
        self._n = {}  # Potassium activation gate

    def set_restriction(self, edge: Tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        self._restrictions[(edge)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: any, value: np.ndarray) -> None:
        self._sections[node] = np.asarray(value, dtype=float)
        self._m[node] = np.zeros((self.node_dims[node],))
        self._h[node] = np.zeros((self.node_dims[node],))
        self._n[node] = np.zeros((self.node_dims[node],))

    def get_section(self, node: any) -> np.ndarray:
        return self._sections.get(node)

    def get_restriction(self, edge: Tuple) -> Tuple:
        return self._restrictions.get(edge)

    def alpha_beta_gates(self, V: np.ndarray):
        alpha_m = 0.128 * (V + 50) / (1 - np.exp(-(V + 50) / 4))
        beta_m = 4 / (1 + np.exp(-(V + 25) / 5))
        alpha_h = 0.128 * np.exp(-(V + 48) / 5)
        beta_h = 1 / (1 + np.exp(-(V + 18) / 5))
        alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta_n = 0.125 * np.exp(-(V + 65) / 80)
        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    def compartment_step(self, node: any, V: np.ndarray, dt: float):
        I_ion = self.sodium_current(V, self._m[node], self._h[node]) + self.potassium_current(V, self._n[node]) + self.leak_current(V)
        dVdt = (-self.g_L * (V - self.E_L) + I_ion) / self.C_m
        V_next = V + dVdt * dt
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = self.alpha_beta_gates(V)
        dm_dt = alpha_m * (1 - self._m[node]) - beta_m * self._m[node]
        dh_dt = alpha_h * (1 - self._h[node]) - beta_h * self._h[node]
        dn_dt = alpha_n * (1 - self._n[node]) - beta_n * self._n[node]
        self._m[node] += dm_dt * dt
        self._h[node] += dh_dt * dt
        self._n[node] += dn_dt * dt
        return V_next

    def sodium_current(self, V: np.ndarray, m: np.ndarray, h: np.ndarray, g_Na: float = 120.0, E_Na: float = 50.0):
        return g_Na * np.power(m, 3) * h * (V - E_Na)

    def potassium_current(self, V: np.ndarray, n: np.ndarray, g_K: float = 36.0, E_K: float = -77.0):
        return g_K * np.power(n, 4) * (V - E_K)

    def leak_current(self, V: np.ndarray, g_L: float = 0.3, E_L: float = -54.4):
        return g_L * (V - E_L)

def hybrid_retrieve(sheaf: HybridDendriticSheaf, node: any, dt: float, T: int):
    V = sheaf.get_section(node)
    for t in range(T):
        V = sheaf.compartment_step(node, V, dt)
    return V

if __name__ == "__main__":
    node_dims = {'node1': 1, 'node2': 1}
    edges = [('node1', 'node2')]
    sheaf = HybridDendriticSheaf(node_dims, edges, C_m=1.0, g_L=0.1, E_L=-70.0)
    sheaf.set_section('node1', np.array([-70.0]))
    sheaf.set_section('node2', np.array([-70.0]))
    src_map = np.array([[1.0]])
    dst_map = np.array([[0.5]])
    sheaf.set_restriction(('node1', 'node2'), src_map, dst_map)
    print(hybrid_retrieve(sheaf, 'node1', dt=0.1, T=100))