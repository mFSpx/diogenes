# DARWIN HAMMER — match 4271, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s2.py (gen4)
# born: 2026-05-29T23:54:38Z

import numpy as np
import math
import random
import sys

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    dVdt = (-g_L * (V - E_L) - g_Na * m**3 * h * (V - E_Na) - g_K * n**4 * (V - E_K) + I_syn) / C_m
    return dVdt

class Sheaf:
    def __init__(self, node_dims, edges, restrictions):
        self.node_dims = node_dims
        self.edges = edges
        self.restrictions = restrictions

def hybrid_energy(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    membrane_energy = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    sheaf_energy = -sum(math.exp(sheaf.node_dims[i]) for i in sheaf.node_dims) / len(sheaf.node_dims)
    return membrane_energy + sheaf_energy

def hybrid_step(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn, dt):
    dVdt = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    V += dVdt * dt
    for i in sheaf.node_dims:
        sheaf.node_dims[i] += np.random.normal(0, dt)
    return sheaf, V

def hybrid_retrieve(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    return sheaf, V

def calculate_rlct(sheaf):
    rlct = 0
    for i in sheaf.node_dims:
        rlct += 1 / (1 + sheaf.node_dims[i]**2)
    return rlct

def calculate_grokking_threshold(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    energy = hybrid_energy(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    return energy / calculate_rlct(sheaf)

if __name__ == "__main__":
    node_dims = {"A": 1, "B": 2, "C": 3}
    edges = [("A", "B"), ("B", "C")]
    restrictions = [("A", "B", 1), ("B", "C", 2)]
    sheaf = Sheaf(node_dims, edges, restrictions)
    V = 0.0
    C_m = 1.0
    g_L = 0.1
    E_L = -0.1
    g_Na = 0.1
    E_Na = 0.1
    m = 0.1
    h = 0.1
    g_K = 0.1
    E_K = -0.1
    n = 0.1
    I_syn = 0.1
    dt = 0.01
    for i in range(100):
        sheaf, V = hybrid_step(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn, dt)
        energy = hybrid_energy(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
        rlct = calculate_rlct(sheaf)
        grokking_threshold = calculate_grokking_threshold(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
        print(f'Energy: {energy}, RLCT: {rlct}, Grokking Threshold: {grokking_threshold}')