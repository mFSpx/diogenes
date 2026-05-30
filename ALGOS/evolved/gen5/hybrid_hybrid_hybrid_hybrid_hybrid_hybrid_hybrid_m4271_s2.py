# DARWIN HAMMER — match 4271, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s2.py (gen4)
# born: 2026-05-29T23:54:38Z

import numpy as np
import math
import random

class Sheaf:
    def __init__(self, node_dims, edges, restrictions):
        self.node_dims = node_dims
        self.edges = edges
        self.restrictions = restrictions

def gaussian_beam(theta, center, width):
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta, center, width, eps=1e-12):
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    dVdt = (-g_L * (V - E_L) - g_Na * m * m * m * h * (V - E_Na) - g_K * n * n * n * n * (V - E_K) + I_syn) / C_m
    return dVdt

def hybrid_energy(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    membrane_energy = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    sheaf_energy = - np.sum([math.exp(1 * dim) for dim in sheaf.node_dims.values()]) / len(sheaf.node_dims)
    return membrane_energy + sheaf_energy

def hybrid_step(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn, dt):
    dVdt = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    V += dVdt * dt
    for i in sheaf.node_dims:
        sheaf.node_dims[i] += 0.1 * np.random.rand() * dt
    return sheaf, V

def hybrid_retrieve(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    return sheaf, V

def main():
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
        print(energy)

if __name__ == "__main__":
    main()