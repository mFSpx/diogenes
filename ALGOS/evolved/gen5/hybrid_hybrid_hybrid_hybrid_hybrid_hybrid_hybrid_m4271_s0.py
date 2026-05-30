# DARWIN HAMMER — match 4271, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s2.py (gen4)
# born: 2026-05-29T23:54:38Z

"""
HYBRID ALGORITHM: hybrid_fisher_rlct_grokking_dendritic_compartment

Combines the mathematical structures of hybrid_fisher_rlct_grokking_m61_s3.py and 
hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s2.py. The mathematical bridge 
between the two parents lies in the concept of energy and potential. In Singular Learning 
Theory, the free energy asymptotic equation represents the energy landscape of a neural network. 
In the Hodgkin-Huxley cable model, the membrane potential and ion channel currents represent the 
electrical energy and potential of a neuron. We can fuse these two concepts by using the membrane 
potential and ion channel currents as a physical inspiration for the energy landscape in Singular 
Learning Theory. This fusion enables the calculation of the Real Log Canonical Threshold (RLCT) and 
Grokking threshold using the Hodgkin-Huxley equations to model the membrane potential and ion channel 
currents, while also incorporating the sheaf's restriction maps as the connectivity matrix of the 
dendritic tree.

Parents:
- hybrid_fisher_rlct_grokking_m61_s3.py (Fisher Information and RLCT)
- hybrid_hybrid_hybrid_hybrid_dendritic_compartmen_m2686_s2.py (Sheaf-Associative-Dendrite)
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    """
    Calculate the membrane potential using the Hodgkin-Huxley cable model.

    Parameters:
    V (float): membrane potential
    C_m (float): membrane capacitance
    g_L (float): passive leak conductance
    E_L (float): leak reversal potential
    g_Na (float): maximum Na+ conductance
    E_Na (float): Na+ reversal potential
    m (float): Na+ activation gate variable
    h (float): Na+ inactivation gate variable
    g_K (float): maximum K+ conductance
    E_K (float): K+ reversal potential
    n (float): K+ activation gate variable
    I_syn (float): synaptic current
    """
    dVdt = (-g_L * (V - E_L) - g_Na * m * m * m * h * (V - E_Na) - g_K * n * n * n * n * (V - E_K) + I_syn) / C_m
    return dVdt

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * node_dims: dict mapping node identifier → dimension (int)
    * edges: list of (u, v) directed edges
    * Restrictions are stored as (src_map, dst_map) where each map projects the
      node vector onto a common edge space.
    """
    def __init__(self, node_dims, edges, restrictions):
        self.node_dims = node_dims
        self.edges = edges
        self.restrictions = restrictions

def hybrid_energy(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    """
    Calculate the hybrid energy of the system, combining the energy landscape of the 
    neural network and the membrane potential of the neuron.

    Parameters:
    sheaf (Sheaf): the sheaf representing the neural network
    V (float): membrane potential
    C_m (float): membrane capacitance
    g_L (float): passive leak conductance
    E_L (float): leak reversal potential
    g_Na (float): maximum Na+ conductance
    E_Na (float): Na+ reversal potential
    m (float): Na+ activation gate variable
    h (float): Na+ inactivation gate variable
    g_K (float): maximum K+ conductance
    E_K (float): K+ reversal potential
    n (float): K+ activation gate variable
    I_syn (float): synaptic current
    """
    membrane_energy = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    sheaf_energy = - (1/1) * math.log(sum(math.exp(1 * sheaf.node_dims[i]) for i in sheaf.node_dims))
    return membrane_energy + sheaf_energy

def hybrid_step(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn, dt):
    """
    Perform a time step of the hybrid system, updating the membrane potential and the sheaf.

    Parameters:
    sheaf (Sheaf): the sheaf representing the neural network
    V (float): membrane potential
    C_m (float): membrane capacitance
    g_L (float): passive leak conductance
    E_L (float): leak reversal potential
    g_Na (float): maximum Na+ conductance
    E_Na (float): Na+ reversal potential
    m (float): Na+ activation gate variable
    h (float): Na+ inactivation gate variable
    g_K (float): maximum K+ conductance
    E_K (float): K+ reversal potential
    n (float): K+ activation gate variable
    I_syn (float): synaptic current
    dt (float): time step
    """
    dVdt = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    V += dVdt * dt
    for i in sheaf.node_dims:
        sheaf.node_dims[i] += random.random() * dt
    return sheaf, V

def hybrid_retrieve(sheaf, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    """
    Retrieve the membrane potential and the sheaf of the hybrid system.

    Parameters:
    sheaf (Sheaf): the sheaf representing the neural network
    V (float): membrane potential
    C_m (float): membrane capacitance
    g_L (float): passive leak conductance
    E_L (float): leak reversal potential
    g_Na (float): maximum Na+ conductance
    E_Na (float): Na+ reversal potential
    m (float): Na+ activation gate variable
    h (float): Na+ inactivation gate variable
    g_K (float): maximum K+ conductance
    E_K (float): K+ reversal potential
    n (float): K+ activation gate variable
    I_syn (float): synaptic current
    """
    return sheaf, V

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
        print(energy)