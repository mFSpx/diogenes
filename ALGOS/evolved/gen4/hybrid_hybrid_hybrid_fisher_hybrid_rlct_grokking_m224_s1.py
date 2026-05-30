# DARWIN HAMMER — match 224, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s3.py (gen3)
# parent_b: hybrid_rlct_grokking_dendritic_compartmen_m61_s1.py (gen1)
# born: 2026-05-29T23:27:38Z

"""
Hybrid algorithm fusing the governing equations of 'hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s3.py' 
and 'hybrid_rlct_grokking_dendritic_compartmen_m61_s1.py'. The mathematical bridge between the two parents 
lies in the concept of energy and potential. In the first parent, the Fisher information and RLCT are used 
to optimize the dimensionality reduction process. In the second parent, the membrane potential and ion 
channel currents are used to model the electrical energy and potential of a neuron. We can fuse these two 
concepts by using the membrane potential and ion channel currents as a physical inspiration for the energy 
landscape in the Fisher information and RLCT.

The hybrid algorithm combines the Fisher information and RLCT with the membrane potential and ion channel 
currents to create a new energy function that represents the energy landscape of a neural network. This 
energy function can then be used to calculate the RLCT and Grokking threshold, providing a new perspective on 
the learning dynamics of neural networks.
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

def count_min_sketch(items, width=64, depth=4):
    table = [[0]*width for _ in range(depth)]
    for item in items:
        for d in range(depth): 
            table[d][int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(),16)%width]+=1
    return table

def estimate_rlct_from_losses(train_losses_per_n, n_values):
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)
    if np.any(ns <= np.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if len(losses) != len(ns):
        raise ValueError("train_losses_per_n and n_values must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)

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
    I_syn (float): synaptic input current

    Returns:
    float: membrane potential
    """
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    dV_dt = (-I_L - I_Na - I_K + I_syn) / C_m
    return V + dV_dt

def calculate_free_energy(n, L0, lambda_rlct, m=1):
    """
    Calculate the free energy using the Singular Learning Theory.

    Parameters:
    n (float): dataset size
    L0 (float): true risk
    lambda_rlct (float): RLCT
    m (float): parameter (default=1)

    Returns:
    float: free energy
    """
    return n * L0 + lambda_rlct * m

def hybrid_energy_function(theta, center, width, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    fisher_info = fisher_score(theta, center, width)
    membrane_potential = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    return fisher_info * membrane_potential

def hybrid_rlct_from_sketch(items, width=64, depth=4, num_theta=100):
    sketch = count_min_sketch(items, width, depth)
    flat_sketch = [item for sublist in sketch for item in sublist]
    losses = [item for item in flat_sketch if item > 0]
    n_values = [i+1 for i in range(len(losses))]
    if len(losses) > 1:
        rlct = estimate_rlct_from_losses(losses, n_values)
    else:
        rlct = 0.0

    # Use the Fisher information to optimize the dimensionality reduction process
    fisher_info = 0.0
    theta_values = np.linspace(-1.0, 1.0, num_theta)
    for theta in theta_values:
        fisher_info += fisher_score(theta, 0.0, 0.1)
    return rlct, fisher_info

def hybrid_grokking_from_free_energy(n, L0, lambda_rlct, m=1):
    free_energy = calculate_free_energy(n, L0, lambda_rlct, m)
    return free_energy

if __name__ == "__main__":
    theta = 0.5
    center = 0.0
    width = 0.1
    V = -65.0
    C_m = 1.0
    g_L = 0.1
    E_L = -65.0
    g_Na = 120.0
    E_Na = 50.0
    m = 0.05
    h = 0.6
    g_K = 36.0
    E_K = -77.0
    n = 0.6
    I_syn = 0.0
    n = 100
    L0 = 0.1
    lambda_rlct = 0.5

    energy = hybrid_energy_function(theta, center, width, V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)
    print("Hybrid Energy Function:", energy)

    items = [i for i in range(100)]
    rlct, fisher_info = hybrid_rlct_from_sketch(items)
    print("RLCT and Fisher Information:", rlct, fisher_info)

    free_energy = hybrid_grokking_from_free_energy(n, L0, lambda_rlct)
    print("Free Energy:", free_energy)