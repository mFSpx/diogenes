# DARWIN HAMMER — match 4521, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_nlms_hybrid_h_m978_s3.py (gen5)
# parent_b: hybrid_rlct_grokking_dendritic_compartmen_m61_s0.py (gen1)
# born: 2026-05-29T23:56:15Z

"""
This module fuses the Radial Basis Function (RBF) from hybrid_hybrid_hybrid_rbf_su_hybrid_model_vram_sc_m61_s0.py
with the Multi-Compartment Dendritic ODEs from rlct_grokking_dendritic_compartmen_m61_s0.py.
The mathematical bridge between these two structures is the concept of energy and its optimization.
The RBF and Grokking algorithm aim to optimize the free energy of a system, while the dendritic ODEs model
the energy dynamics of a neuron. This fusion integrates the energy-based optimization of RBF with
the energy dynamics of the dendritic ODEs to create a novel hybrid system.
"""

import numpy as np
import math
import random
import sys
import pathlib

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

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    return g_Na * (m ** 3) * h * (V - E_Na)

def optimize_energy(V, m, h, n, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0):
    sodium_curr = sodium_current(V, m, h, g_Na, E_Na)
    potassium_curr = g_K * (n ** 4) * (V - E_K)
    energy = sodium_curr + potassium_curr
    return energy

def gaussian_rbf(epsilon, r):
    """Radial basis function with a tunable bandwidth."""
    return math.exp(-((epsilon * r) ** 2))

def compute_phash(values):
    """Simple locality‑sensitive hash based on median threshold."""
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:                     # limit to 64 bits for a 64‑bit int
        bits = (bits << 1) | int(v >= median)
    return bits

def hamming_distance(a, b):
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def hybrid_rbf_dendritic_energy_optimization(V, m, h, n, train_losses_per_n, n_values, epsilon, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0):
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    energy = optimize_energy(V, m, h, n, g_Na, E_Na, g_K, E_K)
    rbf_term = gaussian_rbf(epsilon, np.linalg.norm([V, m, h, n]))  # using the Euclidean distance as the input for the RBF
    optimized_energy = energy - rlct * np.log(np.log(n_values[-1])) + rbf_term  # adding the RBF term to the optimized energy
    return optimized_energy

def hybrid_rlct_rbf_gaussian_noise(V, m, h, n, epsilon, train_losses_per_n, n_values, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0):
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    energy = optimize_energy(V, m, h, n, g_Na, E_Na, g_K, E_K)
    rbf_term = gaussian_rbf(epsilon, np.linalg.norm([V, m, h, n]))  # using the Euclidean distance as the input for the RBF
    gaussian_noise = np.random.normal(0, 1, 1)  # generating a random Gaussian noise
    noise_term = gaussian_rbf(epsilon, np.linalg.norm(gaussian_noise))  # using the Euclidean distance as the input for the RBF
    optimized_energy = energy - rlct * np.log(np.log(n_values[-1])) + rbf_term + noise_term  # adding the RBF term and the noise term to the optimized energy
    return optimized_energy

def hybrid_rbf_dendritic_energy_optimization_nlms(V, m, h, n, train_losses_per_n, n_values, epsilon, learning_rate, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0):
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    energy = optimize_energy(V, m, h, n, g_Na, E_Na, g_K, E_K)
    rbf_term = gaussian_rbf(epsilon, np.linalg.norm([V, m, h, n]))  # using the Euclidean distance as the input for the RBF
    nlms_term = (1 - learning_rate) * energy + learning_rate * rbf_term  # using the NLMS update rule to update the energy
    optimized_energy = energy - rlct * np.log(np.log(n_values[-1])) + nlms_term  # adding the RBF term and the NLMS term to the optimized energy
    return optimized_energy

if __name__ == "__main__":
    V = 10.0
    m = 0.5
    h = 0.5
    n = 0.5
    train_losses_per_n = [10.0, 20.0, 30.0]
    n_values = [100.0, 200.0, 300.0]
    epsilon = 1.0
    learning_rate = 0.1
    print(hybrid_rbf_dendritic_energy_optimization(V, m, h, n, train_losses_per_n, n_values, epsilon))
    print(hybrid_rlct_rbf_gaussian_noise(V, m, h, n, epsilon, train_losses_per_n, n_values))
    print(hybrid_rbf_dendritic_energy_optimization_nlms(V, m, h, n, train_losses_per_n, n_values, epsilon, learning_rate))