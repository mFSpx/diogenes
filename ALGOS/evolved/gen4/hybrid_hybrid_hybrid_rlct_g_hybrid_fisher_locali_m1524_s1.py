# DARWIN HAMMER — match 1524, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s1.py (gen2)
# parent_b: hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s1.py (gen3)
# born: 2026-05-29T23:38:25Z

"""
This module fuses the Pheromone-based RLCT and Infotaxis algorithm from 'hybrid_hybrid_rlct_grokking_hybrid_pheromone_inf_m208_s1.py'
with the Fisher-information scoring and lead-lag transform from 'hybrid_fisher_localization_hybrid_hybrid_path_s_m20_s1.py'.
The mathematical bridge between the two structures is based on representing the Fisher score as a function 
that can be approximated using the expected entropy from the Pheromone-based Infotaxis algorithm.

The Fisher score is used to compute the derivative of the Gaussian beam, which is then used as the input 
to the lead-lag transform. The expected entropy from the Pheromone-based Infotaxis algorithm is used to 
weight the Fisher score, effectively integrating the information-based optimization of RLCT with the 
exploration-exploitation trade-off of Infotaxis.

The hybrid system fuses the estimation of RLCT from losses with the Fisher-information scoring for off-axis 
sensing, and applies the lead-lag transform to the resulting path.
"""

import numpy as np
import math
import random
import sys
import pathlib

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

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

def pheromone_infotaxis_optimization(V, m, h, n, train_losses_per_n, n_values, g_Na=120.0, E_Na=50.0, g_K=36.0, E_K=-77.0, pheromone_signal_half_life=3600.0):
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    sodium_current = g_Na * (m ** 3) * h * (V - E_Na)
    potassium_current = g_K * (n ** 4) * (V - E_K)
    energy = sodium_current + potassium_current
    expected_entropy = -rlct * np.log2(rlct) - (1 - rlct) * np.log2(1 - rlct)
    return energy, expected_entropy

def hybrid_fisher_rlct(theta_values: np.ndarray, center: float, width: float, train_losses_per_n, n_values) -> np.ndarray:
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    features = np.array([fisher_score(theta, center, width) * rlct for theta in theta_values])
    return lead_lag_transform(features.reshape(-1, 1))

def best_angle(candidates: np.ndarray, center: float, width: float, train_losses_per_n, n_values) -> float:
    rlct = estimate_rlct_from_losses(train_losses_per_n, n_values)
    scores = np.array([fisher_score(theta, center, width) * rlct for theta in candidates])
    return candidates[np.argmax(scores)]

def smoke_test():
    theta_values = np.linspace(-10, 10, 100)
    center = 0.0
    width = 1.0
    train_losses_per_n = [0.1, 0.2, 0.3]
    n_values = [10, 20, 30]
    hybrid_path = hybrid_fisher_rlct(theta_values, center, width, train_losses_per_n, n_values)
    best_theta = best_angle(theta_values, center, width, train_losses_per_n, n_values)
    print(hybrid_path.shape)
    print(best_theta)

if __name__ == "__main__":
    smoke_test()