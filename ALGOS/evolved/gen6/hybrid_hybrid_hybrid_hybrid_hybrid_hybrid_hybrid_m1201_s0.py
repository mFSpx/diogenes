# DARWIN HAMMER — match 1201, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py (gen5)
# born: 2026-05-29T23:34:25Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py and 
hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py. 
The mathematical bridge between these two algorithms lies in the concept of 
information entropy and the representation of patterns in a dense associative memory. 
This hybrid algorithm leverages the concept of entropy to integrate the governing 
equations of both parent algorithms, creating a unified system that combines the 
pheromone system with dense associative memory and Kolmogorov-Arnold Network (KAN) transformations, 
and incorporates the Fisher score and developmental rate calculations.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    n = len(x)
    g = len(grid)
    B = np.zeros((n, g + k))
    for i in range(n):
        for j in range(g + k):
            if 0 <= j < k:
                B[i, j] = np.where((grid[j] <= x[i]) & (x[i] < grid[j + 1]), 1, 0)
            else:
                B[i, j] = (x[i] - grid[j - 1]) / (grid[j] - grid[j - 1]) * B[i, j - 1] + (grid[j + 1] - x[i]) / (grid[j + 1] - grid[j]) * B[i, j - 2]
    return B

def kan_transform(M: np.ndarray, grids: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    N = len(M)
    M_hat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_hat[i, j] = np.sum(coeffs * M[i, :] * coeffs)
    return M_hat

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params_rho_25: float = 1.0, params_delta_h_activation: float = 12_000.0, 
                      params_t_low: float = 283.15, params_t_high: float = 307.15, 
                      params_delta_h_low: float = -45_000.0, params_delta_h_high: float = 65_000.0, 
                      params_r_cal: float = 1.987) -> float:
    if temp_k <= 0 or params_rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params_rho_25 * (temp_k / 298.15) * math.exp((params_delta_h_activation / params_r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params_delta_h_low / params_r_cal) * ((1.0 / params_t_low) - (1.0 / temp_k)))
    high = math.exp((params_delta_h_high / params_r_cal) * ((1.0 / params_t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_operation(M: np.ndarray, grids: np.ndarray, coeffs: np.ndarray, theta: float, center: float, width: float, temp_c: float) -> np.ndarray:
    M_hat = kan_transform(M, grids, coeffs)
    fisher = fisher_score(theta, center, width)
    temp_k = c_to_k(temp_c)
    dev_rate = developmental_rate(temp_k)
    return M_hat * fisher * dev_rate

def main_test():
    M = np.random.rand(10, 10)
    grids = np.random.rand(10)
    coeffs = np.random.rand(10, 10)
    theta = 0.5
    center = 0.2
    width = 0.1
    temp_c = 25.0
    result = hybrid_operation(M, grids, coeffs, theta, center, width, temp_c)
    print(result)

if __name__ == "__main__":
    main_test()