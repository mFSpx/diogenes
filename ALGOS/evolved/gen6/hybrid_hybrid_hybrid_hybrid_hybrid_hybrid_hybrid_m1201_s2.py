# DARWIN HAMMER — match 1201, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py (gen5)
# born: 2026-05-29T23:34:26Z

"""
This module defines a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py and 
hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py. 
The mathematical bridge between these two algorithms lies in the concept of entropy 
and information theory. The hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py 
algorithm uses entropy to calculate the expected entropy of a pheromone system, 
while the hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py algorithm uses 
information theory to calculate the Fisher score of a Gaussian beam. 
This hybrid algorithm leverages the concept of entropy and information theory to 
integrate the governing equations of both parent algorithms, creating a unified system 
that combines the pheromone system with dense associative memory and Kolmogorov-Arnold 
Network (KAN) transformations, and applies the Fisher score to the output of the 
KAN transformation.
"""

import math
import numpy as np
import random
import sys
from pathlib import Path

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox‑de Boor recursion for uniform clamped B‑splines.

    Parameters
    ----------
    x : ndarray shape (N,)
        Evaluation points (must lie inside the grid range).
    grid : ndarray shape (G,)
        Interior breakpoints (uniform spacing assumed).
    k : int, default 3
        Spline order (degree = k‑1).  k=3 yields cubic splines.

    Returns
    -------
    B : ndarray shape (N, G+k)
    """
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
    """KAN edgewise transform.

    Parameters
    ----------
    M : ndarray shape (N, N)
        Input matrix.
    grids : ndarray shape (G,)
        Interior breakpoints.
    coeffs : ndarray shape (G, N)
        B-spline coefficients.

    Returns
    -------
    M̂ : ndarray shape (N, N)
    """
    N = len(M)
    M_hat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_hat[i, j] = np.sum(coeffs[:, i] * M[:, j])
    return M_hat

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hybrid_operation(x: np.ndarray, grid: np.ndarray, M: np.ndarray, coeffs: np.ndarray, center: float, width: float) -> float:
    B = bspline_basis(x, grid)
    M_hat = kan_transform(M, grid, coeffs)
    theta = np.sum(B * M_hat)
    return fisher_score(theta, center, width)

def entropy_calculation(p: np.ndarray) -> float:
    return -np.sum(p * np.log2(p))

def hybrid_entropy_calculation(x: np.ndarray, grid: np.ndarray, M: np.ndarray, coeffs: np.ndarray, center: float, width: float) -> float:
    B = bspline_basis(x, grid)
    M_hat = kan_transform(M, grid, coeffs)
    theta = np.sum(B * M_hat)
    p = gaussian_beam(theta, center, width)
    return entropy_calculation(p)

if __name__ == "__main__":
    x = np.array([0.1, 0.2, 0.3])
    grid = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    coeffs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    center = 0.2
    width = 0.1
    print(hybrid_operation(x, grid, M, coeffs, center, width))
    print(hybrid_entropy_calculation(x, grid, M, coeffs, center, width))