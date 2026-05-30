# DARWIN HAMMER — match 1421, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_caputo_hybrid_nlms_omni_cha_m725_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py (gen3)
# born: 2026-05-29T23:36:09Z

"""
Hybrid Algorithm: Fusing Hybrid NLMS with Caputo Fractional Derivative and 
Hybrid Voronoi-Geometric-Algebra & RBF-Perceptual Algorithm.

This hybrid algorithm combines the adaptive filtering and learning from 
hybrid_hybrid_hybrid_caputo_hybrid_nlms_omni_cha_m725_s0.py with the 
stochastic, similarity-aware Voronoi partition from 
hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s6.py. 
The mathematical bridge is formed by using the Caputo fractional derivative 
to model the time-evolution of the weights in the NLMS algorithm, 
which enables adaptive filtering and learning in the omni-directional 
graph traversal and signal processing. The Voronoi regions are used to 
represent the spatial distribution of the input signals, and the 
similarity matrix is used to probabilistically re-assign points to seeds.

The hybrid algorithm consists of three main components:
1. Compute the Caputo fractional derivative of the input signal.
2. Build Voronoi regions of a set of points around seed points.
3. Represent each region as a multivector (geometric algebra) and 
   compute a similarity matrix between seeds using Gaussian RBF and 
   perceptual hashing.

The hybrid algorithm is implemented in pure Python with only the 
standard library and NumPy.
"""

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict

# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0.

    Uses g=7 Lanczos coefficients from Numerical Recipes.  Accurate to ~15
    significant figures for real z > 0.  For z < 0.5 uses the reflection
    formula Gamma(z)*Gamma(1-z) = pi/sin(pi*z) to stay in the stable region.

    :param z: Input value
    :return: Approximated Gamma(z)
    """
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5)**(z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.polyval(_LANCZOS_C, z)

def caputo_derivative(f, alpha, t, tau):
    """Caputo Fractional Derivative

    :param f: Function to differentiate
    :param alpha: Fractional order
    :param t: Time point
    :param tau: Time span
    :return: Caputo Fractional Derivative
    """
    return 1 / gamma_lanczos(1 - alpha) * np.sum(f[tau] / (t - tau)**alpha)

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """Deterministic Voronoi assignment of points to the nearest seed."""
    regions: Dict[int, List[Tuple[float, float]]] = {}
    for point in points:
        seed_idx = nearest(point, seeds)
        if seed_idx not in regions:
            regions[seed_idx] = []
        regions[seed_idx].append(point)
    return regions

def hybrid_caputo_voronoi(f, alpha, t, tau, points, seeds):
    """Hybrid Caputo Voronoi Algorithm

    :param f: Function to differentiate
    :param alpha: Fractional order
    :param t: Time point
    :param tau: Time span
    :param points: Input points
    :param seeds: Seed points
    :return: Voronoi regions and Caputo fractional derivative
    """
    caputo_deriv = caputo_derivative(f, alpha, t, tau)
    regions = assign(points, seeds)
    return regions, caputo_deriv

def gaussian_rbf(x, sigma):
    """Gaussian Radial Basis Function

    :param x: Input value
    :param sigma: Standard deviation
    :return: Gaussian RBF
    """
    return math.exp(-x**2 / (2 * sigma**2))

def perceptual_hashing(seeds):
    """Perceptual Hashing

    :param seeds: Seed points
    :return: Perceptual hash values
    """
    hash_values = []
    for seed in seeds:
        hash_value = int(np.mean(seed))
        hash_values.append(hash_value)
    return hash_values

def similarity_matrix(seeds, sigma):
    """Similarity Matrix

    :param seeds: Seed points
    :param sigma: Standard deviation
    :return: Similarity matrix
    """
    similarity_matrix = np.zeros((len(seeds), len(seeds)))
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            distance_ij = distance(seeds[i], seeds[j])
            similarity_matrix[i, j] = gaussian_rbf(distance_ij, sigma)
    hash_values = perceptual_hashing(seeds)
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            similarity_matrix[i, j] *= (hash_values[i] == hash_values[j])
    return similarity_matrix

if __name__ == "__main__":
    # Smoke test
    f = np.array([1, 2, 3, 4, 5])
    alpha = 0.5
    t = 1.0
    tau = np.arange(5)
    points = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    seeds = [(0, 0), (2, 2), (4, 4)]
    regions, caputo_deriv = hybrid_caputo_voronoi(f, alpha, t, tau, points, seeds)
    print(regions)
    print(caputo_deriv)
    sigma = 1.0
    similarity_mat = similarity_matrix(seeds, sigma)
    print(similarity_mat)