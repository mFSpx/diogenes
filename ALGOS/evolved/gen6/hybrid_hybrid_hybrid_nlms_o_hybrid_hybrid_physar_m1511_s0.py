# DARWIN HAMMER — match 1511, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_sketches_hybr_m882_s0.py (gen5)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s2.py (gen5)
# born: 2026-05-29T23:36:51Z

"""
Hybrid Algorithm: Fusing NLMS and Physarum Flux Dynamics

This module fuses the core topologies of two parent algorithms:
1. hybrid_nlms_omni_chaotic_sprint_m59_s3.py (NLMS)
2. hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s2.py (Physarum)

The mathematical bridge between the two algorithms lies in the interpretation of the adaptive filtering
output as conductance modifiers in the Physarum network. The weekday weight vector is used to modulate
the pressure differences that drive the flux in the network.

The core idea is to use the NLMS update rule to estimate the banded, lead-lag transformed signals,
and then use these estimates as conductance modifiers in the Physarum network.
"""

import numpy as np
import math
import random
import sys
import pathlib

def lead_lag_transform(path):
    """
    Lead-lag transform of a banded signal.

    Parameters:
    path (numpy array): banded signal

    Returns:
    numpy array: lead-lag transformed signal
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     np.zeros(d)])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], np.zeros(d)])
    return out

def bspline_basis(x, grid, k=3):
    """
    B-spline basis functions.

    Parameters:
    x (numpy array): input signal
    grid (numpy array): grid points
    k (int): degree of B-spline

    Returns:
    numpy array: B-spline basis functions
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 1e-12 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 1e-12 else np.zeros(N)
            )
            B_new[:, i] = (term_l + term_r) / (denom_l + denom_r)
        B = B_new

    return B

def nlms_update(x, y, mu, sigma):
    """
    NLMS update rule.

    Parameters:
    x (numpy array): input signal
    y (numpy array): desired signal
    mu (float): step-size
    sigma (float): regularization parameter

    Returns:
    numpy array: NLMS update
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = float(mu)
    sigma = float(sigma)

    w = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        w[i] = (1 - mu) * w[i] + mu * (y[i] / (x[i]**2 + sigma))

    return w

def physarum_flux(x, conductance, edge_length, pressure_a, pressure_b, eps=1e-12):
    """
    Physarum flux on a single edge.

    Parameters:
    x (numpy array): conductance modifier
    conductance (float): conductance value
    edge_length (float): edge length
    pressure_a (float): pressure at node A
    pressure_b (float): pressure at node B
    eps (float): regularization parameter (default: 1e-12)

    Returns:
    float: Physarum flux
    """
    x = np.asarray(x, dtype=float)
    conductance = float(conductance)
    edge_length = float(edge_length)
    pressure_a = float(pressure_a)
    pressure_b = float(pressure_b)
    eps = float(eps)

    return conductance / max(edge_length, eps) * (pressure_a - pressure_b) * x

def hybrid_operation(x, y, mu, sigma, conductance, edge_length, pressure_a, pressure_b):
    """
    Hybrid operation combining NLMS update and Physarum flux.

    Parameters:
    x (numpy array): input signal
    y (numpy array): desired signal
    mu (float): step-size
    sigma (float): regularization parameter
    conductance (float): conductance value
    edge_length (float): edge length
    pressure_a (float): pressure at node A
    pressure_b (float): pressure at node B

    Returns:
    float: hybrid output
    """
    w = nlms_update(x, y, mu, sigma)
    flux_val = physarum_flux(w, conductance, edge_length, pressure_a, pressure_b)
    return flux_val

if __name__ == "__main__":
    import random
    import numpy as np

    # Generate random input and desired signals
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.random.rand(100, 1)

    # Set step-size and regularization parameter
    mu = 0.01
    sigma = 1e-6

    # Set conductance value, edge length, and pressure values
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 10.0
    pressure_b = 5.0

    # Run hybrid operation
    hybrid_out = hybrid_operation(x, y, mu, sigma, conductance, edge_length, pressure_a, pressure_b)

    # Print hybrid output
    print(hybrid_out)