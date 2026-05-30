# DARWIN HAMMER — match 736, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s1.py (gen3)
# parent_b: caputo_fractional.py (gen0)
# born: 2026-05-29T23:30:48Z

"""
This module implements a novel hybrid algorithm that fuses the governing equations 
of the path signature and Kolmogorov-Arnold Networks (KAN) algorithms with the 
Caputo fractional derivative. The mathematical bridge between these structures 
lies in the representation of the path signature as a sequence of iterated integrals, 
which can be approximated using the B-spline basis functions employed in KANs, 
and the power-law decay kernel of the Caputo fractional derivative, which 
modulates the effective learning rate of the matrix update.

The hybrid algorithm replaces the Markovian recurrence in the Caputo fractional 
derivative with a path signature-weighted sum over the full history, 
and approximates the path signature using B-spline basis functions.

Parent algorithms: 
- hybrid_hybrid_path_signatur_hybrid_hybrid_model__m332_s1.py (KAN and path signature)
- caputo_fractional.py (Caputo fractional derivative)
"""

import numpy as np
import math

def lead_lag_transform(path):
    """
    Lead-lag transform: interleave (lead, lag) channels for causality encoding.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def bspline_basis(x, grid, k=3):
    """
    Evaluate B-spline basis functions of order k at positions x.
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
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B

def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0.

    Uses g=7 Lanczos coefficients from Numerical Recipes.  Accurate to ~15
    significant figures for real z > 0.  For z < 0.5 uses the reflection
    formula Gamma(z)*Gamma(1-z) = pi/sin(pi*z) to stay in the stable region.
    """
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

    if z < 0.5:
        # Reflection formula
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1 - z))

    z += _LANCZOS_G - 0.5
    zz = z
    z = np.asfarray(z, dtype=np.float64)

    logterm = (z - 0.5) * np.log(z + _LANCZOS_G) - z
    ser = 1.0 / z
    for i in range(_LANCZOS_G):
        ser += _LANCZOS_C[i] / (z + i + 1)

    return np.exp(logterm) * np.sqrt(2 * np.pi) * ser * zz

def caputo_derivative(f, t, alpha):
    """
    Compute the Caputo fractional derivative of f at t with order alpha.
    """
    dt = t[1] - t[0]
    f_prime = np.gradient(f, dt)
    kernel = (t[:, None] - t[None, :]) ** (alpha - 1) / gamma_lanczos(alpha)
    return np.sum(f_prime[None, :] * kernel, axis=1) / dt

def hybrid_step(path, alpha):
    """
    Perform a hybrid step combining path signature and Caputo fractional derivative.
    """
    lead_lag_path = lead_lag_transform(path)
    t = np.arange(len(path))
    B = bspline_basis(t, t)
    caputo_kernel = (t[:, None] - t[None, :]) ** (alpha - 1) / gamma_lanczos(alpha)
    return np.dot(B, lead_lag_path) * caputo_kernel

def test_hybrid_step():
    np.random.seed(0)
    path = np.random.rand(10, 2)
    alpha = 0.5
    result = hybrid_step(path, alpha)
    print(result)

if __name__ == "__main__":
    test_hybrid_step()