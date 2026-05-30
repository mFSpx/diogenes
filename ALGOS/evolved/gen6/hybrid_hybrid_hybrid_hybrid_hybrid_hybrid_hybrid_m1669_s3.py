# DARWIN HAMMER — match 1669, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s0.py (gen4)
# born: 2026-05-29T23:38:09Z

"""Hybrid algorithm combining:
- Parent A: Multivector‑based physarum conductance updates with an RBF surrogate model.
- Parent B: Path‑signature (lead‑lag) representation with a Caputo fractional derivative.

Mathematical bridge:
The conductance vector of the physarum network is embedded as a grade‑1
Multivector **G(t)** = Σ_i g_i(t) e_i.  Its evolution is driven by a
Caputo fractional derivative of order α, **D^α G**, but the kernel of the
fractional derivative is weighted by the path‑signature coefficients of an
external control path **X(t)**.  The scalar free‑energy estimate **F(t)** is
obtained from a radial‑basis‑function (RBF) surrogate that receives the
components of **G(t)** as input.  The final update rule reads

    G(t+Δt) = G(t) − λ · ( D^α G(t)  ⊙  S_X(t) )  − η · ∇_G F(t),

where **S_X(t)** is the vector of path‑signature weights (obtained from a
lead‑lag transformed B‑spline basis), “⊙” denotes element‑wise multiplication,
and ∇_G F(t) is the gradient of the surrogate free‑energy w.r.t. the
conductance components (analytically available for Gaussian RBFs).

The module implements the three core ingredients and provides a single
function `hybrid_step` that performs one time‑step of this fused dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Dict, List, Tuple, Callable

# ----------------------------------------------------------------------
# Multivector utilities (excerpt from Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """Grade‑1 multivector representing a vector in a Clifford algebra."""
    def __init__(self, components: Dict[frozenset[int], float], n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items()
                           if abs(v) > 1e-15}
        self.n = int(n)

    @staticmethod
    def from_vector(v: np.ndarray) -> "Multivector":
        """Create a grade‑1 multivector from a 1‑D numpy array."""
        comps = {frozenset({i}): float(v[i]) for i in range(v.size)}
        return Multivector(comps, n=v.size)

    def to_vector(self) -> np.ndarray:
        """Extract the underlying vector (grade‑1 part)."""
        vec = np.zeros(self.n, dtype=float)
        for blade, coef in self.components.items():
            if len(blade) == 1:
                i = next(iter(blade))
                vec[i] = coef
        return vec

    def __add__(self, other: "Multivector") -> "Multivector":
        res = dict(self.components)
        for blade, coef in other.components.items():
            res[blade] = res.get(blade, 0.0) + coef
        return Multivector(res, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        res = dict(self.components)
        for blade, coef in other.components.items():
            res[blade] = res.get(blade, 0.0) - coef
        return Multivector(res, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.4g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


# ----------------------------------------------------------------------
# Path‑signature / B‑spline utilities (excerpt from Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape: (T, d). Output shape: (2*T‑1, 2*d).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis of order k at positions x.
    Returns a matrix (len(x), n_basis) where each column is a basis function.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2
    B = np.zeros((x.size, n_basis), dtype=np.float64)

    # zeroth order (piecewise constant)
    for i in range(len(t) - 1):
        mask = (x >= t[i]) & (x < t[i + 1])
        B[mask, i] = 1.0
    # right‑most knot
    B[x == t[-1], -1] = 1.0

    # recursive definition
    for order in range(2, k + 1):
        B_new = np.zeros((x.size, n_basis - (order - 1)), dtype=np.float64)
        for i in range(n_basis - (order - 1)):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = np.zeros_like(x)
            term_r = np.zeros_like(x)

            if denom_l > 0:
                term_l = ((x - t[i]) / denom_l) * B[:, i]
            if denom_r > 0:
                term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1]

            B_new[:, i] = term_l + term_r
        B = B_new
    return B


def path_signature_weights(path: np.ndarray, grid: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Approximate a low‑order path signature by projecting the lead‑lag transformed
    path onto a B‑spline basis. Returns a weight vector of length n_basis.
    """
    transformed = lead_lag_transform(path)  # (2T‑1, 2d)
    # For simplicity we treat each dimension independently and sum the coefficients.
    # Use the first coordinate (any could be used) as the scalar argument for the basis.
    t_vals = np.linspace(0.0, 1.0, transformed.shape[0])
    B = bspline_basis(t_vals, grid, k=order)  # (2T‑1, n_basis)
    # Least‑squares projection
    coeffs, *_ = np.linalg.lstsq(B, transformed[:, 0], rcond=None)
    return coeffs  # shape (n_basis,)


# ----------------------------------------------------------------------
# RBF surrogate model (simplified version of Parent A)
# ----------------------------------------------------------------------
def rbf_gaussian(x: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """Compute Gaussian RBF values for vector x against all centers."""
    diff = centers - x  # (n_centers, dim)
    sq_norm = np.einsum('ij,ij->i', diff, diff)
    return np.exp(-sq_norm / (2 * sigma ** 2))


def rbf_surrogate_predict(x: np.ndarray,
                          centers: np.ndarray,
                          weights: np.ndarray,
                          sigma: float) -> float:
    """
    Predict a scalar output (free energy) from input vector x using a
    linear combination of Gaussian RBFs.
    """
    phi = rbf_gaussian(x, centers, sigma)  # (n_centers,)
    return float(np.dot(weights, phi))


def rbf_surrogate_gradient(x: np.ndarray,
                           centers: np.ndarray,
                           weights: np.ndarray,
                           sigma: float) -> np.ndarray:
    """
    Analytic gradient of the Gaussian RBF surrogate w.r.t. input x.
    Returns a vector of shape (dim,).
    """
    diff = x - centers  # (n_centers, dim)
    phi = rbf_gaussian(x, centers, sigma)[:, None]  # (n_centers,1)
    grad = - (weights[:, None] * phi) * diff / (sigma ** 2)  # (n_centers, dim)
    return grad.sum(axis=0)


# ----------------------------------------------------------------------
# Caputo fractional derivative for a sequence of Multivectors
# ----------------------------------------------------------------------
def caputo_coefficients(alpha: float, n_steps: int, dt: float) -> np.ndarray:
    """
    Compute discrete Caputo coefficients c_k for k=0..n_steps‑1
    using the Grunwald‑Letnikov approximation.
    """
    c = np.zeros(n_steps, dtype=float)
    c[0] = 1.0
    for k in range(1, n_steps):
        c[k] = c[k - 1] * (alpha - (k - 1)) / k
    c = c * (dt ** -alpha)
    return c


def caputo_fractional_derivative_mv(series: List[Multivector],
                                   dt: float,
                                   alpha: float) -> Multivector:
    """
    Approximate the Caputo fractional derivative D^α G(t_n) for the latest
    multivector in `series` (assumed ordered in time).  Uses all past points.
    """
    n = len(series)
    coeffs = caputo_coefficients(alpha, n, dt)
    # D^α G_n ≈ Σ_{k=0}^{n-1} c_k * (G_{n‑k} - G_{n‑k-1})
    # where we define G_{-1}=0 (zero multivector)
    zero = Multivector({}, series[0].n)
    deriv = zero
    for k in range(1, n):
        delta = series[-k] - series[-k - 1]
        deriv = deriv + coeffs[k] * delta
    # Add the first term (k=0) which uses G_n - G_{n-1}
    if n >= 2:
        delta0 = series[-1] - series[-2]
        deriv = deriv + coeffs[0] * delta0
    else:
        # only one sample → treat derivative as zero
        pass
    return deriv


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def hybrid_step(conductance_vec: np.ndarray,
                path: np.ndarray,
                dt: float,
                alpha: float,
                rbf_centers: np.ndarray,
                rbf_weights: np.ndarray,
                rbf_sigma: float,
                lambda_: float,
                eta: float,
                history_mv: List[Multivector],
                spline_grid: np.ndarray) -> Tuple[np.ndarray, List[Multivector]]:
    """
    Perform one hybrid update.
    Returns the new conductance vector and the updated history of Multivectors.
    """
    # 1️⃣ Embed conductance into a multivector
    G_mv = Multivector.from_vector(conductance_vec)

    # 2️⃣ Append to history and compute fractional derivative
    history_mv.append(G_mv)
    D_alpha_G = caputo_fractional_derivative_mv(history_mv, dt, alpha)

    # 3️⃣ Compute path‑signature weights (scalar vector)
    sig_weights = path_signature_weights(path, spline_grid, order=3)  # (n_basis,)

    # 4️⃣ Align dimensions: repeat weights to match conductance dimension
    # (simple broadcasting: take first n components)
    w_vec = np.resize(sig_weights, conductance_vec.shape)

    # 5️⃣ Element‑wise product (fractional derivative modulated by signature)
    modulated = D_alpha_G.to_vector() * w_vec

    # 6️⃣ RBF surrogate free‑energy prediction and gradient
    F_pred = rbf_surrogate_predict(conductance_vec, rbf_centers, rbf_weights, rbf_sigma)
    grad_F = rbf_surrogate_gradient(conductance_vec, rbf_centers, rbf_weights, rbf_sigma)

    # 7️⃣ Hybrid update rule
    new_vec = (conductance_vec
               - lambda_ * modulated
               - eta * grad_F)

    # 8️⃣ Clip to keep conductances positive (physarum interpretation)
    new_vec = np.maximum(new_vec, 1e-8)

    return new_vec, history_mv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny physarum network (5 edges)
    dim = 5
    g = np.random.rand(dim) + 0.1

    # Synthetic control path (T=20, d=2)
    T, d = 20, 2
    t = np.linspace(0, 1, T)
    path = np.stack([np.sin(2 * math.pi * t), np.cos(2 * math.pi * t)], axis=1)

    # RBF surrogate parameters (3 centres)
    centers = np.random.rand(3, dim)
    weights = np.random.randn(3)
    sigma = 0.5

    # Hyper‑parameters
    dt = 0.01
    alpha = 0.7
    lambda_ = 0.05
    eta = 0.02

    # History container for multivectors
    history: List[Multivector] = []

    # Spline grid for signature projection
    spline_grid = np.linspace(0, 1, 8)

    # Run a few hybrid steps
    for step in range(10):
        g, history = hybrid_step(g, path, dt, alpha,
                                 centers, weights, sigma,
                                 lambda_, eta, history,
                                 spline_grid)
        print(f"Step {step:02d} | Conductance: {g}")

    print("Hybrid simulation completed without errors.")