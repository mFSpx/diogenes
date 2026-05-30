# DARWIN HAMMER — match 1201, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py (gen5)
# born: 2026-05-29T23:34:26Z

"""Hybrid Algorithm Fusion of `hybrid_hybrid_hybrid_pherom_hybrid_gliner_zero_s_m94_s0.py`
and `hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m503_s3.py`.

Both parent algorithms manipulate high‑dimensional pattern matrices.
Algorithm A represents patterns with B‑spline bases and applies a
Kolmogorov‑Arnold Network (KAN) edge‑wise transform; entropy of the
resulting distribution is the core scalar it uses.  
Algorithm B works in the information‑geometric domain, computing Fisher
information (via `fisher_score`) for Gaussian‑shaped beams and scaling
matrices by a temperature‑dependent developmental rate.

The mathematical bridge is the *information‑theoretic duality* between
Shannon entropy **H** and Fisher information **I**.  For a probability
distribution *p(θ)* derived from the KAN‑transformed pattern matrix we
can compute


H = -∑ p log p
I = ∑ (∂ log p / ∂θ)² p   (≈ Fisher score of a Gaussian beam)


The fused algorithm therefore:

1. Projects the raw pattern matrix onto a B‑spline basis.
2. Applies the KAN edge‑wise transform.
3. Normalises rows to a probability simplex (softmax) → entropy.
4. Evaluates a Gaussian‑beam Fisher score for each row using a
   user‑provided angle `θ`.
5. Scales the whole matrix by a temperature‑dependent developmental
   rate (Schoolfield model) to embed physiological dynamics.

The three principal functions below illustrate this hybrid pipeline."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from parent B (retained for completeness)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987

# ----------------------------------------------------------------------
# Core utilities from parent A
# ----------------------------------------------------------------------
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
    # extend grid with clamped knots at the ends
    knots = np.concatenate((
        np.full(k, grid[0]),
        grid,
        np.full(k, grid[-1])
    ))
    for i in range(n):
        for j in range(g + k):
            if k == 1:
                B[i, j] = 1.0 if knots[j] <= x[i] < knots[j + 1] else 0.0
            else:
                left = (x[i] - knots[j]) / (knots[j + k - 1] - knots[j]) if knots[j + k - 1] != knots[j] else 0.0
                right = (knots[j + k] - x[i]) / (knots[j + k] - knots[j + 1]) if knots[j + k] != knots[j + 1] else 0.0
                B[i, j] = left * B[i, j] + right * B[i, j + 1] if j + 1 < g + k else 0.0
    return B

def kan_transform(M: np.ndarray, grids: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Edge‑wise KAN transform using B‑spline bases.

    The matrix is flattened, each element is projected onto the spline
    basis defined by `grids`, then recombined with `coeffs`.  The result
    is reshaped back to the original matrix size.

    Parameters
    ----------
    M : ndarray shape (N, N)
        Input pattern matrix.
    grids : ndarray shape (G,)
        Interior breakpoints for the spline basis.
    coeffs : ndarray shape (G + k, N)
        Learned B‑spline coefficients (k is the spline order, default 3).

    Returns
    -------
    M̂ : ndarray shape (N, N)
        KAN‑transformed matrix.
    """
    N = M.shape[0]
    k = 3  # cubic splines, consistent with bspline_basis default
    flat = M.ravel()
    B = bspline_basis(flat, grids, k=k)          # shape (N*N, G+k)
    transformed = B @ coeffs                       # shape (N*N, N)
    # Reduce back to square shape by averaging across the coefficient dimension
    transformed = transformed.mean(axis=1)        # shape (N*N,)
    return transformed.reshape(N, N)

# ----------------------------------------------------------------------
# Information‑theoretic utilities (bridge between A and B)
# ----------------------------------------------------------------------
def softmax_rows(M: np.ndarray) -> np.ndarray:
    """Row‑wise softmax producing a probability distribution per row."""
    maxes = np.max(M, axis=1, keepdims=True)
    exp = np.exp(M - maxes)
    return exp / exp.sum(axis=1, keepdims=True)

def shannon_entropy(P: np.ndarray) -> np.ndarray:
    """Entropy per row of a probability matrix P."""
    eps = np.finfo(float).eps
    return -np.sum(P * np.log(P + eps), axis=1)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel evaluated at angle theta."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a single Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

# ----------------------------------------------------------------------
# Hybrid functions demonstrating the fused topology
# ----------------------------------------------------------------------
def hybrid_entropy_fisher(
    M: np.ndarray,
    grids: np.ndarray,
    coeffs: np.ndarray,
    theta: float,
    center: float,
    width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1. Apply KAN transform to the pattern matrix.
    2. Convert rows to a probability simplex (softmax) → entropy per row.
    3. Compute a Fisher score for each row using a Gaussian beam centred at `center`.
    4. Return (entropy_vector, fisher_vector).

    The two vectors have the same length (number of rows) and can be
    combined downstream for decision making.
    """
    transformed = kan_transform(M, grids, coeffs)          # (N,N)
    prob = softmax_rows(transformed)                      # (N,N)
    entropy = shannon_entropy(prob)                       # (N,)
    # Fisher score is independent of the matrix values; we reuse the same theta per row.
    fisher = np.array([fisher_score(theta, center, width) for _ in range(M.shape[0])])
    return entropy, fisher

def developmental_scaled_transform(
    M: np.ndarray,
    temp_celsius: float,
    params: SchoolfieldParams = SchoolfieldParams()
) -> np.ndarray:
    """
    Scale the entire matrix by a temperature‑dependent developmental rate.
    The temperature is supplied in Celsius and internally converted to Kelvin.
    """
    temp_k = temp_celsius + 273.15
    scale = developmental_rate(temp_k, params)
    return M * scale

def bandit_action_from_hybrid(
    action_id: str,
    entropy: np.ndarray,
    fisher: np.ndarray,
    base_propensity: float = 0.5
) -> BanditAction:
    """
    Construct a `BanditAction` whose expected reward is a weighted combination
    of normalized entropy and Fisher information.  This demonstrates how the
    hybrid scalar fields can feed directly into a bandit‑type decision module.
    """
    # Normalise entropy and fisher to [0,1]
    e_norm = (entropy - entropy.min()) / (entropy.ptp() + np.finfo(float).eps)
    f_norm = (fisher - fisher.min()) / (fisher.ptp() + np.finfo(float).eps)
    # Aggregate across rows (simple mean)
    reward_estimate = float(0.6 * e_norm.mean() + 0.4 * f_norm.mean())
    confidence = 1.0 / (1.0 + np.exp(-10 * (reward_estimate - 0.5)))  # logistic mapping
    return BanditAction(
        action_id=action_id,
        propensity=base_propensity,
        expected_reward=reward_estimate,
        confidence_bound=confidence,
        algorithm="HybridEntropyFisherBandit"
    )

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic pattern matrix
    N = 5
    rng = np.random.default_rng(42)
    M = rng.normal(loc=0.0, scale=1.0, size=(N, N))

    # Define spline grid and random coefficients for the KAN transform
    grid = np.linspace(-3, 3, 8)                # 8 interior knots
    coeffs = rng.normal(size=(len(grid) + 3, N))  # (G+k, N)

    # Hybrid entropy‑Fisher computation
    theta = math.pi / 4
    center = 0.0
    width = 0.5
    entropy_vec, fisher_vec = hybrid_entropy_fisher(M, grid, coeffs, theta, center, width)

    # Apply developmental scaling at 25 °C
    M_scaled = developmental_scaled_transform(M, temp_celsius=25.0)

    # Build a bandit action from the hybrid scalars
    action = bandit_action_from_hybrid("test_action", entropy_vec, fisher_vec)

    # Print results (simple verification that code runs)
    print("Entropy per row:", entropy_vec)
    print("Fisher per row :", fisher_vec)
    print("Scaled matrix  :", M_scaled)
    print("Bandit action  :", action)