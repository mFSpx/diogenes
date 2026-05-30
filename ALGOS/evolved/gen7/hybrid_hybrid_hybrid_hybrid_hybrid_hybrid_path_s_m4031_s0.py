# DARWIN HAMMER — match 4031, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_hybrid_m1646_s0.py (gen6)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s2.py (gen3)
# born: 2026-05-29T23:53:13Z

"""
Hybrid Algorithm: Fusion of Hoeffding‑Tree Bandit Propensity (Parent A) and
Lead‑Lag / B‑Spline Path Signature (Parent B)

Mathematical Bridge
-------------------
* Parent A decides a tree split by comparing the gain gap with a Hoeffding
  confidence bound that is modulated by a *propensity* score coming from a
  contextual bandit.
* Parent B transforms a temporal path with a lead‑lag operator and smooths
  scalar quantities using B‑spline basis functions; it also measures
  information‑theoretic entropy of probability‑like vectors.

The hybrid algorithm couples these two ideas:

1. **Entropy‑adjusted Hoeffding bound** – the bandit propensities form a
   probability distribution. Its Shannon entropy 𝓗 is used to shrink the
   failure probability 𝛿 inside the Hoeffding bound:
   
   δ̂ = δ·(1‑p̄)·exp(‑𝓗)
   ε = sqrt( (r²·log(1/δ̂) + π·gini/6) / (2·n) )
   
   where *p̄* is the mean propensity and *gini* is the impurity coefficient.

2. **B‑spline smoothing of gain candidates** – raw gain values produced by
   candidate splits are treated as samples of a (noisy) gain curve.  A
   uniform knot grid is built and the B‑spline basis (order k) is used to
   obtain a smoothed estimate of the best and second‑best gain.

3. **Lead‑lag enriched features** – a multivariate time‑series path is first
   passed through the lead‑lag transform; the resulting doubled‑dimensional
   representation can be fed to any downstream learner (e.g. a Hoeffding
   tree).  In this module we expose the transform for completeness, but the
   core hybrid decision logic resides in `hybrid_should_split`.

The three public functions below demonstrate the fused computation:
`entropy`, `hybrid_hoeffding_bound`, and `hybrid_should_split`.  An additional
helper `smooth_gains` uses the B‑spline basis from Parent B.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Dataclasses – identical to those defined in Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float  # must lie in (0, 1)
    expected_reward: float
    confidence_bound: float


# ----------------------------------------------------------------------
# Parent B utilities (unchanged)
# ----------------------------------------------------------------------
def lead_lag_transform(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Lead‑lag transform that interleaves successive points."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute the B‑spline basis matrix of order `k` for points `x`
    over a knot grid defined by `grid`.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Construct the knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    M = len(t) - 1  # number of intervals for order‑1 basis

    # Order‑1 (piecewise constant) basis
    B = np.zeros((N, M), dtype=np.float64)
    for i in range(M):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    # The right‑most knot is inclusive
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recursive definition for higher orders
    for order in range(2, k + 1):
        M_new = len(t) - order
        B_new = np.zeros((N, M_new), dtype=np.float64)
        for i in range(M_new):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 0 else 0.0
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 0 else 0.0
            B_new[:, i] = term_l + term_r
        B = B_new

    return B


# ----------------------------------------------------------------------
# Information‑theoretic helper (new)
# ----------------------------------------------------------------------
def entropy(probabilities: Sequence[float]) -> float:
    """
    Shannon entropy of a probability‑like vector.
    Probabilities are assumed to be non‑negative and sum to 1; the function
    normalises them internally.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.size == 0:
        return 0.0
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    # Guard against log(0) by using where
    return -np.sum(np.where(p > 0, p * np.log(p), 0.0))


# ----------------------------------------------------------------------
# Hybrid Hoeffding bound that incorporates propensity and entropy
# ----------------------------------------------------------------------
def hybrid_hoeffding_bound(
    r: float,
    delta: float,
    n: int,
    gini_coeff: float,
    propensity: float,
    entropy_val: float,
) -> float:
    """
    Compute an entropy‑adjusted Hoeffding bound.

    Parameters
    ----------
    r : float
        Range of the random variable (max – min).
    delta : float
        Desired failure probability (0 < delta < 1).
    n : int
        Number of observations.
    gini_coeff : float
        Gini impurity coefficient (typically in [0,1]).
    propensity : float
        Mean propensity score from the bandit (0 < propensity < 1).
    entropy_val : float
        Shannon entropy of the propensity distribution.

    Returns
    -------
    epsilon : float
        The confidence bound.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound.")
    if not (0 < propensity < 1):
        raise ValueError("Propensity must lie in (0,1).")

    regularization = gini_coeff * math.pi / 6.0
    # Entropy reduces the effective delta exponentially
    delta_hat = delta * (1.0 - propensity) * math.exp(-entropy_val)
    # Clamp to avoid log(0)
    delta_hat = max(delta_hat, sys.float_info.epsilon)

    return math.sqrt((r * r * math.log(1.0 / delta_hat) + regularization) / (2.0 * n))


# ----------------------------------------------------------------------
# Gain smoothing via B‑spline (bridge to Parent B)
# ----------------------------------------------------------------------
def smooth_gains(gains: Sequence[float], k: int = 3) -> np.ndarray:
    """
    Produce a smooth approximation of the gain curve using a cubic B‑spline.
    The function builds a uniform knot grid covering the range of `gains`,
    evaluates the basis, and solves a least‑squares problem to obtain the
    spline coefficients.

    Returns the smoothed gain values evaluated at the original points.
    """
    gains_arr = np.asarray(gains, dtype=np.float64)
    if gains_arr.size == 0:
        return gains_arr

    # Uniform grid with a modest number of knots
    grid_points = max(5, int(np.cbrt(len(gains_arr))))  # heuristic
    grid = np.linspace(gains_arr.min(), gains_arr.max(), grid_points)

    B = bspline_basis(gains_arr, grid, k=k)

    # Least‑squares solution for coefficients: B c ≈ gains
    coeffs, *_ = np.linalg.lstsq(B, gains_arr, rcond=None)
    smoothed = B @ coeffs
    return smoothed


# ----------------------------------------------------------------------
# Core hybrid decision routine (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_should_split(
    gains: Sequence[float],
    r: float,
    delta: float,
    n: int,
    gini_coeff: float,
    bandit_actions: Sequence[BanditAction],
    tie_threshold: float = 0.05,
    k: int = 3,
) -> SplitDecision:
    """
    Decide whether to split a node by:

    1. Computing the entropy of the bandit propensities.
    2. Adjusting the Hoeffding bound with the mean propensity and the entropy.
    3. Smoothing the raw gain candidates with a B‑spline to obtain robust
       best/second‑best estimates.
    4. Comparing the gain gap with the bound (or the tie threshold).

    Returns a `SplitDecision` dataclass instance.
    """
    # --- 1. Entropy & mean propensity ---------------------------------
    propensities = np.asarray([a.propensity for a in bandit_actions], dtype=np.float64)
    if propensities.size == 0:
        # Fallback to neutral values if no bandit information is available
        mean_propensity = 0.5
        entropy_val = 0.0
    else:
        mean_propensity = propensities.mean()
        entropy_val = entropy(propensities)

    # --- 2. Entropy‑adjusted Hoeffding bound ---------------------------
    eps = hybrid_hoeffding_bound(r, delta, n, gini_coeff, mean_propensity, entropy_val)

    # --- 3. B‑spline smoothing of gains --------------------------------
    smoothed = smooth_gains(gains, k=k)
    if smoothed.size == 0:
        best_gain = second_best = 0.0
    else:
        best_gain = smoothed.max()
        # second best: max of values strictly less than the best (tolerance for float)
        mask = smoothed < best_gain - 1e-12
        second_best = smoothed[mask].max() if np.any(mask) else best_gain

    gap = best_gain - second_best

    # --- 4. Split decision logic ----------------------------------------
    split = (gap > eps) or (eps < tie_threshold)
    reason = (
        "gap_exceeds_bound"
        if gap > eps
        else ("tie_threshold" if eps < tie_threshold else "wait")
    )

    return SplitDecision(should_split=split, epsilon=eps, gain_gap=gap, reason=reason)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic temporal path (e.g., sensor readings)
    np.random.seed(42)
    path = np.cumsum(np.random.randn(20, 2), axis=0)  # 20 timesteps, 2‑D

    # Apply lead‑lag transform (Parent B component)
    transformed = lead_lag_transform(path)

    # Mock gain candidates – imagine they are information‑gain estimates for splits
    raw_gains = np.random.rand(7) * 0.5 + 0.1  # values in (0.1, 0.6)

    # Mock bandit actions (Parent A component)
    actions = [
        BanditAction(action_id=f"a{i}", propensity=random.uniform(0.1, 0.9),
                     expected_reward=random.uniform(0, 1),
                     confidence_bound=random.uniform(0, 0.5))
        for i in range(5)
    ]

    decision = hybrid_should_split(
        gains=raw_gains,
        r=1.0,                # assumed range of gain metric
        delta=0.05,
        n=150,                # number of observations at the node
        gini_coeff=0.4,
        bandit_actions=actions,
        tie_threshold=0.04,
        k=3,
    )

    print("Lead‑lag transformed shape:", transformed.shape)
    print("Raw gains:", raw_gains)
    print("Bandit propensities:", [a.propensity for a in actions])
    print("Hybrid split decision:", decision)