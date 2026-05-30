# DARWIN HAMMER — match 4383, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1674_s0.py (gen5)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_rlct_g_m1454_s2.py (gen5)
# born: 2026-05-29T23:55:24Z

"""
Hybrid Doomsday‑RLMS Scheduler

This module fuses the two parent algorithms:

* **Parent A** – provides a deterministic Doomsday calendar (`doomsday`),
  a weekday‑dependent weight vector (`weekday_weight_vector`), a lead‑lag
  transform for path signatures and a B‑spline basis evaluator.
* **Parent B** – supplies a NLMS adaptive filter (`nlms_predict`,
  `nlms_update`) whose learning‑rate is modulated by a Real‑Log‑Canonical‑Threshold
  (RLCT) estimate and by a regret‑weighted exploration factor derived from the
  Shannon entropy of a hybrid discrete state.

The mathematical bridge is the **weekday index**.  The index is turned into a
continuous weight vector (A) that scales the NLMS step‑size (B).  Simultaneously,
the path signature (approximated by a lead‑lag transform and B‑spline basis)
produces the NLMS input vector.  Thus a single update step simultaneously
leverages calendar‑derived temporal information, geometric‑product‑style
feature extraction, and adaptive‑filter learning dynamics.
"""

import sys
import math
import random
import pathlib
from datetime import date as _date

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Doomsday calendar utilities
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Classic Doomsday algorithm (0=Sunday … 6=Saturday)."""
    return (year % 7 + math.floor((13 * (month + 1)) / 5) + day) % 7


def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """Continuous weight vector on the unit simplex, modulated by weekday."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input shape (T, d). Output shape (2·T‑1, 2·d).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: float, knots: np.ndarray, k: int = 3) -> float:
    """
    Cox–de Boor recursion for a single B‑spline basis value.
    `knots` must be a non‑decreasing 1‑D array of length >= k+2.
    """
    # Find the knot span index i such that knots[i] <= x < knots[i+1]
    i = np.searchsorted(knots, x, side='right') - 1
    i = max(k, min(i, len(knots) - k - 2))

    # Zero‑order basis
    N = np.zeros(len(knots) - 1)
    N[i] = 1.0 if knots[i] <= x < knots[i + 1] else 0.0

    # Recursion
    for d in range(1, k + 1):
        Nd = np.zeros_like(N)
        for j in range(i - d, i + 1):
            left = (x - knots[j]) / (knots[j + d] - knots[j]) if knots[j + d] != knots[j] else 0.0
            right = (knots[j + d + 1] - x) / (knots[j + d + 1] - knots[j + 1]) if knots[j + d + 1] != knots[j + 1] else 0.0
            Nd[j] = left * N[j] + right * N[j + 1]
        N = Nd
    return N[i]


# ----------------------------------------------------------------------
# Parent B – NLMS adaptive filter and RLCT‑regret bridge
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def rlct_estimate(error_seq: np.ndarray) -> float:
    """
    Very simple RLCT surrogate: log‑mean‑square‑error slope.
    For a real implementation this would be derived from free‑energy asymptotics.
    """
    if error_seq.size == 0:
        return 0.0
    mse = np.mean(error_seq ** 2)
    # Clamp to avoid log(0)
    mse = max(mse, 1e-12)
    return math.log(mse)


def shannon_entropy(probs: np.ndarray) -> float:
    """Entropy H = -∑ p·log(p) (base e)."""
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def regret_factor(state: np.ndarray) -> float:
    """
    Compute a regret‑weighted exploration factor from the Shannon entropy
    of a hybrid discrete state (one‑hot weekday + ternary token).
    The factor lives in (0, 1] and grows with uncertainty.
    """
    # Convert integer state entries to a probability distribution
    counts = np.bincount(state.astype(int), minlength=3)
    probs = counts / counts.sum()
    entropy = shannon_entropy(probs)
    # Normalise by the maximal entropy log(3)
    max_entropy = math.log(3)
    return 1.0 - (entropy / max_entropy) + 1e-6  # avoid exact zero


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                d: float,
                mu_base: float,
                error_history: np.ndarray) -> np.ndarray:
    """
    Perform a single NLMS weight update with a step‑size that is
    *modulated by*:
        – the RLCT estimate of the error sequence,
        – the regret factor derived from a hybrid state.
    Returns the updated weight vector.
    """
    # Prediction and instantaneous error
    y = nlms_predict(weights, x)
    e = d - y

    # RLCT‑based damping
    rlct = rlct_estimate(error_history)
    mu_rlct = mu_base / (1.0 + rlct)

    # Regret‑based damping (state is constructed later)
    # For the purpose of this function we accept a placeholder factor of 1.
    mu = mu_rlct

    # Normalised step size (NLMS)
    norm_x2 = np.dot(x, x) + 1e-12
    weights_new = weights + (mu / norm_x2) * e * x
    return weights_new


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge between the two parents
# ----------------------------------------------------------------------
def hybrid_feature_extraction(path: np.ndarray,
                              year: int,
                              month: int,
                              day: int,
                              groups: list[str]) -> np.ndarray:
    """
    Build a feature vector for the NLMS filter:

    1. Lead‑lag transform of the raw path.
    2. Approximate each transformed coordinate with a B‑spline basis
       evaluated on an equally spaced knot vector.
    3. Concatenate the weekday‑dependent weight vector (A) to the spline
       coefficients, yielding a single 1‑D array.
    """
    # 1. Lead‑lag transform
    ll = lead_lag_transform(path)                     # shape (2T‑1, 2d)

    # 2. B‑spline projection (simple inner product with basis values)
    T2, dim = ll.shape
    knots = np.linspace(0, 1, T2 + 4)                 # k=3 → len = T2 + k + 1
    spline_coeffs = np.empty(dim, dtype=float)
    for j in range(dim):
        # Sample basis at normalized positions i/(T2‑1)
        coeff = 0.0
        for i in range(T2):
            x = i / max(T2 - 1, 1)
            coeff += ll[i, j] * bspline_basis(x, knots, k=3)
        spline_coeffs[j] = coeff / T2

    # 3. Weekday weight vector
    dow = doomsday(year, month, day)
    wvec = weekday_weight_vector(groups, dow)        # length = len(groups)

    # Concatenate (spline coefficients first, then calendar weights)
    return np.concatenate([spline_coeffs, wvec])


def hybrid_nlms_step(weights: np.ndarray,
                     path: np.ndarray,
                     target: float,
                     year: int,
                     month: int,
                     day: int,
                     groups: list[str],
                     mu_base: float,
                     error_history: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Perform one hybrid NLMS iteration.

    Returns:
        (updated_weights, prediction_error)
    """
    # Extract features that fuse calendar info with geometric path encoding
    x = hybrid_feature_extraction(path, year, month, day, groups)

    # Build hybrid discrete state for regret factor:
    #   - one‑hot weekday (length 7)
    #   - ternary token derived from a simple hash of the date
    weekday_one_hot = np.zeros(7, dtype=int)
    weekday_one_hot[doomsday(year, month, day) % 7] = 1
    token = (year * 31 + month * 13 + day) % 3
    ternary = np.array([token, (token + 1) % 3, (token + 2) % 3], dtype=int)
    hybrid_state = np.concatenate([weekday_one_hot, ternary])

    # Regret factor influences the effective step size
    reg_factor = regret_factor(hybrid_state)
    mu_effective = mu_base * reg_factor

    # NLMS weight update with the fused step size
    updated_weights = nlms_update(weights, x, target, mu_effective, error_history)

    # Compute prediction error for bookkeeping
    pred = nlms_predict(updated_weights, x)
    error = target - pred
    return updated_weights, error


def hybrid_predict(path: np.ndarray,
                   weights: np.ndarray,
                   year: int,
                   month: int,
                   day: int,
                   groups: list[str]) -> float:
    """
    Produce a prediction for a given path and date using the hybrid feature map.
    """
    x = hybrid_feature_extraction(path, year, month, day, groups)
    return nlms_predict(weights, x)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy path: 5 time steps, 2‑dimensional
    T, d = 5, 2
    path = np.random.randn(T, d)

    # Random initial NLMS weights (dimension = spline coeffs + group weights)
    groups = ["codex", "groq", "cohere", "local_models"]
    dummy_year, dummy_month, dummy_day = 2023, 11, 17

    # Build a temporary feature vector to size the weight array
    temp_feat = hybrid_feature_extraction(path,
                                          dummy_year,
                                          dummy_month,
                                          dummy_day,
                                          groups)
    weights = np.zeros_like(temp_feat)

    # Target value (synthetic)
    target = 0.5

    # Error history buffer (empty at start)
    error_hist = np.array([], dtype=float)

    # Perform a single hybrid NLMS step
    new_weights, err = hybrid_nlms_step(weights,
                                        path,
                                        target,
                                        dummy_year,
                                        dummy_month,
                                        dummy_day,
                                        groups,
                                        mu_base=0.1,
                                        error_history=error_hist)

    # Show results
    print("Initial prediction:", hybrid_predict(path, weights,
                                                dummy_year,
                                                dummy_month,
                                                dummy_day,
                                                groups))
    print("Updated prediction :", hybrid_predict(path, new_weights,
                                                dummy_year,
                                                dummy_month,
                                                dummy_day,
                                                groups))
    print("Prediction error after update:", err)