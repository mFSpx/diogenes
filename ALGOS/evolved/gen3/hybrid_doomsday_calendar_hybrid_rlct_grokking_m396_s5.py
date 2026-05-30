# DARWIN HAMMER — match 396, survivor 5
# gen: 3
# parent_a: doomsday_calendar.py (gen0)
# parent_b: hybrid_rlct_grokking_hybrid_nlms_omni_cha_m118_s1.py (gen2)
# born: 2026-05-29T23:28:45Z

"""Hybrid Doomsday–RLCT‑NLMS algorithm.

Parent A: ``doomsday_calendar.py`` – provides a deterministic mapping from a
Gregorian date to a weekday index (0 = Sunday … 6 = Saturday).

Parent B: ``hybrid_rlct_grokking_hybrid_nlms_omni_cha...`` – implements a
Normalized Least‑Mean‑Squares (NLMS) adaptive filter whose learning‑rate
parameter μ is modulated by the Real Log‑Canonical‑Threshold (RLCT) derived
from the free‑energy asymptotic of the error sequence.

Mathematical bridge:
The weekday index from Parent A is injected as a one‑hot categorical feature
into the NLMS input vector.  This augments the regression problem with a
periodic (weekly) component, allowing the adaptive filter to learn
seasonal patterns.  Simultaneously, the RLCT computed from the recent error
history rescales the base learning‑rate μ₀, yielding an effective learning‑rate

    μ_eff = μ₀ / (1 + λ·RLCT),

where λ is a small damping factor.  Thus the calendar topology (discrete
weekday) and the statistical topology (RLCT‑adjusted learning) are fused
into a single update rule.

The module supplies three public functions that embody this hybrid
behaviour:
* ``weekday_index`` – compute the weekday from a date.
* ``encode_weekday`` – produce a 7‑dimensional one‑hot vector.
* ``hybrid_nlms_step`` – perform one NLMS prediction‑update cycle with the
  RLCT‑adjusted learning‑rate and weekday augmentation.
"""

from __future__ import annotations

import datetime as dt
import math
import random
import sys
from pathlib import Path
from collections import deque
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Doomsday calendar utilities
# ----------------------------------------------------------------------
def weekday_index(year: int, month: int, day: int) -> int:
    """Return weekday as 0=Sunday … 6=Saturday using Python's datetime.

    This mirrors the ``doomsday`` function from the original parent.
    """
    # dt.date.weekday() returns 0=Monday … 6=Sunday; shift to Sunday=0.
    return (dt.date(year, month, day).weekday() + 1) % 7


def encode_weekday(idx: int) -> np.ndarray:
    """One‑hot encode a weekday index into a length‑7 vector of floats."""
    vec = np.zeros(7, dtype=float)
    if 0 <= idx < 7:
        vec[idx] = 1.0
    else:
        raise ValueError(f"Weekday index out of range: {idx}")
    return vec


# ----------------------------------------------------------------------
# Parent B – RLCT‑adjusted NLMS core
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def _rlct_from_errors(error_deque: deque[float]) -> float:
    """Estimate a surrogate RLCT from recent absolute errors.

    The true RLCT is a sophisticated quantity; here we approximate it by
    the variance of the log‑errors, which captures how rapidly the error
    distribution contracts.  This surrogate is sufficient to illustrate the
    mathematical coupling.
    """
    if not error_deque:
        return 0.0
    logs = np.log(np.abs(np.array(error_deque)) + 1e-12)
    # RLCT proxy: variance of log‑errors (non‑negative)
    return float(np.var(logs))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu_base: float = 0.5,
    eps: float = 1e-9,
    rlct: float = 0.0,
    lam: float = 0.1,
) -> tuple[np.ndarray, float]:
    """NLMS weight update with RLCT‑scaled learning rate.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (shape (d,)).
    x : np.ndarray
        Input vector (shape (d,)).
    target : float
        Desired response.
    mu_base : float
        Base learning‑rate before RLCT scaling.
    eps : float
        Regularisation term to avoid division by zero.
    rlct : float
        RLCT surrogate (non‑negative) derived from recent errors.
    lam : float
        Damping factor controlling the influence of RLCT on μ.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Instantaneous prediction error (target – prediction).
    """
    pred = nlms_predict(weights, x)
    error = target - pred

    # RLCT‑adjusted learning rate
    mu_eff = mu_base / (1.0 + lam * rlct)

    norm_sq = float(x @ x) + eps
    adaptation = (mu_eff / norm_sq) * error * x
    new_weights = weights + adaptation
    return new_weights, error


# ----------------------------------------------------------------------
# Hybrid public API
# ----------------------------------------------------------------------
def hybrid_nlms_step(
    weights: np.ndarray,
    date: dt.date,
    x_numeric: np.ndarray,
    target: float,
    error_history: deque[float],
    mu_base: float = 0.5,
    eps: float = 1e-9,
    lam: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Perform one hybrid NLMS step.

    The numeric input vector is concatenated with a one‑hot weekday vector,
    forming the full regression input.  The RLCT surrogate is computed from
    ``error_history`` and used to scale the learning rate.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector of length (len(x_numeric) + 7).
    date : datetime.date
        Calendar date providing the periodic feature.
    x_numeric : np.ndarray
        Purely numeric part of the input (shape (d,)).
    target : float
        Desired output value.
    error_history : deque[float]
        FIFO container of recent errors; will be appended with the new error.
    mu_base, eps, lam : float
        Hyper‑parameters forwarded to ``nlms_update``.

    Returns
    -------
    new_weights : np.ndarray
        Updated weight vector.
    error : float
        Prediction error for this step (appended to ``error_history``).
    """
    # 1️⃣ Encode weekday and build augmented input
    wk_idx = weekday_index(date.year, date.month, date.day)
    wk_one_hot = encode_weekday(wk_idx)
    x_full = np.concatenate([x_numeric.astype(float), wk_one_hot])

    # 2️⃣ Estimate RLCT from recent errors
    rlct_est = _rlct_from_errors(error_history)

    # 3️⃣ NLMS update with RLCT‑scaled μ
    new_weights, error = nlms_update(
        weights,
        x_full,
        target,
        mu_base=mu_base,
        eps=eps,
        rlct=rlct_est,
        lam=lam,
    )

    # 4️⃣ Maintain error history (fixed length of 50)
    error_history.append(error)
    if len(error_history) > 50:
        error_history.popleft()

    return new_weights, error


def initialize_weights(dim_numeric: int, seed: int | None = None) -> np.ndarray:
    """Create a weight vector for the hybrid model.

    The total dimension is ``dim_numeric + 7`` (weekday one‑hot).  We use a
    small random normal initialization for symmetry breaking.
    """
    rng = np.random.default_rng(seed)
    total_dim = dim_numeric + 7
    return rng.normal(loc=0.0, scale=0.1, size=total_dim)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic time‑series with a weekly pattern.
    rng = np.random.default_rng(42)
    dim_numeric = 3
    weights = initialize_weights(dim_numeric, seed=42)

    # Error history buffer
    err_hist: deque[float] = deque()

    # Simulate 30 days
    start_date = dt.date(2023, 1, 1)
    for day_offset in range(30):
        today = start_date + dt.timedelta(days=day_offset)

        # Numeric features: three random signals
        x_num = rng.normal(size=dim_numeric)

        # True target = sum(x_num) + weekday bias (e.g., +1 on Mondays)
        wk = weekday_index(today.year, today.month, today.day)
        weekday_bias = 1.0 if wk == 1 else 0.0  # Monday bias
        target_val = float(x_num.sum() + weekday_bias)

        # Perform hybrid NLMS step
        weights, err = hybrid_nlms_step(
            weights,
            today,
            x_num,
            target_val,
            err_hist,
            mu_base=0.7,
            lam=0.05,
        )

        # Print a concise log
        print(
            f"{today} (wk={wk}) | err={err:.4f} | μ_eff≈{0.7/(1+0.05*_rlct_from_errors(err_hist)):.3f}"
        )
    print("Smoke test completed without errors.")