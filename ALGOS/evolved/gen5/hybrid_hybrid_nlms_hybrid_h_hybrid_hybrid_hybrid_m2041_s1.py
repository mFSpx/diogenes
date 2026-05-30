# DARWIN HAMMER — match 2041, survivor 1
# gen: 5
# parent_a: hybrid_nlms_hybrid_hybrid_worksh_m167_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s2.py (gen4)
# born: 2026-05-29T23:40:37Z

"""Hybrid NLMS‑LTC & Weekday‑Sheaf Algorithm

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Implements a Normalised Least‑Mean‑Squares (NLMS) weight update
  augmented by a Linear‑Time‑Constant (LTC) ordinary differential equation.
  Its gating term `g_t` depends on a *learned gating* scalar, a *minhash*
  similarity scalar and a *weekday weight* vector.

* **Parent B** – Provides a principled construction of the *weekday weight*
  vector via a sinusoidal rotation over a set of model groups, and a
  Structural‑Similarity (SSIM) metric for comparing two numeric sequences.

The mathematical bridge is the **weekday weight vector**:
Parent A treats it as an arbitrary vector, while Parent B generates it
systematically from the day‑of‑week and a list of groups.  The fused
algorithm therefore:

1. Generates `weekday_weight` with `weekday_weight_vector` (Parent B).
2. Computes the NLMS‑LTC update where the gating term `g_t` incorporates
   this generated vector (Parent A).
3. Optionally modulates the learning rate `mu` by the SSIM between the
   current prediction and the target, using the SSIM implementation from
   Parent B.

The resulting system retains the adaptive NLMS learning dynamics while
injecting structured temporal bias from the weekday‑based sheaf model,
and gains a similarity‑aware learning‑rate schedule.

Author: hybrid‑fusion generator
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import Tuple, Sequence, List, Optional

# ----------------------------------------------------------------------
# Constants (from Parent B)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (Parent B)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute the Structural Similarity (SSIM) index between two 1‑D signals.
    The implementation follows the classic SSIM formula for luminance,
    contrast and structure components.
    """
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    l = (2 * mx * my + c1) / (mx ** 2 + my ** 2 + c1)
    c = (2 * math.sqrt(vx) * math.sqrt(vy) + c2) / (vx + vy + c2)
    s = (cov + c2 / 2) / (math.sqrt(vx) * math.sqrt(vy) + c2 / 2)

    return l * c * s


# ----------------------------------------------------------------------
# Core hybrid algorithm (fusion of Parent A & B)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def generate_weekday_weight(
    groups: Sequence[str] = GROUPS,
    date_obj: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Produce the weekday weight vector expected by the NLMS‑LTC update.
    If ``date_obj`` is ``None`` the current local weekday is used.
    """
    if date_obj is None:
        dow = (np.datetime64('today').astype('datetime64[D]').astype(int) + 4) % 7  # 0=Sunday
    else:
        year, month, day = date_obj
        # Python's weekday(): Mon=0 … Sun=6 → shift to Sun=0
        dow = (np.datetime64(f'{year:04d}-{month:02d}-{day:02d}')
               .astype('datetime64[D]').astype(int) + 4) % 7
    return weekday_weight_vector(groups, dow)


def nlms_ltc_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    tau: float = 1.0,
    beta: float = 1.0,
    learned_gating: Optional[float] = None,
    minhash_similarity: Optional[float] = None,
    weekday_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Perform one NLMS step and simultaneously evaluate the LTC ODE term.
    The gating vector ``g_t`` follows Parent A but receives its weekday
    component from ``weekday_weight`` generated via Parent B.
    """
    if weights.shape != x.shape:
        raise ValueError("weights and input must have equal length")
    if not 0 < mu < 2:
        raise ValueError("mu must be in the interval (0, 2)")

    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    next_weights = weights + mu * error * x / power

    # ----- LTC ODE terms -----
    if learned_gating is None:
        learned_gating = np.clip(y, 0.0, 1.0)  # scalar gating
    if minhash_similarity is None:
        minhash_similarity = random.random()
    if weekday_weight is None:
        weekday_weight = np.random.uniform(0, 1, size=weights.shape)

    # Broadcasting scalar components to vector shape
    g_t = learned_gating + minhash_similarity + beta * weekday_weight
    g_t = np.clip(g_t, 0.0, 1.0)

    # dx/dt = -(1/τ + g_t) * x + g_t * ξ ,  ξ ~ Uniform[0,1]
    noise = np.random.uniform(0.0, 1.0, size=weights.shape)
    dxdt = -(1.0 / tau + g_t) * x + g_t * noise

    return next_weights, error, dxdt


def hybrid_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
    tau: float = 1.0,
    beta: float = 1.0,
    groups: Sequence[str] = GROUPS,
    date_obj: Optional[Tuple[int, int, int]] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    High‑level update that:
    1. Generates a weekday weight vector via ``weekday_weight_vector`` (Parent B).
    2. Computes an SSIM‑based similarity between the current prediction and
       the target (treated as length‑1 sequences for compatibility).
    3. Scales the learning rate ``mu`` by the similarity, giving larger steps
       when prediction and target are already close.
    4. Calls ``nlms_ltc_update`` with the generated weekday weight.
    """
    # Step 1: weekday bias
    w_weight = generate_weekday_weight(groups, date_obj)

    # Step 2: similarity‑aware mu
    pred = predict(weights, x)
    # For SSIM we need sequences; replicate scalar to length‑2 vectors
    ssim = compute_ssim([pred, pred], [target, target])
    mu_adj = mu * (0.5 + 0.5 * ssim)  # keep mu in (0, mu*1)

    # Step 3+4: NLMS‑LTC core update
    next_w, err, dxdt = nlms_ltc_update(
        weights,
        x,
        target,
        mu=mu_adj,
        eps=eps,
        tau=tau,
        beta=beta,
        learned_gating=None,
        minhash_similarity=None,
        weekday_weight=w_weight,
    )
    return next_w, err, dxdt


def hybrid_step(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    **kwargs,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Convenience wrapper performing a single hybrid iteration.
    Returns updated weights, instantaneous error, and the LTC derivative.
    """
    return hybrid_update(weights, x, target, **kwargs)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    dim = 8
    w = np.random.randn(dim)
    x = np.random.randn(dim)
    target_val = 0.7

    # Run a few hybrid steps to ensure no exceptions
    for i in range(3):
        w, err, dx = hybrid_step(w, x, target_val, mu=0.6, tau=0.9, beta=0.8)
        print(f"Step {i+1}: error={_pct(err)}, weight_norm={_pct(np.linalg.norm(w))}")
    print("Hybrid NLMS‑LTC + Weekday‑Sheaf test completed successfully.")