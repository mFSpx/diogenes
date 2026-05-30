# DARWIN HAMMER — match 1454, survivor 3
# gen: 5
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s5.py (gen3)
# parent_b: hybrid_hybrid_rlct_grokking_hybrid_hybrid_regret_m1149_s0.py (gen4)
# born: 2026-05-29T23:36:37Z

"""Hybrid Doomsday–RLCT‑NLMS‑MinHash‑Ternary Algorithm
====================================================

Parents
-------
* **Parent A** – ``doomsday_calendar.py``  
  Supplies a deterministic mapping from a Gregorian date to a weekday index
  (0 = Sunday … 6 = Saturday) and a one‑hot encoder for that index.

* **Parent B** – ``hybrid_hybrid_rlct_grokking_hybrid_hybrid_regret_m1149_s0.py``  
  Provides a Real Log‑Canonical‑Threshold (RLCT) estimator, a Normalized
  Least‑Mean‑Squares (NLMS) adaptive filter and a MinHash‑ternary “lens”
  whose Shannon entropy drives a regret‑weighted exploration factor.

Mathematical Bridge
-------------------
Both parents expose a *scalar temperature* that modulates learning:

* From **Parent A** the weekday index is injected as a categorical (periodic)
  feature, augmenting the NLMS input space.
* From **Parent B** the RLCT estimate rescales the NLMS base step‑size
  ``μ₀`` → ``μ_R = μ₀ / (1 + λ·RLCT)``.
* The MinHash‑ternary lens yields a discrete hybrid state ``h``.
  Its Shannon entropy ``H(h)`` controls a regret‑weighted exploration factor
  ``ρ = 1 – ε·regret`` where ``regret = 1 – sim(σ,σ_ref)`` and ``ε`` is a
  small constant.

The unified temperature is the product of the two modulators:


μ_eff = μ_R * (1 + α·H(h)) * ρ


where ``α`` balances information‑theoretic influence.  The NLMS update uses
the *augmented* input vector


x̃ = [x_continuous , one_hot(weekday) , ternary_vector]


Thus the calendar topology (discrete weekday), the geometric topology
(RLCT) and the informational topology (MinHash‑ternary entropy) are
mathematically fused into a single adaptive step.

The module below implements this hybrid system with three public
functions that showcase the combined behaviour:
* ``weekday_index`` / ``encode_weekday`` – calendar utilities.
* ``compute_rlct`` – RLCT estimator from an error history.
* ``hybrid_nlms_step`` – NLMS prediction‑update that incorporates the
  weekday one‑hot, the ternary lens, the MinHash entropy and the RLCT‑scaled
  learning‑rate.  The function returns the prediction, the updated weight
  vector and the new error value."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import sys
from collections import deque, Counter
from pathlib import Path
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Calendar utilities (Parent A)
# ----------------------------------------------------------------------


def weekday_index(year: int, month: int, day: int) -> int:
    """Return weekday as 0=Sunday … 6=Saturday.

    Python's ``datetime.weekday()`` yields 0=Monday … 6=Sunday, so we shift.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


def encode_weekday(idx: int) -> np.ndarray:
    """One‑hot encode a weekday index into a length‑7 float vector."""
    vec = np.zeros(7, dtype=float)
    if 0 <= idx < 7:
        vec[idx] = 1.0
    else:
        raise ValueError(f"weekday index out of range: {idx}")
    return vec


# ----------------------------------------------------------------------
# RLCT estimator (Parent B – simplified)
# ----------------------------------------------------------------------


def compute_rlct(error_history: deque[float]) -> float:
    """Estimate the Real Log‑Canonical‑Threshold from recent absolute errors.

    A full RLCT computation requires asymptotic free‑energy analysis.
    Here we use a cheap proxy: the coefficient of variation of the
    absolute error sequence.  Larger variability → larger RLCT.
    """
    if not error_history:
        return 0.0
    arr = np.array(list(error_history), dtype=float)
    mean = np.mean(np.abs(arr))
    std = np.std(np.abs(arr))
    # Avoid division by zero; add a tiny epsilon.
    return std / (mean + 1e-12)


# ----------------------------------------------------------------------
# MinHash & ternary lens (Parent B)
# ----------------------------------------------------------------------


def minhash_signature(data: bytes, num_perm: int = 32) -> np.ndarray:
    """Compute a simple MinHash‑like signature.

    For each permutation ``i`` we hash ``data || i`` with SHA‑256 and keep the
    integer value of the first 8 bytes.  The result is a ``num_perm``‑dimensional
    integer vector.
    """
    sig = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        h = hashlib.sha256(data + i.to_bytes(4, "little")).digest()
        sig[i] = int.from_bytes(h[:8], "little")
    return sig


def ternary_vector(seed: int, length: int = 7) -> np.ndarray:
    """Generate a deterministic ternary vector (values -1, 0, +1)."""
    rng = random.Random(seed)
    vec = np.empty(length, dtype=float)
    for i in range(length):
        vec[i] = rng.choice([-1.0, 0.0, 1.0])
    return vec


def shannon_entropy(state: np.ndarray) -> float:
    """Shannon entropy of a discrete state vector.

    The vector is first quantised to integer symbols, then the usual
    ``-∑ p·log₂ p`` is computed.
    """
    # Quantise to a small set of symbols.
    symbols = np.rint(state).astype(int)
    counts = Counter(symbols)
    total = len(symbols)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


# ----------------------------------------------------------------------
# NLMS core (Parent B)
# ----------------------------------------------------------------------


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """Perform a single NLMS adaptation step.

    Returns the updated weight vector and the instantaneous error ``e = d - ŷ``.
    """
    y_pred = nlms_predict(weights, x)
    error = target - y_pred
    norm_sq = np.dot(x, x) + eps
    weights_new = weights + (mu / norm_sq) * error * x
    return weights_new, error


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------


def hybrid_nlms_step(
    weights: np.ndarray,
    cont_features: np.ndarray,
    date: Tuple[int, int, int],
    target: float,
    error_hist: deque[float],
    *,
    base_mu: float = 0.5,
    rlct_lambda: float = 0.1,
    entropy_alpha: float = 0.3,
    regret_eps: float = 0.2,
    ref_signature: np.ndarray | None = None,
) -> Tuple[float, np.ndarray, float]:
    """One hybrid NLMS step that merges calendar, RLCT and MinHash‑ternary logic.

    Parameters
    ----------
    weights
        Current NLMS weight vector (size must match the augmented input).
    cont_features
        Continuous feature vector (e.g. sensor readings) – shape ``(d,)``.
    date
        ``(year, month, day)`` tuple used to obtain the weekday one‑hot.
    target
        Desired scalar output.
    error_hist
        Deque holding recent errors; will be appended with the new error.
    base_mu, rlct_lambda, entropy_alpha, regret_eps
        Hyper‑parameters controlling the influence of each component.
    ref_signature
        Reference MinHash signature for regret computation; if ``None`` a
        zero‑vector of appropriate length is used.

    Returns
    -------
    y_pred
        The NLMS prediction before the update.
    new_weights
        Updated weight vector.
    new_error
        Instantaneous error after the prediction (``target - y_pred``).
    """
    # 1️⃣ Calendar augmentation -------------------------------------------------
    w_idx = weekday_index(*date)
    w_onehot = encode_weekday(w_idx)

    # 2️⃣ Ternary lens ---------------------------------------------------------
    # Use the weekday index as a deterministic seed for reproducibility.
    tern_vec = ternary_vector(seed=w_idx, length=7)

    # 3️⃣ MinHash signature ----------------------------------------------------
    # Build a byte representation from the date and continuous features.
    raw = (
        f"{date[0]:04d}{date[1]:02d}{date[2]:02d}"
        + "".join(f"{v:.6f}" for v in cont_features)
    ).encode("utf-8")
    sig = minhash_signature(raw, num_perm=16)

    # 4️⃣ Regret factor --------------------------------------------------------
    if ref_signature is None:
        ref_signature = np.zeros_like(sig)
    # Simple similarity: proportion of equal hash entries.
    similarity = np.mean(sig == ref_signature)
    regret = 1.0 - similarity
    regret_factor = 1.0 - regret_eps * regret  # ≤ 1, reduces step when regret is high.

    # 5️⃣ Entropy of the hybrid discrete state ---------------------------------
    hybrid_state = np.concatenate([sig.astype(float), tern_vec])
    entropy = shannon_entropy(hybrid_state)

    # 6️⃣ RLCT‑scaled learning rate --------------------------------------------
    rlct = compute_rlct(error_hist)
    mu_rlct = base_mu / (1.0 + rlct_lambda * rlct)

    # 7️⃣ Final effective learning rate ----------------------------------------
    mu_eff = mu_rlct * (1.0 + entropy_alpha * entropy) * regret_factor

    # 8️⃣ Build the augmented NLMS input vector --------------------------------
    x_aug = np.concatenate([cont_features, w_onehot, tern_vec])

    # 9️⃣ NLMS prediction and update -------------------------------------------
    y_pred, error = nlms_predict(weights, x_aug), target - nlms_predict(weights, x_aug)
    new_weights, new_error = nlms_update(weights, x_aug, target, mu_eff)

    # 10️⃣ Update error history -------------------------------------------------
    error_hist.append(abs(new_error))
    # Keep the deque at a reasonable length (e.g. 100).
    if len(error_hist) > 100:
        error_hist.popleft()

    return y_pred, new_weights, new_error


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Simple synthetic scenario: predict a noisy sinusoid with weekly seasonality.
    rng = np.random.default_rng(seed=42)

    # Continuous feature: a single scalar (time index)
    def make_features(t: int) -> np.ndarray:
        return np.array([math.sin(0.1 * t)], dtype=float)

    # Initialise NLMS weights for: 1 cont + 7 weekday + 7 ternary = 15 dimensions.
    dim = 1 + 7 + 7
    w = np.zeros(dim, dtype=float)

    # Error history for RLCT estimation.
    err_hist: deque[float] = deque(maxlen=100)

    # Reference MinHash signature (fixed for the test).
    reference_sig = minhash_signature(b"reference", num_perm=16)

    for step in range(50):
        t = step
        x_cont = make_features(t)
        # Use an arbitrary date that cycles weekly.
        date = (2026, 1, (t % 28) + 1)  # ensures valid day numbers.
        # True target: sinusoid + weekday bias.
        true_weekday = weekday_index(*date)
        target = math.sin(0.1 * t) + 0.2 * (true_weekday - 3)  # bias centred at 0.
        # Add Gaussian noise.
        target += rng.normal(scale=0.05)

        y, w, err = hybrid_nlms_step(
            weights=w,
            cont_features=x_cont,
            date=date,
            target=target,
            error_hist=err_hist,
            base_mu=0.8,
            rlct_lambda=0.05,
            entropy_alpha=0.2,
            regret_eps=0.1,
            ref_signature=reference_sig,
        )

        if step % 10 == 0:
            print(
                f"step={step:02d}  pred={y: .3f}  target={target: .3f}  err={err: .3f}"
            )
    print("Smoke test completed without errors.")