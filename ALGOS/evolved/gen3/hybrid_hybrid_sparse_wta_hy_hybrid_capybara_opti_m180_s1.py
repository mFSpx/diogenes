# DARWIN HAMMER — match 180, survivor 1
# gen: 3
# parent_a: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# parent_b: hybrid_capybara_optimizatio_tri_algo_conduit_m55_s2.py (gen1)
# born: 2026-05-29T23:27:23Z

"""Hybrid Sparse‑WTA / Capybara‑Tri Conduit Algorithm.

Parents
-------
* **Parent A** – `sparse_wta.py` provides a hash‑based sparse expansion
  (`expand`) and a differential‑privacy aggregation (Laplace noise on the
  sum of the expanded vector).
* **Parent B** – `capybara_optimization.py` + `tri_algo_conduit.py` supplies
  an exponential evasion schedule (`evasion_delta`), a Hoeffding‑based
  confidence term and simple vector‑clamping utilities.

Mathematical Bridge
-------------------
1. An input record `v ∈ ℝⁿ` is first *sparsely projected* to a high‑dimensional
   space `e = expand(v, m) ∈ ℝᵐ`.  
2. The aggregate `S = Σ e_i` is perturbed with Laplace noise `η₁ ~ Lap(1/ε₁)`
   to obtain a differentially‑private sum `Ŝ`.  
3. A **privacy risk** `ρ = uqid / N` (unique quasi‑identifiers over total
   records) scales a second Laplace term `η₂ ~ Lap(ρ/ε₂)`.  
   The noisy, normalised vector `ê = (e + η₂·1_m) / ‖e + η₂·1_m‖` is the
   *query* for the optimisation stage.
4. The confidence scalar `c = 1 / (1 + ε_H)` where `ε_H` is the Hoeffding
   epsilon derived from the observation count.  
   This scalar rescales the evasion magnitude
   `δ(t) = evasion_delta(t, T, δ_max, α)·(1 + c)`.
5. The rescaled evasion step is applied only to the *top‑k* active
   components (WTA mask).  The update rule is

   `x_{t+1} = clamp( x_t + δ(t)·sign(ê)·mask, lower, upper )`

   where `mask` is the binary mask from `top_k_mask`.

The three functions below illustrate this fused pipeline:
* `expand_and_privacy` – sparse projection + DP noise + risk‑scaled noise.
* `hoeffding_epsilon` – computes the Hoeffding confidence term.
* `hybrid_update` – combines the above with the capybara evasion schedule
  to produce a privacy‑aware optimisation step.

All operations rely only on `numpy`, the Python standard library and `math`."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Sparse Winner‑Take‑All utilities
# ----------------------------------------------------------------------


def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`.

    Each input component contributes three signed entries chosen by a
    SHA‑256 hash of ``salt:i:r`` (i = index, r = repetition).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with 1 at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two equal‑length binary lists."""
    if len(a) != len(b):
        raise ValueError("vectors must be of equal length")
    return sum(1 for x, y in zip(a, b) if x != y)


# ----------------------------------------------------------------------
# Parent B – Capybara optimisation & Tri‑conduit utilities
# ----------------------------------------------------------------------


def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Sequence[float], lower: float, upper: float) -> List[float]:
    """Clamp each component of a vector to the interval [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


def _shannon_entropy(data: Sequence[int]) -> float:
    """Shannon entropy (bits) for a sequence of byte values."""
    if not data:
        return 0.0
    counts = np.bincount(np.asarray(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


# ----------------------------------------------------------------------
# Hybrid primitives
# ----------------------------------------------------------------------


def _laplace_noise(scale: float) -> float:
    """Draw a single Laplace(0, scale) sample."""
    # numpy implements Laplace with location=0
    return float(np.random.laplace(0.0, scale))


def expand_and_privacy(
    values: List[float],
    m: int,
    eps_privacy: float,
    uqid: int,
    total_records: int,
    salt: str = "",
) -> List[float]:
    """Sparse expansion followed by two Laplace perturbations.

    1. `e = expand(values, m, salt)`.
    2. Aggregate `S = sum(e)`.
    3. Add DP noise `η₁ ~ Lap(1/eps_privacy)` → `Ŝ = S + η₁`.
    4. Compute risk `ρ = uqid / total_records`.
    5. Add risk‑scaled noise `η₂ ~ Lap(ρ/eps_privacy)` to each component.
    6. Return the normalised noisy vector.
    """
    if eps_privacy <= 0:
        raise ValueError("eps_privacy must be positive")
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    e = np.array(expand(values, m, salt), dtype=np.float64)

    # DP‑noised aggregate (not used directly later, but kept for completeness)
    eta1 = _laplace_noise(1.0 / eps_privacy)
    _ = e.sum() + eta1  # placeholder for potential downstream use

    # Risk‑scaled componentwise noise
    risk = uqid / total_records
    scale_risk = risk / eps_privacy
    noise_vec = np.random.laplace(0.0, scale_risk, size=m)
    e_noisy = e + noise_vec

    norm = np.linalg.norm(e_noisy)
    if norm == 0:
        return [0.0] * m
    return (e_noisy / norm).tolist()


def hoeffding_epsilon(observations: int, delta: float = 0.05) -> float:
    """Hoeffding bound epsilon for a Bernoulli variable.

    ε = sqrt( ln(2/δ) / (2 * observations) )
    """
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must lie in (0,1)")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * observations))


def hybrid_update(
    values: List[float],
    m: int,
    k: int,
    eps_privacy: float,
    uqid: int,
    total_records: int,
    t: int,
    t_max: int,
    delta_max: float = 1.0,
    alpha: float = 3.0,
    observations: int = 100,
    lower: float = -10.0,
    upper: float = 10.0,
    salt: str = "",
) -> List[float]:
    """Perform one hybrid optimisation step.

    Steps
    -----
    1. Obtain a privacy‑preserving sparse representation via
       `expand_and_privacy`.
    2. Compute a top‑k mask on the *absolute* values.
    3. Derive a confidence scalar `c = 1 / (1 + ε_H)` where `ε_H`
       is the Hoeffding epsilon.
    4. Compute the evasion magnitude
       `δ = evasion_delta(t, t_max, delta_max, alpha) * (1 + c)`.
    5. Update the masked components:
       `x' = clamp( x + δ * sign(x) * mask, lower, upper )`
       where `x` is the noisy vector from step 1.
    """
    # 1. Privacy‑aware sparse projection
    x = np.array(
        expand_and_privacy(values, m, eps_privacy, uqid, total_records, salt),
        dtype=np.float64,
    )

    # 2. Top‑k mask on absolute values
    mask = np.array(top_k_mask(list(map(abs, x.tolist())), k), dtype=np.float64)

    # 3. Confidence from Hoeffding bound
    eps_H = hoeffding_epsilon(observations)
    confidence = 1.0 / (1.0 + eps_H)

    # 4. Rescaled evasion magnitude
    delta = evasion_delta(t, t_max, delta_max, alpha) * (1.0 + confidence)

    # 5. Apply update only on masked entries
    sign_vec = np.sign(x)
    update = delta * sign_vec * mask
    x_new = clamp(x + update, lower, upper)

    return x_new


def hybrid_score(
    vector: Sequence[float],
    observations: int,
    delta: float = 0.05,
) -> float:
    """Hybrid quality score mixing Hoeffding confidence with Shannon entropy.

    The score is defined as

        score = (1 - ε_H) * H(vector)

    where ε_H is the Hoeffding epsilon and H is the Shannon entropy of the
    quantised vector (8‑bit bins).
    """
    eps_H = hoeffding_epsilon(observations, delta)
    # Quantise to 256 bins for entropy estimation
    arr = np.asarray(vector, dtype=np.float64)
    # Normalise to [0,255] safely
    if arr.max() == arr.min():
        quant = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = (arr - arr.min()) / (arr.max() - arr.min())
        quant = (norm * 255).astype(np.uint8)
    entropy = _shannon_entropy(quant.tolist())
    return (1.0 - eps_H) * entropy


if __name__ == "__main__":
    # Simple smoke test exercising the hybrid pipeline
    random.seed(42)
    np.random.seed(42)

    # Synthetic input vector
    v = [random.uniform(-1, 1) for _ in range(8)]

    # Parameters
    m_dim = 128
    top_k = 10
    eps = 0.8
    uqid = 23
    total = 1000
    t_step = 5
    t_max = 20

    # Perform a hybrid update
    updated = hybrid_update(
        values=v,
        m=m_dim,
        k=top_k,
        eps_privacy=eps,
        uqid=uqid,
        total_records=total,
        t=t_step,
        t_max=t_max,
        delta_max=1.0,
        alpha=2.5,
        observations=250,
        lower=-5.0,
        upper=5.0,
        salt="test",
    )
    print("Updated vector (first 10 entries):", updated[:10])

    # Compute a hybrid score for the resulting vector
    score = hybrid_score(updated, observations=250)
    print("Hybrid quality score:", score)