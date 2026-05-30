# DARWIN HAMMER — match 3248, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2057_s1.py (gen5)
# born: 2026-05-29T23:50:06Z

"""
Hybrid Entropic‑MinHash‑SSIM Strike (HEMSS)

This module fuses the two parent algorithms:

* **Parent A** – Entropic MinHash (infotaxis + minhash).  It provides
  entropy of a probability distribution, a MinHash signature for a list
  of probabilities and a binary differential hash (d‑hash) whose Hamming
  similarity is interpreted as a physical “force”.

* **Parent B** – Ternary‑router + SSIM + weekday‑weight vector.  It
  supplies a structural weight vector that depends on the day‑of‑week,
  and a Structural‑Similarity‑Index‑Metric (SSIM) that measures the
  similarity of two numeric streams.

**Mathematical bridge**

Both parents output a *similarity* between two objects:

* `hamming_similarity` – normalised Hamming similarity of the d‑hashes
  derived from MinHash signatures.
* `ssim_similarity` – SSIM of the original probability vectors (or any
  derived streams).

We combine them linearly, modulated by a weekday‑dependent weight vector
`w(dow)` from Parent B and by the expected entropy reduction from
Parent A.  The resulting scalar is used as a *force* in a simple
drag‑limited Newtonian integration (the “strike” dynamics of Parent A).

The three core functions below illustrate the hybrid operation:

1. `combined_similarity` – builds MinHash signatures, d‑hashes, computes
   both similarity measures and fuses them.
2. `drag_limited_integration` – integrates a particle under the
   combined force with linear drag.
3. `hybrid_strike` – end‑to‑end pipeline that takes two probability
   distributions and a weekday index, and returns the final
   `StrikeState`.

The resulting system can be used for information‑theoretic search,
clustering, or any task where similarity of probability distributions
must be evaluated under a physical cost model that is aware of temporal
(weekday) context.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – entropy, MinHash and d‑hash utilities
# ----------------------------------------------------------------------


def entropy(probabilities: List[float]) -> float:
    """Shannon entropy of a probability distribution."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if np.any(probs < 0):
        raise ValueError("probabilities must be non‑negative")
    total = probs.sum()
    if total == 0:
        raise ValueError("sum of probabilities must be positive")
    probs = probs / total
    # avoid log(0) by masking zero entries
    mask = probs > 0
    return -np.sum(probs[mask] * np.log2(probs[mask]))


def _hash_int(value: int, seed: int = 0) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    h = hashlib.sha256((str(value) + str(seed)).encode())
    return int.from_bytes(h.digest()[:8], "big")


def entropic_minhash(probabilities: List[float], k: int = 64) -> List[int]:
    """
    Build a MinHash signature of length ``k`` from a probability distribution.
    The i‑th component is the index of the smallest hash value after weighting
    by the probability.
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1‑D list/array")
    n = probs.shape[0]
    signature = []
    for i in range(k):
        min_hash = None
        min_idx = -1
        for idx in range(n):
            # weight the hash by the probability (larger prob → smaller hash)
            h = _hash_int(idx, seed=i)
            weighted = int(h / (probs[idx] + 1e-12))
            if (min_hash is None) or (weighted < min_hash):
                min_hash = weighted
                min_idx = idx
        signature.append(min_idx)
    return signature


def differential_hash(signature: List[int]) -> int:
    """
    Convert a MinHash signature to a binary differential hash (d‑hash).
    The i‑th bit is 1 iff s_i > s_{i+1}.
    Returns the integer representation of the (k‑1)‑bit pattern.
    """
    bits = 0
    for i in range(len(signature) - 1):
        bits = (bits << 1) | int(signature[i] > signature[i + 1])
    return bits


def hamming_similarity(dhash1: int, dhash2: int, bits: int) -> float:
    """
    Normalised Hamming similarity in [0, 1] for two d‑hashes of length ``bits``.
    """
    xor = dhash1 ^ dhash2
    distance = bin(xor).count("1")
    return 1.0 - distance / bits


# ----------------------------------------------------------------------
# Parent B – weekday weight vector and SSIM utilities
# ----------------------------------------------------------------------


GROUPS = ("codex", "groq", "cohere", "local_models")


def weekday_weight_vector(groups: List[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    ``dow`` follows the convention 0 = Sunday … 6 = Saturday.
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
    Structural Similarity Index (SSIM) for two 1‑D signals.
    Returns a value in [-1, 1] (theoretical range for arbitrary signals).
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

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid core – combine similarities and run strike dynamics
# ----------------------------------------------------------------------


def combined_similarity(
    prob_a: List[float],
    prob_b: List[float],
    k: int = 64,
    dow: int = 0,
    groups: List[str] | None = None,
) -> Tuple[float, dict]:
    """
    Compute a fused similarity measure between two probability distributions.

    Steps
    -----
    1. Build MinHash signatures and d‑hashes.
    2. Compute normalised Hamming similarity ``h_sim``.
    3. Compute SSIM similarity ``s_sim`` on the raw probability vectors.
    4. Compute a weekday‑dependent scalar ``w`` from ``weekday_weight_vector``.
    5. Fuse them: ``force = w * (α * h_sim + β * s_sim)`` where
       ``α`` = expected entropy reduction and ``β`` = 1‑α.

    Returns
    -------
    force : float
        The scalar that will be interpreted as a force.
    info : dict
        Diagnostic information (signatures, hashes, components).
    """
    if groups is None:
        groups = list(GROUPS)

    # 1. MinHash signatures
    sig_a = entropic_minhash(prob_a, k=k)
    sig_b = entropic_minhash(prob_b, k=k)

    # 2. d‑hashes
    dh_a = differential_hash(sig_a)
    dh_b = differential_hash(sig_b)
    bits = k - 1
    h_sim = hamming_similarity(dh_a, dh_b, bits)

    # 3. SSIM on the probability vectors
    s_sim = compute_ssim(prob_a, prob_b, dynamic_range=1.0)

    # 4. Weekday weight (use first component as a scalar proxy)
    w_vec = weekday_weight_vector(groups, dow)
    w = w_vec[0]  # scalar modulation; can be replaced by any aggregation

    # 5. Entropy‑based weighting
    ent_a = entropy(prob_a)
    ent_b = entropy(prob_b)
    # Expected entropy reduction if we moved from a to b (symmetrised)
    delta_ent = abs(ent_a - ent_b) / max(ent_a, ent_b, 1e-12)
    alpha = delta_ent  # in [0,1]
    beta = 1.0 - alpha

    force = w * (alpha * h_sim + beta * s_sim)

    info = {
        "signature_a": sig_a,
        "signature_b": sig_b,
        "dhash_a": dh_a,
        "dhash_b": dh_b,
        "hamming_similarity": h_sim,
        "ssim_similarity": s_sim,
        "weekday_weight": w,
        "entropy_a": ent_a,
        "entropy_b": ent_b,
        "entropy_weight_alpha": alpha,
        "combined_force": force,
    }
    return force, info


@dataclass
class StrikeState:
    """State of the particle after the strike integration."""
    position: float
    velocity: float
    time: float


def drag_limited_integration(
    force: float,
    mass: float = 1.0,
    drag_coeff: float = 0.1,
    dt: float = 0.01,
    steps: int = 500,
) -> StrikeState:
    """
    Simple 1‑D Newtonian integration with linear drag.

    The equation of motion:
        m * a = F - c * v
    where ``c`` is the drag coefficient.
    """
    v = 0.0
    x = 0.0
    t = 0.0
    for _ in range(steps):
        a = (force - drag_coeff * v) / mass
        v += a * dt
        x += v * dt
        t += dt
    return StrikeState(position=x, velocity=v, time=t)


def hybrid_strike(
    prob_a: List[float],
    prob_b: List[float],
    dow: int = 0,
    groups: List[str] | None = None,
    k: int = 64,
    mass: float = 1.0,
    drag_coeff: float = 0.1,
    dt: float = 0.01,
    steps: int = 500,
) -> Tuple[StrikeState, dict]:
    """
    End‑to‑end hybrid operation.

    Returns a tuple ``(final_state, diagnostics)`` where
    ``final_state`` is a :class:`StrikeState` and ``diagnostics`` contains the
    intermediate similarity components.
    """
    force, diag = combined_similarity(
        prob_a, prob_b, k=k, dow=dow, groups=groups
    )
    state = drag_limited_integration(
        force,
        mass=mass,
        drag_coeff=drag_coeff,
        dt=dt,
        steps=steps,
    )
    diag["final_state"] = state
    return state, diag


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two random probability distributions of length 10
    rng = np.random.default_rng(seed=42)
    p1 = rng.random(10).tolist()
    p2 = rng.random(10).tolist()

    # Normalise them
    p1 = (np.asarray(p1) / np.sum(p1)).tolist()
    p2 = (np.asarray(p2) / np.sum(p2)).tolist()

    # Use today's weekday (0=Sunday … 6=Saturday)
    from datetime import datetime

    today_dow = (datetime.utcnow().weekday() + 1) % 7

    final_state, diagnostics = hybrid_strike(p1, p2, dow=today_dow)

    print("Final strike state:")
    print(f"  position = {final_state.position:.6f}")
    print(f"  velocity = {final_state.velocity:.6f}")
    print(f"  time     = {final_state.time:.6f}")

    print("\nDiagnostics:")
    for key, val in diagnostics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")