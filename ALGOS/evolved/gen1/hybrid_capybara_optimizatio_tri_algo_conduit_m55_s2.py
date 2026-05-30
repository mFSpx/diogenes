# DARWIN HAMMER — match 55, survivor 2
# gen: 1
# parent_a: capybara_optimization.py (gen0)
# parent_b: tri_algo_conduit.py (gen0)
# born: 2026-05-29T23:23:56Z

"""Hybrid Capybara‑Tri Conduit Algorithm.

This module fuses the continuous optimisation primitives of *capybara_optimization.py*
(social interaction, predator evasion, exponential evasion schedule) with the
statistical gating logic of *tri_algo_conduit.py* (signal/noise scoring,
Hoeffding‑tree split decision, recovery priority).

Mathematical bridge
------------------
* The signal‑to‑noise gap `Δ = signal - noise` is interpreted as a
  confidence scalar.  It rescales the random coefficient `r` used in
  `social_interaction` and the step size `δ` used in `predator_evasion`.
* The Hoeffding epsilon `ε` (derived from the observations count) is
  combined with the exponential evasion schedule `δ(t)` to produce a
  hybrid evasion magnitude `δ_h = δ(t) * (1 + ε)`.
* The Hoeffding gain gap `gain_gap` drives the magnitude of the
  attraction towards the global best `g_best` and also modulates the
  probability of entering *standby* versus *burst*.

The three core functions below implement this fused behaviour.

"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

Vector = Sequence[float]


# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule for evasion magnitude."""
    if t < 0 or t_max <= 0 or delta_max < 0 or alpha < 0:
        raise ValueError("invalid evasion schedule")
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)


def clamp(x: Vector, lower: float, upper: float) -> list[float]:
    """Clamp each component of a vector to [lower, upper]."""
    return [min(upper, max(lower, xi)) for xi in x]


# ----------------------------------------------------------------------
# Minimal re‑implementation of Parent B primitives
# ----------------------------------------------------------------------
def _shannon_entropy(data: Sequence[int]) -> float:
    """Return Shannon entropy in bits for a sequence of byte values."""
    if not data:
        return 0.0
    counts = np.bincount(np.array(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -np.sum(probs * np.log2(probs))


def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    chunk = data[:sample]
    return _shannon_entropy(list(chunk)) / 8.0  # normalise to [0,1]


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    """Bounded (signal, noise) scores for an ingress candidate."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(
        0.0,
        min(
            1.0,
            0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


@dataclass(frozen=True)
class HoeffdingSplit:
    should_split: bool
    gain_gap: float
    epsilon: float
    reason: str


def hoeffding_split(
    signal: float,
    noise: float,
    range_r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.03,
) -> HoeffdingSplit:
    """Conservative Hoeffding split test."""
    gain_gap = signal - noise
    # Hoeffding bound (two‑class Bernoulli case)
    epsilon = math.sqrt((range_r ** 2 * math.log(1.0 / delta)) / (2.0 * max(1, n)))
    should = (gain_gap > epsilon) and (gain_gap > tie_threshold)
    reason = "gain>eps" if should else "gain<=eps"
    return HoeffdingSplit(should, gain_gap, epsilon, reason)


def acceptance_probability(noise_minus_signal: float, temperature: float) -> float:
    """Logistic acceptance probability used for standby (Thanatosis analogue)."""
    # Clamp temperature to avoid overflow
    temp = max(1e-6, temperature)
    return 1.0 / (1.0 + math.exp(-temp * (-noise_minus_signal)))


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def recovery_priority(morph: Morphology, max_index: float = 12.0) -> float:
    """Simple proxy for serpentina recovery priority."""
    raw = morph.mass / (morph.length + morph.width + morph.height + 1e-9)
    # Normalise to [0,1] using a sigmoid scaled by max_index
    return 1.0 / (1.0 + math.exp(- (raw - max_index) / max_index))


def recovery_from_payload(data: bytes, max_bytes: int, parse_error: bool = False) -> float:
    size_ratio = min(4.0, len(data) / max(1, max_bytes))
    morph = Morphology(
        length=1.0 + size_ratio * 8.0,
        width=2.0 + (2.0 if parse_error else 0.5),
        height=max(0.5, 3.0 - size_ratio),
        mass=1.0 + size_ratio * 6.0 + (3.0 if parse_error else 0.0),
    )
    return recovery_priority(morph, max_index=12.0)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_social_interaction(
    x: Vector,
    g_best: Vector,
    confidence: float,
    k: int = 1,
    seed: int | str | None = None,
) -> np.ndarray:
    """
    Scaled version of `social_interaction` where the random coefficient `r`
    is biased by the confidence scalar `c = 0.5 + 0.5*confidence` (confidence in [0,1]).
    """
    if len(x) != len(g_best):
        raise ValueError("dimension mismatch")
    if k not in (1, 2):
        raise ValueError("k must be 1 or 2")
    rng = random.Random(seed)
    base_r = rng.random()
    r = 0.5 + 0.5 * confidence * (2 * base_r - 1)  # map confidence -> tighter range around 0.5
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)
    result = x_arr + r * (g_arr - k * x_arr)
    return result


def hybrid_predator_evasion(
    x: Vector,
    delta_t: float,
    gain_gap: float,
    seed: int | str | None = None,
) -> np.ndarray:
    """
    Combine the exponential evasion magnitude `delta_t` with the Hoeffding
    gain gap `gain_gap`.  The step size becomes `step = (2r-1) * delta_t * (1+gain_gap)`.
    """
    if delta_t < 0:
        raise ValueError("delta_t must be non‑negative")
    rng = random.Random(seed)
    r = rng.random()
    step = (2 * r - 1) * delta_t * (1.0 + max(0.0, gain_gap))
    x_arr = np.asarray(x, dtype=float)
    return x_arr + step * x_arr


def hybrid_decide(
    x: Vector,
    g_best: Vector,
    data: bytes,
    observations: int,
    t: int,
    t_max: int,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    max_bytes: int = 1_500_000,
    delta: float = 1e-5,
    range_r: float = 1.0,
    tie_threshold: float = 0.03,
    standby_temperature: float = 0.35,
    parse_error: bool = False,
    k: int = 1,
    seed: int | str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Perform a full hybrid step:

    1. Compute signal / noise scores.
    2. Run Hoeffding split test.
    3. Derive a hybrid evasion magnitude `δ_h = evasion_delta(t) * (1+ε)`.
    4. Apply predator evasion.
    5. Apply confidence‑scaled social interaction toward `g_best`.
    6. Clamp the result to the search space [−10, 10] (arbitrary choice).

    Returns the updated position vector and a textual decision tag
    (\"recover\", \"burst\", or \"standby\").
    """
    # 1. Scores
    signal, noise = signal_scores(data, status_code, mime, keyword_hits, structural_links)

    # 2. Hoeffding decision
    split = hoeffding_split(signal, noise, range_r, delta, observations, tie_threshold)

    # 3. Hybrid evasion magnitude
    base_delta = evasion_delta(t, t_max)
    delta_h = base_delta * (1.0 + split.epsilon)

    # 4. Predator evasion
    after_evasion = hybrid_predator_evasion(x, delta_h, split.gain_gap, seed=seed)

    # 5. Confidence‑scaled social interaction
    confidence = max(0.0, min(1.0, signal - noise + 0.5))  # map gap to [0,1] roughly
    after_social = hybrid_social_interaction(after_evasion, g_best, confidence, k=k, seed=seed)

    # 6. Clamp to a reasonable bound
    new_pos = np.clip(after_social, -10.0, 10.0)

    # Decision tag
    if parse_error or recovery_from_payload(data, max_bytes, parse_error) >= 0.82:
        decision = "recover"
    elif split.should_split and signal >= noise:
        decision = "burst"
    else:
        # standby probability derived from Thanatosis analogue
        dormancy = 1.0 - acceptance_probability(noise - signal, standby_temperature)
        decision = "standby" if dormancy > 0.5 else "burst"

    return new_pos, decision


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic test
    rng = np.random.default_rng(42)
    dim = 5
    x0 = rng.uniform(-5, 5, size=dim).tolist()
    g_best = rng.uniform(-5, 5, size=dim).tolist()

    # Dummy payload (random bytes)
    payload = bytes(rng.integers(0, 256, size=1024).tolist())

    new_vec, tag = hybrid_decide(
        x=x0,
        g_best=g_best,
        data=payload,
        observations=30,
        t=10,
        t_max=100,
        status_code=200,
        mime="application/json",
        keyword_hits=3,
        structural_links=2,
        seed=123,
    )

    print("Initial vector :", x0)
    print("Global best    :", g_best)
    print("Updated vector :", new_vec.tolist())
    print("Decision tag   :", tag)
    sys.exit(0)