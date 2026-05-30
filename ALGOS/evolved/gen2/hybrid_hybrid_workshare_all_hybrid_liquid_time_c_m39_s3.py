# DARWIN HAMMER — match 39, survivor 3
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:23:40Z

from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


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


def allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Split ``total_units`` into deterministic and LLM residual parts, then
    distribute the residual across ``groups`` using the weekday weight vector.
    Returns a dict mirroring the original schema with added calendar metadata.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)

    llm_per_group = llm_units * weight_vec
    share_pct_per_group = weight_vec * 100.0

    lanes = [
        {
            "group": grp,
            "llm_units": _pct(llm_per_group[i]),
            "llm_share_pct": _pct(share_pct_per_group[i]),
            "weekday_weight": _pct(weight_vec[i]),
        }
        for i, grp in enumerate(groups)
    ]

    jzloads: list[dict[str, Any]] = [
        {
            "kind": "OBJECT",
            "id": "project2501_hybrid_workshare_policy",
            "type": "workshare_policy",
            "deterministic_target_pct": _pct(deterministic_target_pct),
            "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        },
        {
            "kind": "EVENT",
            "id": "project2501_hybrid_workshare_allocation",
            "type": "allocation_computed",
            "total_units": _pct(total_units),
            "deterministic_units": _pct(deterministic_units),
            "llm_units": _pct(llm_units),
            "date": date.isoformat(),
            "weekday": dow,
            "lanes": lanes,
        },
    ]

    return {"allocation": jzloads, "weight_vector": weight_vec, "dow": dow}


# ----------------------------------------------------------------------
# MinHash utilities (parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def shingles(text: str, width: int = 5) -> set[str]:
    """Extract width‑word shingles from *text*."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}


# ----------------------------------------------------------------------
# LTC primitives (parent B)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element‑wise sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gating function f(x,I) = σ(W·[x;I] + b)."""
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)


# ----------------------------------------------------------------------
# Hybrid LTC‑MinHash‑Calendar step
# ----------------------------------------------------------------------
def hybrid_ltc_step(
    x: np.ndarray,
    A: np.ndarray,
    tau: float,
    W: np.ndarray,
    b: np.ndarray,
    s_t: float,
    alpha: float,
    beta: float,
    w_dow: np.ndarray,
) -> np.ndarray:
    """
    One integration step of the fused LTC-MinHash- Calendar dynamics.

    Args:
    - x: current state
    - A: learned attractor vector
    - tau: base liquid time constant
    - W: weight matrix for gating
    - b: bias for gating
    - s_t: MinHash similarity scalar
    - alpha: scalar mixing coefficient for MinHash term
    - beta: scalar mixing coefficient for calendar term
    - w_dow: weekday weight vector

    Returns:
    - next state
    """
    g_t = ltc_f(x, np.array([s_t]), W, b) + alpha * s_t + beta * w_dow
    dxdt = -(1.0 / tau + g_t) * x + g_t * A
    return x + dxdt * 0.01  # assuming a fixed time step of 0.01


def run_hybrid_process(
    texts: List[str],
    tau: float,
    alpha: float,
    beta: float,
    A: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    groups: Tuple[str, ...] = GROUPS,
):
    """
    Demonstrate the full pipeline on a sequence of texts.

    Args:
    - texts: list of input texts
    - tau: base liquid time constant
    - alpha: scalar mixing coefficient for MinHash term
    - beta: scalar mixing coefficient for calendar term
    - A: learned attractor vector
    - W: weight matrix for gating
    - b: bias for gating
    """
    states = []
    current_state = np.zeros(len(groups))
    current_date = dt.date.today()
    for text in texts:
        shingles_set = shingles(text)
        sig = minhash_signature(shingles_set)
        s_t = minhash_similarity(sig, minhash_signature(shingles(current_date.isoformat())))
        dow = doomsday(current_date.year, current_date.month, current_date.day)
        w_dow = weekday_weight_vector(groups, dow)
        current_state = hybrid_ltc_step(
            current_state,
            A,
            tau,
            W,
            b,
            s_t,
            alpha,
            beta,
            w_dow,
        )
        states.append(current_state)
        current_date += dt.timedelta(days=1)
    return states


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    texts = ["This is a test.", "Another test.", "And another one."]
    tau = 1.0
    alpha = 0.5
    beta = 0.5
    A = np.random.rand(4)
    W = np.random.rand(1, 8)
    b = np.random.rand(1)

    states = run_hybrid_process(texts, tau, alpha, beta, A, W, b)
    print(states)