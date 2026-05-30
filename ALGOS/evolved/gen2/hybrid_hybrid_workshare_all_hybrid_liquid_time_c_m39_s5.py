# DARWIN HAMMER — match 39, survivor 5
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:23:40Z

from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (Parent A)
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0 = Sunday … 6 = Saturday."""
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
# MinHash utilities (Parent B)
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
# LTC primitives (Parent B)
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
# Hybrid LTC‑MinHash‑Calendar core (Improved)
# ----------------------------------------------------------------------


def _validate_vector(vec: np.ndarray, name: str, expected_len: int) -> None:
    if vec.ndim != 1 or vec.shape[0] != expected_len:
        raise ValueError(f"{name} must be a 1‑D array of length {expected_len}")


def hybrid_ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    s_t: float,
    w_vec: np.ndarray,
    *,
    tau: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.2,
    dt_step: float = 0.01,
) -> np.ndarray:
    """
    Perform a single explicit Euler integration step of the fused LTC dynamics.

    The gating vector is now **intertwined** with calendar and MinHash signals:

        g_raw   = ltc_f(x, I, W, b)                     # base gating ∈ [0,1]^N
        g_mod   = g_raw * (1 + alpha * s_t) * (1 + beta * w_vec)
        τ_eff   = tau / (1 + gamma * w_vec)             # per‑group time‑constant

    The ODE (element‑wise) becomes:

        dx/dt = -(1/τ_eff + g_mod) * x + g_mod * A

    where ``A`` is a learned attractor (here taken as the bias vector ``b`` for
    simplicity).  The function returns the updated hidden state ``x_next``.
    """
    n = x.shape[0]

    _validate_vector(x, "x", n)
    _validate_vector(I, "I", I.shape[0])
    _validate_vector(b, "b", n)
    _validate_vector(w_vec, "w_vec", n)

    if not (0.0 <= s_t <= 1.0):
        raise ValueError("s_t must lie in [0, 1]")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if dt_step <= 0.0:
        raise ValueError("dt_step must be positive")

    # Base gating from LTC cell
    g_raw = ltc_f(x, I, W, b)  # shape (n,)

    # Fuse external signals multiplicatively (deeper integration)
    g_mod = g_raw * (1.0 + alpha * s_t) * (1.0 + beta * w_vec)

    # Per‑group effective time constant
    tau_eff = tau / (1.0 + gamma * w_vec)

    # Choose attractor A – using bias as a simple proxy (learnable)
    A = b

    # Explicit Euler step (element‑wise)
    dx = -(1.0 / tau_eff + g_mod) * x + g_mod * A
    x_next = x + dt_step * dx

    # Clamp to avoid numerical blow‑up (optional but practical)
    np.clip(x_next, -1e6, 1e6, out=x_next)

    return x_next


# ----------------------------------------------------------------------
# Higher‑level orchestration (Improved)
# ----------------------------------------------------------------------


def _prepare_ltc_parameters(
    hidden_dim: int,
    input_dim: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise LTC weight matrix ``W`` and bias ``b`` with a reproducible
    random generator.  ``W`` maps concatenated ``[x; I]`` (size hidden+input) to
    ``hidden_dim`` outputs.
    """
    total_in = hidden_dim + input_dim
    # Xavier‑like init for stability
    limit = math.sqrt(6.0 / total_in)
    W = np.array(
        [[rng.uniform(-limit, limit) for _ in range(total_in)] for _ in range(hidden_dim)],
        dtype=np.float64,
    )
    b = np.zeros(hidden_dim, dtype=np.float64)
    return W, b


def run_hybrid_process(
    texts: Sequence[str],
    *,
    hidden_dim: int = len(GROUPS),
    input_dim: int = 64,
    total_units: float = 1000.0,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.2,
    tau: float = 1.0,
    dt_step: float = 0.01,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    End‑to‑end demo:

    1. Allocate workshare (calendar side) → obtains weekday weight vector.
    2. Initialise LTC hidden state ``x`` and random input projection matrix.
    3. For each consecutive pair of texts compute:
       * MinHash similarity ``s_t``.
       * Input embedding ``I`` (here a simple random projection of token counts).
       * One hybrid LTC integration step.
    4. Return the list of hidden states for inspection.
    """
    rng = random.Random(seed)

    # 1. Calendar allocation
    alloc = allocate_hybrid(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=GROUPS,
    )
    w_vec = alloc["weight_vector"]  # shape (hidden_dim,)

    # 2. Initialise LTC parameters
    W, b = _prepare_ltc_parameters(hidden_dim, input_dim, rng)

    # Hidden state starts at zero
    x = np.zeros(hidden_dim, dtype=np.float64)

    # Pre‑compute MinHash signatures for the first text
    prev_sig = minhash_signature(shingles(texts[0]))

    states: List[np.ndarray] = [x.copy()]

    for idx in range(1, len(texts)):
        # ---- MinHash similarity -------------------------------------------------
        cur_sig = minhash_signature(shingles(texts[idx]))
        s_t = minhash_similarity(prev_sig, cur_sig)
        prev_sig = cur_sig

        # ---- Input embedding ----------------------------------------------------
        # Simple bag‑of‑words count vector, projected to ``input_dim`` via random matrix
        token_counts = {}
        for token in texts[idx].split():
            token_counts[token] = token_counts.get(token, 0) + 1
        # Random projection matrix (fixed for reproducibility)
        proj = np.array(
            [[rng.uniform(-0.1, 0.1) for _ in range(len(token_counts))] for _ in range(input_dim)],
            dtype=np.float64,
        )
        count_vec = np.array(list(token_counts.values()), dtype=np.float64)
        I = proj @ count_vec  # shape (input_dim,)

        # ---- Hybrid LTC step ----------------------------------------------------
        x = hybrid_ltc_step(
            x,
            I,
            W,
            b,
            s_t,
            w_vec,
            tau=tau,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            dt_step=dt_step,
        )
        states.append(x.copy())

    return states