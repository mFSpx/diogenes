# DARWIN HAMMER — match 39, survivor 2
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:23:40Z

"""Hybrid Workshare‑Calendar & Liquid‑Time‑Constant‑MinHash Network.

Parents
-------
* **Parent A** – `hybrid_workshare_allocator_doomsday_calendar_m14_s2.py`
  Provides a deterministic vs. LLM token split and a weekday‑dependent
  stochastic weight vector `w(dow)` that maps a residual pool onto a set of
  groups.

* **Parent B** – `hybrid_liquid_time_constant_minhash_m10_s2.py`
  Implements a Liquid‑Time‑Constant (LTC) recurrent cell whose gating function
  `f(x, I)` is modulated by a MinHash similarity scalar `s_t` derived from
  successive token‑set signatures.

Mathematical Bridge
-------------------
We treat the *group* dimension (size N) as the hidden state dimension of the
LTC cell.  The weekday weight vector `w(dow) ∈ ℝⁿ` (row‑stochastic) is used as
an *extrinsic* additive bias to the LTC gating, exactly like the MinHash
similarity term.  For a given time step *t* the combined gating becomes

    g_t = f(x_t, I_t) + α·s_t·1⃗ + β·w(dow)               (1)

where
* `f` is the learned sigmoid gating,
* `s_t ∈ [0,1]` is the MinHash similarity between signatures at *t‑1* and *t*,
* `α, β ≥ 0` are scalar mixing coefficients,
* `1⃗` is a vector of ones (broadcasted scalar `s_t`).

The LTC ODE is then

    dx/dt = -(1/τ + g_t)·x_t + g_t·A                     (2)

with `τ` the base liquid time constant and `A` a learned attractor vector.
Equation (2) together with (1) fuses the calendar topology (via `w`) and the
set‑similarity topology (via `s_t`) into a single unified dynamical system.

The module below implements:
* `weekday_weight_vector` – calendar side (parent A).
* `minhash_signature` / `minhash_similarity` – set‑similarity side (parent B).
* `ltc_f` – learned gating (parent B).
* `hybrid_ltc_step` – one integration step of the fused dynamics (new).
* `allocate_hybrid` – deterministic/LLM split with weekday weighting (parent A).
* `run_hybrid_process` – demonstrates the full pipeline on a sequence of texts.

All code is pure Python 3 with only the allowed standard‑library and NumPy
dependencies.
"""

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
    I: np.ndarray,
    prev_sig: List[int],
    cur_sig: List[int],
    weight_vec: np.ndarray,
    params: dict,
    alpha: float = 0.5,
    beta: float = 0.3,
    dt_: float = 0.1,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    One Euler integration step of the fused dynamics.

    Combined gating (Eq. 1):
        g = f(x, I) + α·s_t·1⃗ + β·w(dow)

    Returns
    -------
    x_new : np.ndarray
        Updated hidden state.
    tau_eff : float
        Mean effective liquid time constant after the step.
    cur_sig : List[int]
        Current MinHash signature (to be used as ``prev_sig`` next step).
    """
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    # Learned gating
    f_val = ltc_f(x, I, W, b)  # shape (hidden_dim,)

    # MinHash similarity term
    s_t = minhash_similarity(prev_sig, cur_sig) if prev_sig else 0.0
    s_vec = np.full_like(f_val, s_t)

    # Calendar weight term (already normalised, broadcast to hidden_dim)
    w_vec = weight_vec.astype(x.dtype)

    # Combined gating
    g = f_val + alpha * s_vec + beta * w_vec

    # ODE (Eq. 2)
    dx_dt = -(1.0 / tau + g) * x + g * A
    x_new = x + dt_ * dx_dt

    # Effective time constant per neuron, then averaged
    tau_eff_vec = tau / (1.0 + tau * g)
    tau_eff = float(np.mean(tau_eff_vec))

    return x_new, tau_eff, cur_sig


# ----------------------------------------------------------------------
# High‑level demonstration pipeline
# ----------------------------------------------------------------------


def run_hybrid_process(
    total_units: float,
    date: dt.date,
    texts: List[str],
    deterministic_target_pct: float = 90.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    dt_: float = 0.1,
) -> Dict[str, Any]:
    """
    1. Allocate LLM residuals across groups using the weekday weight vector.
    2. Initialise an LTC hidden state with those allocations (scaled to a
       convenient magnitude).
    3. Iterate over ``texts``; at each step compute a MinHash signature from
       shingles, then perform ``hybrid_ltc_step``.
    Returns a dictionary containing the allocation, the final hidden state,
    and a trace of effective time constants.
    """
    # ---- Step 1: allocation ----
    alloc = allocate_hybrid(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=GROUPS,
    )
    weight_vec = alloc["weight_vector"]  # shape (N,)

    # ---- Step 2: initialise LTC state ----
    hidden_dim = len(GROUPS)
    # Use the LLM units per group as a rough proxy for initial hidden activations
    init_llm_units = np.array(
        [lane["llm_units"] for lane in alloc["allocation"][1]["lanes"]], dtype=np.float64
    )
    # Scale to a small range to keep dynamics stable
    x = init_llm_units / max(1.0, np.max(init_llm_units))

    # Randomly initialise LTC parameters (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    input_dim = hidden_dim  # for simplicity we feed a same‑sized external input
    W = rng.normal(scale=0.5, size=(hidden_dim, hidden_dim + input_dim))
    b = rng.normal(scale=0.1, size=(hidden_dim,))
    tau = 1.0  # base liquid time constant
    A = rng.normal(scale=0.1, size=(hidden_dim,))

    params = {"W": W, "b": b, "tau": tau, "A": A}

    # ---- Step 3: iterate over texts ----
    prev_sig: List[int] = []  # empty for first step
    tau_trace: List[float] = []
    hidden_trace: List[np.ndarray] = []

    for txt in texts:
        token_set = shingles(txt, width=5)
        cur_sig = minhash_signature(token_set, k=64)

        # External input I: we simply reuse the weight vector as a placeholder
        I = weight_vec.copy()

        x, tau_eff, _ = hybrid_ltc_step(
            x,
            I,
            prev_sig,
            cur_sig,
            weight_vec,
            params,
            alpha=alpha,
            beta=beta,
            dt_=dt_,
        )
        tau_trace.append(tau_eff)
        hidden_trace.append(x.copy())
        prev_sig = cur_sig

    result = {
        "allocation": alloc,
        "final_hidden_state": x,
        "tau_trace": tau_trace,
        "hidden_trace": hidden_trace,
    }
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    today = dt.date.today()
    total = 1_000.0
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence drives modern software development.",
        "Hybrid algorithms combine strengths of distinct paradigms.",
        "Calendar-aware resource allocation can improve fairness.",
    ]

    out = run_hybrid_process(
        total_units=total,
        date=today,
        texts=sample_texts,
        deterministic_target_pct=85.0,
        alpha=0.6,
        beta=0.2,
        dt_=0.05,
    )

    # Simple verification prints
    print("Allocation summary:")
    for lane in out["allocation"]["allocation"][1]["lanes"]:
        print(
            f"  {lane['group']}: {lane['llm_units']} units "
            f"({lane['llm_share_pct']}% of LLM pool, weight {lane['weekday_weight']})"
        )
    print("\nEffective time constants per step:")
    for i, tau in enumerate(out["tau_trace"], 1):
        print(f"  Step {i}: τ_eff = {tau:.4f}")

    print("\nFinal hidden state (rounded):")
    print([round(v, 4) for v in out["final_hidden_state"]])