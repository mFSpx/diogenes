# DARWIN HAMMER — match 39, survivor 4
# gen: 2
# parent_a: hybrid_workshare_allocator_doomsday_calendar_m14_s2.py (gen1)
# parent_b: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# born: 2026-05-29T23:23:40Z

import datetime as dt
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
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

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def shingles(text: str, width: int = 5) -> set[str]:
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)

def hybrid_ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    A: np.ndarray,
    tau: float,
    alpha: float,
    beta: float,
    dow: int,
    weight_vector: np.ndarray,
    s_t: float,
) -> np.ndarray:
    g_t = ltc_f(x, I, W, b) + alpha * s_t + beta * weight_vector
    dx_dt = -(1 / tau + g_t) * x + g_t * A
    return dx_dt

def run_hybrid_process(
    texts: List[str],
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    alpha: float = 0.1,
    beta: float = 0.1,
    tau: float = 1.0,
) -> None:
    allocation = allocate_hybrid(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
    )
    weight_vector = allocation["weight_vector"]
    dow = allocation["dow"]
    n = len(groups)
    x = np.random.rand(n)
    I = np.random.rand(n)
    W = np.random.rand(2 * n, n)
    b = np.random.rand(n)
    A = np.random.rand(n)
    for text in texts:
        tokens = shingles(text)
        sig = minhash_signature(tokens)
        if not hasattr(run_hybrid_process, "prev_sig"):
            run_hybrid_process.prev_sig = sig
        else:
            s_t = minhash_similarity(run_hybrid_process.prev_sig, sig)
            run_hybrid_process.prev_sig = sig
            x += hybrid_ltc_step(x, I, W, b, A, tau, alpha, beta, dow, weight_vector, s_t)
        print(x)

run_hybrid_process(["This is a test text.", "This is another test text."], 100.0, dt.date(2024, 9, 16))