# DARWIN HAMMER — match 1454, survivor 4
# gen: 5
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s5.py (gen3)
# parent_b: hybrid_hybrid_rlct_grokking_hybrid_hybrid_regret_m1149_s0.py (gen4)
# born: 2026-05-29T23:36:37Z

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

def weekday_index(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7

def encode_weekday(idx: int) -> np.ndarray:
    vec = np.zeros(7, dtype=float)
    if 0 <= idx < 7:
        vec[idx] = 1.0
    else:
        raise ValueError(f"weekday index out of range: {idx}")
    return vec

def compute_rlct(error_history: deque[float]) -> float:
    if not error_history:
        return 0.0
    arr = np.array(list(error_history), dtype=float)
    mean = np.mean(np.abs(arr))
    std = np.std(np.abs(arr))
    return std / (mean + 1e-12)

def minhash_signature(data: bytes, num_perm: int = 32) -> np.ndarray:
    sig = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        h = hashlib.sha256(data + i.to_bytes(4, "little")).digest()
        sig[i] = int.from_bytes(h[:8], "little")
    return sig

def ternary_vector(seed: int, length: int = 7) -> np.ndarray:
    rng = random.Random(seed)
    vec = np.empty(length, dtype=float)
    for i in range(length):
        vec[i] = rng.choice([-1.0, 0.0, 1.0])
    return vec

def shannon_entropy(state: np.ndarray) -> float:
    symbols = np.rint(state).astype(int)
    counts = Counter(symbols)
    total = len(symbols)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    y_pred = nlms_predict(weights, x)
    error = target - y_pred
    norm_sq = np.dot(x, x) + eps
    weights_new = weights + (mu / norm_sq) * error * x
    return weights_new, error

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
    w_idx = weekday_index(*date)
    w_onehot = encode_weekday(w_idx)

    tern_vec = ternary_vector(seed=w_idx, length=7)
    rlct = compute_rlct(error_hist)
    mu_R = base_mu / (1 + rlct_lambda * rlct)

    if ref_signature is None:
        ref_signature = np.zeros_like(tern_vec)
    sim = np.dot(tern_vec, ref_signature) / (np.linalg.norm(tern_vec) * np.linalg.norm(ref_signature))
    regret = 1 - sim
    rho = 1 - regret_eps * regret
    H = shannon_entropy(tern_vec)
    mu_eff = mu_R * (1 + entropy_alpha * H) * rho

    x_aug = np.concatenate((cont_features, w_onehot, tern_vec))
    y_pred = nlms_predict(weights, x_aug)
    new_weights, error = nlms_update(weights, x_aug, target, mu_eff)
    error_hist.append(abs(error))
    return y_pred, new_weights, error