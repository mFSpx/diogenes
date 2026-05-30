# DARWIN HAMMER — match 3448, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_label_foundry_m1273_s0.py (gen5)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_hybrid_hybrid_m734_s0.py (gen4)
# born: 2026-05-29T23:50:13Z

"""
Hybrid Regret‑Weighted Ternary‑Decision Analyzer with Doomsday‑Gini‑MinHash Fusion (RW‑TD‑DG‑MH)

Parents:
- Parent A: Hybrid Regret‑Weighted Ternary‑Decision Analyzer with Path Signature Pruning and
  Label‑Foundry weak supervision.
- Parent B: Doomsday weekday calculation + Gini inequality coefficient + MinHash signature.

Mathematical bridge:
1. Regret‑weighted probabilities are discretised into a ternary alphabet (A, B, C).
2. Each ternary symbol is timestamped by a calendar date; the Doomsday algorithm maps the
   date to a weekday (0‑6). This yields a (symbol, weekday) pair.
3. For every weekday we count occurrences of each ternary symbol, forming a count vector.
   The Gini coefficient of this vector quantifies the inequality of symbol distribution
   across the week.
4. The full ternary sequence is fed to a MinHash generator, producing a compact signature
   that captures similarity between different decision traces.
5. A weak‑supervision labeling function combines the Gini score and the MinHash signature
   to emit a final classification.

The three core functions below demonstrate the fused pipeline.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent‑B utilities (trimmed to allowed imports)
# ----------------------------------------------------------------------
def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Vectorised Doomsday weekday calculation.
    Returns an array of weekday indices where 0 = Sunday … 6 = Saturday.
    """
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    # Shift so that 0 = Sunday as in the original spec
    return (py_weekday + 1) % 7


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array.
    """
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = sorted_vals.size
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_vals)) / (n * sum_vals)
    return gini


def _hash(seed: int, token: str) -> int:
    """
    Simple deterministic hash mixing a seed with a token.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(4, byteorder="little", signed=False))
    h.update(token.encode('utf-8'))
    # Return unsigned 64‑bit integer
    return int.from_bytes(h.digest(), byteorder="little", signed=False)


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Generate a MinHash signature of length k from a list of string tokens.
    """
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not token_set:
        return [2 ** 63 - 1] * k
    signature = []
    for i in range(k):
        min_hash = min(_hash(i, t) for t in token_set)
        signature.append(min_hash)
    return signature


# ----------------------------------------------------------------------
# Parent‑A core: regret‑weighted → ternary mapping
# ----------------------------------------------------------------------
def map_probabilities_to_ternary(probabilities: np.ndarray) -> str:
    """
    Map regret‑weighted probabilities to a ternary alphabet.
    - p < 1/3   → 'A'
    - 1/3 ≤ p < 2/3 → 'B'
    - p ≥ 2/3  → 'C'
    Returns the concatenated string.
    """
    ternary_chars = []
    for p in probabilities.flat:
        if p < 1.0 / 3.0:
            ternary_chars.append('A')
        elif p < 2.0 / 3.0:
            ternary_chars.append('B')
        else:
            ternary_chars.append('C')
    return ''.join(ternary_chars)


# ----------------------------------------------------------------------
# Hybrid pipeline functions
# ----------------------------------------------------------------------
def compute_symbol_weekday_counts(ternary_seq: str, weekdays: np.ndarray) -> np.ndarray:
    """
    Build a count matrix of shape (7, 3) where rows are weekdays (0‑6) and columns are
    ternary symbols (A, B, C). The entry (w, s) is the number of times symbol s appears
    on weekday w.
    """
    if len(ternary_seq) != weekdays.size:
        raise ValueError('Length of ternary sequence must match number of weekdays')
    count_matrix = np.zeros((7, 3), dtype=int)
    symbol_to_idx = {'A': 0, 'B': 1, 'C': 2}
    for sym, wd in zip(ternary_seq, weekdays.flat):
        idx = symbol_to_idx.get(sym)
        if idx is not None:
            count_matrix[int(wd), idx] += 1
    return count_matrix


def hybrid_gini_minhash_score(probabilities: np.ndarray,
                              years: np.ndarray,
                              months: np.ndarray,
                              days: np.ndarray,
                              minhash_k: int = 64) -> Tuple[float, List[int]]:
    """
    End‑to‑end hybrid score:
    1. Convert probabilities → ternary sequence.
    2. Compute weekday indices via Doomsday.
    3. Count symbol occurrences per weekday and derive a Gini coefficient.
    4. Generate a MinHash signature of the ternary sequence.
    Returns (gini_score, minhash_signature).
    """
    ternary_seq = map_probabilities_to_ternary(probabilities)
    weekdays = doomsday_numpy(years, months, days)
    counts = compute_symbol_weekday_counts(ternary_seq, weekdays)
    # Flatten counts to a 1‑D vector for Gini
    gini_score = gini_coefficient(counts.ravel())
    mh_sig = minhash_signature(list(ternary_seq), k=minhash_k)
    return gini_score, mh_sig


def weak_label_from_hybrid(gini_score: float, minhash_sig: List[int],
                           gini_threshold: float = 0.3,
                           mh_distance_threshold: int = 1_000_000) -> str:
    """
    Simple weak‑supervision labeling function.
    - If Gini > threshold → 'needs_conversion'
    - Else if the median of the MinHash signature is large → 'research_only'
    - Otherwise → 'usable_now'
    """
    if gini_score > gini_threshold:
        return "needs_conversion"
    median_hash = int(np.median(minhash_sig))
    if median_hash > mh_distance_threshold:
        return "research_only"
    return "usable_now"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic batch of 100 decision traces
    n = 100
    # Regret‑weighted probabilities (uniform random)
    probs = np.random.rand(n)

    # Random dates between 2000‑01‑01 and 2025‑12‑31
    start = datetime(2000, 1, 1).timestamp()
    end = datetime(2025, 12, 31).timestamp()
    timestamps = np.random.uniform(start, end, size=n)
    dates = np.array([datetime.utcfromtimestamp(ts) for ts in timestamps])

    years = np.array([d.year for d in dates], dtype=int)
    months = np.array([d.month for d in dates], dtype=int)
    days = np.array([d.day for d in dates], dtype=int)

    gini, mh = hybrid_gini_minhash_score(probs, years, months, days, minhash_k=64)
    label = weak_label_from_hybrid(gini, mh)

    print(f"Gini coefficient: {gini:.4f}")
    print(f"MinHash signature (first 8 values): {mh[:8]}")
    print(f"Weak‑supervision label: {label}")