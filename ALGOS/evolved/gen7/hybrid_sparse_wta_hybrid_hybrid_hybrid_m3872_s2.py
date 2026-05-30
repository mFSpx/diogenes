# DARWIN HAMMER — match 3872, survivor 2
# gen: 7
# parent_a: sparse_wta.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1645_s0.py (gen6)
# born: 2026-05-29T23:52:14Z

import numpy as np
import math
import random
import sys
from pathlib import Path
import hashlib
from typing import Tuple, List

GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    from datetime import date
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: Tuple[str], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy."""
    return -np.sum(p * np.log2(p))

def sign_quantisation(p: np.ndarray) -> np.ndarray:
    """Quantize probabilities to -1, 0, or 1."""
    return np.where(p > 0.5, 1, np.where(p < 0.5, -1, 0))

def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """
    Expand values into a larger space using a hash function.

    Args:
    values: List of values to expand.
    m: Size of the expanded space.
    salt: Optional salt for the hash function.

    Returns:
    List of expanded values.
    """
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> List[int]:
    """
    Compute a top-k mask.

    Args:
    values: List of values.
    k: Number of top values to select.

    Returns:
    List of 0s and 1s indicating top-k values.
    """
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]

def hamming(a: List[int], b: List[int]) -> int:
    """
    Compute Hamming distance.

    Args:
    a: List of 0s and 1s.
    b: List of 0s and 1s.

    Returns:
    Hamming distance between a and b.
    """
    if len(a) != len(b):
        raise ValueError('vectors must be same length')
    return sum(x != y for x, y in zip(a, b))

def hybrid_operation(weight_vec: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Modulate the effective liquid time constant based on the weekday weight vector.

    Args:
    weight_vec: Weekday weight vector.
    p: Probability distribution.

    Returns:
    Modulated probability distribution.
    """
    return np.multiply(weight_vec, p)

def hybrid_sparse_wta(values: List[float], m: int, salt: str = '', groups: Tuple[str] = GROUPS, dow: int = 0) -> List[float]:
    """
    Apply the sparse WTA algorithm with the hybrid operation.

    Args:
    values: List of values.
    m: Size of the expanded space.
    salt: Optional salt for the hash function.
    groups: Tuple of group names.
    dow: Day of the week.

    Returns:
    List of modulated probabilities.
    """
    weight_vec = weekday_weight_vector(groups, dow)
    expanded = expand(values, m, salt)
    p = np.array(expanded) / np.sum(expanded)
    hybrid_p = hybrid_operation(weight_vec, p)
    # Normalize hybrid_p to ensure it sums to 1
    return list(hybrid_p / hybrid_p.sum())

def hybrid_top_k_mask(values: List[float], k: int, groups: Tuple[str] = GROUPS, dow: int = 0) -> List[int]:
    """
    Apply the top-k mask algorithm with the hybrid operation.

    Args:
    values: List of values.
    k: Number of top values to select.
    groups: Tuple of group names.
    dow: Day of the week.

    Returns:
    List of 0s and 1s indicating top-k values.
    """
    weight_vec = weekday_weight_vector(groups, dow)
    p = np.array(values) / np.sum(values)
    hybrid_p = hybrid_operation(weight_vec, p)
    # Normalize hybrid_p to ensure it sums to 1
    return top_k_mask(list(hybrid_p / hybrid_p.sum()), k)

def hybrid_hamming(a: List[int], b: List[int], groups: Tuple[str] = GROUPS, dow: int = 0) -> int:
    """
    Apply the Hamming distance algorithm with the hybrid operation.

    Args:
    a: List of 0s and 1s.
    b: List of 0s and 1s.
    groups: Tuple of group names.
    dow: Day of the week.

    Returns:
    Hamming distance between a and b.
    """
    weight_vec = weekday_weight_vector(groups, dow)
    p_a = np.array(a) / np.sum(a)
    p_b = np.array(b) / np.sum(b)
    hybrid_p_a = hybrid_operation(weight_vec, p_a)
    hybrid_p_b = hybrid_operation(weight_vec, p_b)
    # Quantize hybrid_p_a and hybrid_p_b to 0s and 1s
    quantized_a = sign_quantisation(hybrid_p_a)
    quantized_b = sign_quantisation(hybrid_p_b)
    return hamming(list(quantized_a), list(quantized_b))

if __name__ == "__main__":
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    m = 5
    salt = 'example'
    groups = GROUPS
    dow = doomsday(2026, 5, 29)
    print(hybrid_sparse_wta(values, m, salt, groups, dow))
    print(hybrid_top_k_mask(values, 3, groups, dow))
    print(hybrid_hamming([1, 0, 1, 0, 1], [0, 1, 0, 1, 0], groups, dow))