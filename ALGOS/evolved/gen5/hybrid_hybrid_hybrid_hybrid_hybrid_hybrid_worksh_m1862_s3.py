# DARWIN HAMMER — match 1862, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s4.py (gen4)
# parent_b: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s1.py (gen2)
# born: 2026-05-29T23:39:12Z

"""
Hybrid Algorithm integrating:
- Parent A: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_fracti_m222_s4.py (DARWIN HAMMER — match 222, survivor 4)
- Parent B: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s1.py (DARWIN HAMMER — match 39, survivor 1)

The mathematical bridge is the integration of the fractional power hypervector binding 
from Parent A into the learned gating function of the Liquid-Time-Constant network 
from Parent B. The MinHash signature is turned into a hyper-vector via a random complex 
hypervector generator and fractionally powered with an exponent derived from the 
weekday-dependent weight vector. The resulting scalar modulates the effective liquid 
time constant, allowing the algorithm to adapt to changing inputs and weekdays.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import date

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

# ----------------------------------------------------------------------
# Calendar helper
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7

def weekday_weight_vector(groups: tuple, dow: int) -> np.ndarray:
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

# ----------------------------------------------------------------------
# MinHash and hypervector utilities
# ----------------------------------------------------------------------
def minhash_for_text(text: str, k: int = 64) -> list[int]:
    """Create a k‑length MinHash signature of *text*."""
    cleaned = text.replace(" ", "").strip().lower()
    shingles = [cleaned[i : i + 5] for i in range(len(cleaned) - 4)]
    signature = [sys.maxsize] * k
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h % 1_000_000)
    return signature

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hyper‑vector.

    Parameters
    ----------
    d: dimension of the hyper‑vector.
    kind: "complex", "bipolar" or "real".
    seed: optional RNG seed.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    # "real"
    vec = rng.normal(size=d)
    return vec / np.linalg.norm(vec)

def fractional_power(vec: np.ndarray, power: float) -> np.ndarray:
    """Apply a fractional power to a hypervector."""
    return np.power(vec, power)

# ----------------------------------------------------------------------
# Liquid-Time-Constant network with fractional power modulation
# ----------------------------------------------------------------------
def liquid_time_constant(gating: float, minhash: list[int], hv_dim: int = 10000) -> np.ndarray:
    """Simulate a Liquid-Time-Constant network with a MinHash-modulated gating function."""
    hv = random_hv(hv_dim)
    minhash_hv = np.array([random_hv().dot(np.array(minhash))])
    gating_value = gating * np.abs(minhash_hv)
    return fractional_power(hv, gating_value)

def hybrid_liquid_time_constant(text: str, dow: int) -> np.ndarray:
    """Fuse the weekday-dependent weight vector with the MinHash-modulated Liquid-Time-Constant network."""
    weight_vec = weekday_weight_vector(GROUPS, dow)
    minhash_sig = minhash_for_text(text)
    hv = random_hv()
    powered_hv = fractional_power(hv, weight_vec.dot(np.array(minhash_sig)) / len(minhash_sig))
    return powered_hv

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def test_liquid_time_constant():
    text = "This is a test sentence."
    dow = doomsday(2024, 1, 1)
    result = hybrid_liquid_time_constant(text, dow)
    print(result)

def test_fractional_power():
    hv = random_hv()
    power = 0.5
    result = fractional_power(hv, power)
    print(result)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_liquid_time_constant()
    test_fractional_power()