# DARWIN HAMMER — match 4348, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1370_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m2224_s0.py (gen6)
# born: 2026-05-29T23:55:14Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_fusion.py

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1370_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m2224_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a *weekday weight vector* `w ∈ ℝ^G` that encodes
temporal (day‑of‑week) and epistemic certainty information.
Algorithm B operates on *multivectors* `M` in a geometric algebra where
basis blades are identified by frozensets of feature indices.
The fusion treats the weight vector `w` as a scalar field that linearly
scales the components of a multivector built from the data.  The scaled
multivector `w ⊙ M` is then combined with a geometric product to obtain
a new multivector whose magnitude encodes a geometrically‑informed
candidate split quality.  Finally the Hoeffding bound from Algorithm A
gives a statistical confidence interval for that quality, yielding a
single unified decision metric.

The module implements:
1. `weekday_weight_vector` – temporal/epistemic weighting (A).
2. `Multivector` with geometric product – algebraic core (B).
3. `geometric_weighted_product` – fuses (1) and (2).
4. `nlms_update` – adaptive weight update (B) applied to the weekday
   weights.
5. `hybrid_split_score` – end‑to‑end computation of a split score using
   the above pieces together with the Hoeffding bound.
"""

import math
import random
import sys
import pathlib
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Constants from Algorithm A
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

# ----------------------------------------------------------------------
# Helper functions from Algorithm A
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to 6 decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a deterministic day‑of‑week index (0=Monday,…,6=Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: List[str], dow: int, epistemic_flags: List[str]) -> np.ndarray:
    """
    Produce a probability vector of length len(groups) that varies sinusoidally
    with the day‑of‑week and is modulated by epistemic certainty flags.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    # Map each flag to a weight in [0,1]
    epistemic_weights = np.array(
        [EPISTEMIC_FLAGS.index(flag) / len(EPISTEMIC_FLAGS) for flag in epistemic_flags]
    )
    # Broadcast to match groups length
    if epistemic_weights.size != n:
        # Pad or truncate to length n
        if epistemic_weights.size < n:
            pad = np.full(n - epistemic_weights.size, epistemic_weights.mean())
            epistemic_weights = np.concatenate([epistemic_weights, pad])
        else:
            epistemic_weights = epistemic_weights[:n]
    raw = 1.0 + amplitude * np.sin(base_angles + phase) * epistemic_weights
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a random variable bounded in [0, r]."""
    if n <= 0:
        raise ValueError("n must be positive")
    return math.sqrt((r ** 2 * math.log(2.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Geometric Algebra core from Algorithm B
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple multivector implementation where each component is identified
    by a frozenset of basis indices.  For example, the scalar part is
    `frozenset()` and a vector component e_i is `frozenset({i})`.
    """

    def __init__(self, components: Dict[frozenset, float] = None):
        self.components: Dict[frozenset, float] = components if components else {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components.copy())
        for blade, val in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + val
        return result

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result = Multivector()
        for blade_a, val_a in self.components.items():
            for blade_b, val_b in other.components.items():
                new_blade, sign = _geometric_product_blades(blade_a, blade_b)
                result.components[new_blade] = result.components.get(new_blade, 0.0) + sign * val_a * val_b
        return result

    def norm(self) -> float:
        """Euclidean norm of the multivector (square root of sum of squares)."""
        return math.sqrt(sum(v * v for v in self.components.values()))

    def __repr__(self) -> str:
        return f"Multivector({self.components})"


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Sort the list of indices using bubble sort while counting the parity
    of swaps.  Returns the sorted list and the sign (+1 or -1) of the permutation.
    """
    lst = indices[:]
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
    return lst, sign


def _geometric_product_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """
    Compute the geometric product of two basis blades.
    The result is another blade (as a frozenset) together with a sign.
    Duplicate indices cancel (e_i * e_i = 1) and are removed.
    """
    combined = list(blade_a) + list(blade_b)
    # Count occurrences to cancel duplicates
    counts: Dict[int, int] = {}
    for idx in combined:
        counts[idx] = counts.get(idx, 0) + 1
    remaining = [idx for idx, cnt in counts.items() if cnt % 2 == 1]

    # The sign is the parity of the swaps needed to bring the remaining indices
    # into sorted order.
    sorted_remaining, sign = _blade_sign(remaining)
    return frozenset(sorted_remaining), sign

# ----------------------------------------------------------------------
# NLMS adaptive update (Algorithm B)
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray,
                input_vec: np.ndarray,
                desired: float,
                mu: float = 0.01,
                eps: float = 1e-6) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares update.
    weights ← weights + (mu / (||x||² + eps)) * e * x
    where e = desired − w·x
    """
    if weights.shape != input_vec.shape:
        raise ValueError("weights and input_vec must have the same shape")
    norm_sq = np.dot(input_vec, input_vec) + eps
    error = desired - np.dot(weights, input_vec)
    correction = (mu / norm_sq) * error * input_vec
    return weights + correction

# ----------------------------------------------------------------------
# Fusion primitives
# ----------------------------------------------------------------------
def geometric_weighted_product(weight_vec: np.ndarray, mv: Multivector) -> Multivector:
    """
    Scale each vector‑grade component of `mv` by the corresponding entry of
    `weight_vec` and then compute the geometric product of the resulting
    scaled multivector with the original one.
    This yields a new multivector that embeds the temporal/epistemic
    weighting into the geometric algebra space.
    """
    if weight_vec.ndim != 1:
        raise ValueError("weight_vec must be a 1‑D array")
    # Build a scaling multivector S = Σ_i w_i e_i
    scaling_components = {frozenset({i}): weight_vec[i] for i in range(len(weight_vec))}
    scaling_mv = Multivector(scaling_components)

    # Geometric product: (S) ⊙ (mv)
    return scaling_mv * mv

def data_to_multivector(data: np.ndarray) -> Multivector:
    """
    Convert a 2‑D data array (samples × features) into a multivector.
    For each feature index i we store the mean value across samples as the
    coefficient of basis blade e_i.
    """
    if data.ndim != 2:
        raise ValueError("data must be a 2‑D array")
    means = data.mean(axis=0)
    components = {frozenset({i}): float(means[i]) for i in range(means.size)}
    return Multivector(components)

def hybrid_split_score(data: np.ndarray,
                       labels: np.ndarray,
                       groups: List[str],
                       today: date,
                       epistemic_flags: List[str],
                       delta: float = 0.05) -> Dict[str, Any]:
    """
    End‑to‑end hybrid metric for a candidate decision‑tree split.

    Steps:
    1. Build a weekday weight vector w using groups, day‑of‑week and flags.
    2. Convert the feature matrix into a multivector M (means per feature).
    3. Form the weighted geometric product G = geometric_weighted_product(w, M).
    4. Use the norm of G as a raw split quality score Q.
    5. Apply the Hoeffding bound to the label distribution to obtain a confidence ε.
    6. Return a dictionary with the raw score, confidence interval and
       an NLMS‑updated weight vector (pretending the desired quality is Q).
    """
    # 1. Weight vector
    dow = today.weekday()  # Monday=0
    w = weekday_weight_vector(groups, dow, epistemic_flags)

    # 2. Data → multivector
    M = data_to_multivector(data)

    # 3. Weighted geometric product
    G = geometric_weighted_product(w, M)

    # 4. Raw quality (norm)
    raw_quality = G.norm()

    # 5. Hoeffding confidence on labels (assume binary 0/1)
    n = int(labels.size)
    r = 1.0  # range of binary variable
    epsilon = hoeffding_bound(r, delta, n)

    # 6. NLMS update of the weight vector using raw_quality as the desired value.
    #    Treat the mean of each feature as the input vector for the update.
    input_vec = np.array([M.components.get(frozenset({i}), 0.0) for i in range(len(w))])
    updated_w = nlms_update(w, input_vec, desired=raw_quality)

    return {
        "weekday_weight_vector": w,
        "multivector": M,
        "weighted_product": G,
        "raw_quality": raw_quality,
        "hoeffding_epsilon": epsilon,
        "confidence_interval": (raw_quality - epsilon, raw_quality + epsilon),
        "updated_weight_vector": updated_w,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data: 100 samples, 4 features (matching GROUPS length)
    np.random.seed(42)
    synthetic_data = np.random.rand(100, len(GROUPS))
    synthetic_labels = (synthetic_data[:, 0] + synthetic_data[:, 2] > 1.0).astype(int)

    # Random epistemic flags (one per group)
    random_flags = [random.choice(EPISTEMIC_FLAGS) for _ in GROUPS]

    result = hybrid_split_score(
        data=synthetic_data,
        labels=synthetic_labels,
        groups=list(GROUPS),
        today=date.today(),
        epistemic_flags=random_flags,
        delta=0.05,
    )

    # Simple sanity prints (no external libraries)
    print("Weekday weight vector:", result["weekday_weight_vector"])
    print("Raw quality (norm):", _pct(result["raw_quality"]))
    print("Hoeffding epsilon:", _pct(result["hoeffding_epsilon"]))
    print("Confidence interval:", ( _pct(result["confidence_interval"][0]), _pct(result["confidence_interval"][1]) ))
    print("Updated weight vector (first 4 values):", result["updated_weight_vector"][:4])
    print("Multivector components (sample):", list(result["multivector"].components.items())[:4])