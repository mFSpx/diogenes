# DARWIN HAMMER — match 4021, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m905_s2.py (gen4)
# born: 2026-05-29T23:53:17Z

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Set, Tuple, FrozenSet
import numpy as np
from dataclasses import dataclass, asdict
from itertools import combinations

# ----------------------------------------------------------------------
# Core constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant
ALPHA: float = 5.0             # gating steepness
LAMBDA: float = 0.7            # VFE weighting factor
MINHASH_K: int = 64            # number of hash functions for MinHash
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing

# ----------------------------------------------------------------------
# Parent A structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """Kernel weight φ(|S|, F) used in exact Shapley enumeration."""
    return (math.factorial(subset_size) *
            math.factorial(feature_count - subset_size - 1) /
            math.factorial(feature_count))

# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector w(d) for the given weekday index ``dow`` (0=Sun … 6=Sat).

    The vector follows a sinusoidal pattern whose phase shifts with the weekday.
    """
    if not (0 <= dow <= 6):
        raise ValueError("weekday index must be in 0..6")
    weights = np.zeros(len(groups))
    for i, _ in enumerate(groups):
        # Phase offset ensures each group peaks on a different day.
        phase = 2 * math.pi * (dow + i) / 7
        weights[i] = math.sin(phase) + 1.0   # shift to be non‑negative
    return weights / np.sum(weights)

def minhash_signature(data: bytes, k: int = MINHASH_K) -> np.ndarray:
    """Compute a MinHash signature of length *k* for a byte string."""
    rng = np.random.default_rng(seed=int.from_bytes(data[:8], "little", signed=False))
    # Simple pseudo‑hash: generate k 64‑bit random numbers and xor with data hash.
    base = int.from_bytes(data[:8], "little", signed=False)
    sig = np.empty(k, dtype=np.uint64)
    for i in range(k):
        rand = rng.integers(0, MAX64, dtype=np.uint64)
        sig[i] = (base ^ rand) & MAX64
    return sig

def jaccard_from_minhash(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same shape")
    return np.mean(sig1 == sig2)

def statistical_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson‑like similarity used in Parent B:
    ρ = cov(x,y) / (σ_x σ_y)
    Returns a value in [‑1, 1].
    """
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    if x.size == 0:
        raise ValueError("vectors must be non‑empty")
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    cov = np.mean((x - mx) * (y - my))
    return cov / (sx * sy)

def liquid_time_constant(similarity: float) -> float:
    """
    Gated liquid‑time constant τ(ρ) = BASE_TAU / (1 + exp(‑ALPHA·(ρ‑0.5))).
    The similarity ρ is first linearly shifted from [‑1, 1] to [0, 1].
    """
    rho = (similarity + 1.0) / 2.0          # map to [0,1]
    return BASE_TAU / (1.0 + math.exp(-ALPHA * (rho - 0.5)))

def variational_free_energy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Simple VFE proxy: cross‑entropy between a probabilistic prediction and target.
    Assumes predictions are logits; we apply softmax internally.
    """
    if pred.shape != target.shape:
        raise ValueError("prediction and target must share shape")
    # softmax
    e = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    # avoid log(0)
    eps = np.finfo(probs.dtype).eps
    ce = -np.sum(target * np.log(probs + eps))
    return ce / pred.shape[0]

def shannon_entropy(weights: np.ndarray) -> float:
    """Weight‑scaled Shannon entropy used by Parent B."""
    w = weights / np.sum(weights)
    eps = np.finfo(w.dtype).eps
    return -np.sum(w * np.log(w + eps))

# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_shapley_value(
    value_fn: Callable[[FrozenSet[int]], float],
    feature_count: int,
    similarity: float,
) -> np.ndarray:
    """
    Compute exact Shapley values for *feature_count* features, but weight each
    marginal contribution by the liquid‑time constant τ(ρ) derived from a
    similarity measure (Parent B).  The result is a length‑F array.
    """
    tau = liquid_time_constant(similarity)
    shapley_vals = np.zeros(feature_count, dtype=np.float64)
    for i in range(feature_count):
        v_si = value_fn(FrozenSet({i}))
        v_s = 0.0
        for subset_size in range(feature_count):
            for subset in combinations(range(feature_count), subset_size):
                if i not in subset:
                    s = FrozenSet(subset)
                    v_s += shapley_kernel_weight(len(s), feature_count) * (value_fn(s | {i}) - value_fn(s))
        shapley_vals[i] = tau * v_si * v_s
    return shapley_vals

def hybrid_quality_metric(
    pred: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    similarity: float,
) -> float:
    """
    Combine variational free energy (VFE) from Parent A with a Shannon‑entropy‑based
    hygiene score from Parent B using a learned factor LAMBDA.
    """
    vfe = variational_free_energy(pred, target)
    entropy = shannon_entropy(weights)
    tau = liquid_time_constant(similarity)
    return LAMBDA * vfe + (1 - LAMBDA) * tau * entropy

def improved_hybrid_shapley_value(
    value_fn: Callable[[FrozenSet[int]], float],
    feature_count: int,
    similarity: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Improved version of hybrid Shapley value computation with better numerical stability.
    """
    tau = liquid_time_constant(similarity)
    shapley_vals = np.zeros(feature_count, dtype=np.float64)
    for i in range(feature_count):
        v_si = value_fn(FrozenSet({i}))
        v_s = 0.0
        for subset_size in range(feature_count):
            for subset in combinations(range(feature_count), subset_size):
                if i not in subset:
                    s = FrozenSet(subset)
                    v_s_i = value_fn(s | {i})
                    delta_v = v_s_i - value_fn(s)
                    weight = shapley_kernel_weight(len(s), feature_count)
                    v_s += weight * delta_v
        shapley_vals[i] = tau * v_si * v_s
    # stabilize the computation by adding a small epsilon
    return np.maximum(shapley_vals, epsilon)

def improved_hybrid_quality_metric(
    pred: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    similarity: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Improved version of hybrid quality metric computation with better numerical stability.
    """
    vfe = variational_free_energy(pred, target)
    entropy = shannon_entropy(weights)
    tau = liquid_time_constant(similarity)
    return LAMBDA * vfe + (1 - LAMBDA) * np.maximum(tau * entropy, epsilon)