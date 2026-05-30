# DARWIN HAMMER — match 4021, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m905_s2.py (gen4)
# born: 2026-05-29T23:53:17Z

"""Hybrid Algorithm combining Parent A (Shapley attribution, geometry, circuit breaker)
and Parent B (similarity‑modulated liquid time constant, MinHash, variational free‑energy).

Mathematical bridge:
* Parent B provides a statistical similarity ρ(x, y) ∈ [‑1, 1] between two
  sequences (means/variances/covariance).  This similarity is injected into
  Parent A by scaling the liquid‑time constant τ used to weight MinHash
  signatures that appear inside the Shapley‑value kernel.
* The weighted Shapley contribution of a feature *i* becomes

      w_i = τ(ρ) · φ(|S|, F) · v(S∪{i}) – v(S)

  where φ is the classic Shapley kernel weight, τ(ρ) = BASE_TAU /
  (1 + exp(‑ALPHA·(ρ‑0.5))) is the sigmoid‑gated liquid‑time constant
  from Parent B, and v(·) is an arbitrary value function.
* The variational free‑energy (VFE) from Parent A is blended with the
  Shannon‑entropy‑based hygiene score from Parent B using the factor LAMBDA,
  yielding a final hybrid quality metric Q.

The module therefore fuses topology A (geometric descriptors, circuit‑breaker,
exact Shapley enumeration) with topology B (similarity, MinHash, τ‑gating) into
a single, mathematically coherent system."""
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
# Core constants (from Parent B)
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
    shapley_vals = np.zeros(feature_count, dtype=float)

    # Enumerate all subsets S ⊂ F
    all_features = set(range(feature_count))
    for subset_size in range(feature_count):
        weight = shapley_kernel_weight(subset_size, feature_count) * tau
        for S in combinations(all_features, subset_size):
            S_set = frozenset(S)
            v_S = value_fn(S_set)
            for i in all_features - S_set:
                marginal = value_fn(S_set | {i}) - v_S
                shapley_vals[i] += weight * marginal

    return shapley_vals

def hybrid_similarity_weighted_signature(data_a: bytes, data_b: bytes, dow: int) -> float:
    """
    Produce a similarity score that blends:
    * MinHash Jaccard estimate (Parent A style)
    * Weekday‑dependent group weighting (Parent B style)
    * Statistical similarity ρ (Parent B) used to gate τ
    The final scalar is suitable for downstream weighting.
    """
    sig_a = minhash_signature(data_a)
    sig_b = minhash_signature(data_b)
    jaccard = jaccard_from_minhash(sig_a, sig_b)

    # weekday weighting vector influences the effective Jaccard
    w_vec = weekday_weight_vector(GROUPS, dow)
    weighted_jaccard = np.dot(w_vec, np.full(len(GROUPS), jaccard))

    # statistical similarity on raw byte frequencies
    freq_a = np.frombuffer(data_a, dtype=np.uint8).astype(float)
    freq_b = np.frombuffer(data_b, dtype=np.uint8).astype(float)
    rho = statistical_similarity(freq_a, freq_b)

    tau = liquid_time_constant(rho)
    return weighted_jaccard * tau

def hybrid_quality_metric(
    morphology: Morphology,
    data_a: bytes,
    data_b: bytes,
    pred: np.ndarray,
    target: np.ndarray,
    dow: int,
    circuit: EndpointCircuitBreaker,
) -> float:
    """
    End‑to‑end hybrid metric Q that fuses:
    * Geometric sphericity (Parent A)
    * Hybrid similarity‑driven τ (Parent B)
    * Variational free‑energy (Parent A)
    * Shannon entropy hygiene (Parent B)

    The circuit breaker guards the whole computation; on failure the
    function returns NaN and records the failure.
    """
    if not circuit.allow():
        return float("nan")

    try:
        # 1. Geometry term
        sph = sphericity_index(morphology.length,
                               morphology.width,
                               morphology.height)

        # 2. Hybrid similarity term
        sim_score = hybrid_similarity_weighted_signature(data_a, data_b, dow)

        # 3. VFE term
        vfe = variational_free_energy(pred, target)

        # 4. Hygiene term
        # Use weekday weights as a proxy for “group importance”
        w_vec = weekday_weight_vector(GROUPS, dow)
        hygiene = shannon_entropy(w_vec)

        # Combine: Q = sph * sim_score * (LAMBDA * vfe + (1‑LAMBDA) * hygiene)
        quality = sph * sim_score * (LAMBDA * vfe + (1.0 - LAMBDA) * hygiene)

        circuit.record_success()
        return quality
    except Exception:
        circuit.record_failure()
        return float("nan")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy geometry
    morph = Morphology(length=2.5, width=1.2, height=0.8, mass=3.4)

    # Random byte payloads
    data1 = random.randbytes(128)
    data2 = random.randbytes(128)

    # Dummy classification logits and one‑hot target
    logits = np.random.randn(5, 10)  # batch of 5, 10 classes
    targets = np.zeros_like(logits)
    for i in range(5):
        targets[i, random.randint(0, 9)] = 1.0

    # Circuit breaker instance
    cb = EndpointCircuitBreaker(failure_threshold=2)

    # Compute hybrid metric for each weekday
    for dow in range(7):
        q = hybrid_quality_metric(
            morphology=morph,
            data_a=data1,
            data_b=data2,
            pred=logits,
            target=targets,
            dow=dow,
            circuit=cb,
        )
        print(f"Weekday {dow}: quality = {q:.6f}")

    # Demonstrate hybrid Shapley on a tiny toy value function
    def toy_value_fn(coalition: FrozenSet[int]) -> float:
        # simple linear value: sum of feature ids + 1
        return sum(coalition) + 1.0

    sim = statistical_similarity(
        np.arange(10, dtype=float),
        np.arange(10, dtype=float) * 0.9 + 0.5,
    )
    shap_vals = hybrid_shapley_value(toy_value_fn, feature_count=4, similarity=sim)
    print("Hybrid Shapley values:", shap_vals)