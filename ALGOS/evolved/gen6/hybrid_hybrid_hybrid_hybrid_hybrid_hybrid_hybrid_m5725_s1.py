# DARWIN HAMMER — match 5725, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1774_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s2.py (gen4)
# born: 2026-05-30T00:04:31Z

"""Hybrid Algorithm integrating Morphology‑driven Entropy, Circuit‑Breaker logic,
and Structural Similarity of Periodic Signals with VRAM‑aware Model Pooling.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1774_s0.py (EndpointCircuitBreaker,
  Morphology, sphericity_index, Shannon entropy, Minimum‑Cost Tree)
- hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s2.py (SSIM‑like similarity,
  periodic signal generation, ModelPool, VRAM budgeting)

Mathematical Bridge:
The sphericity index of a Morphology object is used as a multiplicative weight
for Shannon entropy of a data vector, producing a morphology‑aware information
measure. This weighted entropy is interpreted as a reconstruction‑risk score
that drives the VRAM‑aware ModelPool loading decisions. Simultaneously,
periodic signals generated from model‑specific frequencies are compared with
reference GPU‑memory‑usage patterns via an SSIM‑like similarity. The final
hybrid similarity score is the product of the weighted entropy (risk) and the
SSIM value, thus fusing information‑theoretic, geometric, and resource‑aware
components into a single unified metric.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Core data structures from Parent A
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
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

def sphericity_index(morphology: Morphology) -> float:
    """Ratio of geometric mean of dimensions to the longest dimension."""
    dimensions = np.array([morphology.length, morphology.width, morphology.height])
    geometric_mean = np.prod(dimensions) ** (1.0 / dimensions.size)
    longest = dimensions.max()
    return geometric_mean / longest if longest != 0 else 0.0

# ----------------------------------------------------------------------
# Information‑theoretic utilities (Parent A)
# ----------------------------------------------------------------------
def shannon_entropy(data: np.ndarray) -> float:
    """Shannon entropy of a 1‑D array of non‑negative values."""
    probs = data.astype(float)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs = probs / total
    # avoid log(0) by masking zeros
    mask = probs > 0
    return -np.sum(probs[mask] * np.log2(probs[mask]))

def weighted_entropy(morph: Morphology, data: np.ndarray) -> float:
    """Entropy weighted by the morphology's sphericity index."""
    base = shannon_entropy(data)
    weight = sphericity_index(morph)
    return base * weight

# ----------------------------------------------------------------------
# Minimum‑Cost Tree (simple Prim implementation) – Parent A
# ----------------------------------------------------------------------
def min_spanning_tree_cost(points: np.ndarray) -> float:
    """
    Compute the total weight of a Minimum Spanning Tree over a set of points.
    Points shape: (N, D). Edge weight = Euclidean distance.
    """
    if points.shape[0] == 0:
        return 0.0
    n = points.shape[0]
    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_cost = 0.0

    for _ in range(n):
        # select the unvisited vertex with smallest distance
        idx = np.argmin(np.where(visited, np.inf, min_dist))
        visited[idx] = True
        total_cost += min_dist[idx]

        # update distances to the remaining vertices
        diff = points - points[idx]
        dists = np.linalg.norm(diff, axis=1)
        min_dist = np.minimum(min_dist, dists)

    return total_cost

# ----------------------------------------------------------------------
# Core data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

# Example model tiers (could be extended)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

class ModelPool:
    """VRAM‑aware pool that respects a global RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.sensitive_records: List[Tuple[str, str]] = []  # (model, reason)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Attempt to load a model; raise RuntimeError if insufficient VRAM."""
        if not self.can_load(model):
            raise RuntimeError(f"Insufficient VRAM to load {model.name}")
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)

# ----------------------------------------------------------------------
# Periodic signal generation and SSIM‑like similarity – Parent B
# ----------------------------------------------------------------------
def generate_periodic_signal(freq_hz: float, duration_s: float, sample_rate: int = 1000) -> np.ndarray:
    """Generate a sine wave of given frequency and duration."""
    t = np.arange(0, duration_s, 1.0 / sample_rate)
    return np.sin(2 * np.pi * freq_hz * t)

def ssim(signal_x: np.ndarray, signal_y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified SSIM for 1‑D signals.
    Returns a value in [0, 1] where 1 indicates perfect similarity.
    """
    if signal_x.shape != signal_y.shape:
        raise ValueError("Signals must have the same shape")
    mu_x = signal_x.mean()
    mu_y = signal_y.mean()
    sigma_x = signal_x.var()
    sigma_y = signal_y.var()
    sigma_xy = ((signal_x - mu_x) * (signal_y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid operations (the required three+ functions)
# ----------------------------------------------------------------------
def hybrid_similarity(
    morph: Morphology,
    data_vector: np.ndarray,
    freq_ref: float,
    freq_model: float,
    duration: float = 1.0,
    sample_rate: int = 1000,
) -> float:
    """
    Compute a hybrid similarity score:
      - weighted entropy of `data_vector` (morphology‑aware risk)
      - SSIM between a reference periodic signal (freq_ref) and a model‑specific
        periodic signal (freq_model)
    The final score is the product of the two components, yielding a metric that
    simultaneously captures information content, geometric shape, and signal
    similarity.
    """
    # Component 1: morphology‑weighted entropy (risk)
    risk = weighted_entropy(morph, data_vector)

    # Component 2: SSIM between two periodic signals
    ref_sig = generate_periodic_signal(freq_ref, duration, sample_rate)
    model_sig = generate_periodic_signal(freq_model, duration, sample_rate)
    similarity = ssim(ref_sig, model_sig)

    return risk * similarity

def load_model_hybrid(
    pool: ModelPool,
    breaker: EndpointCircuitBreaker,
    model: ModelTier,
    morph: Morphology,
    data_vector: np.ndarray,
    risk_threshold: float = 1.0,
) -> bool:
    """
    Attempt to load a model using the hybrid risk assessment.
    - Compute weighted entropy as a risk score.
    - If risk exceeds `risk_threshold`, record a failure in the circuit breaker.
    - If the circuit breaker is open, abort loading.
    - Otherwise, load the model into the VRAM‑aware pool.
    Returns True on success, False otherwise.
    """
    if breaker.open:
        return False

    risk = weighted_entropy(morph, data_vector)

    if risk > risk_threshold:
        breaker.record_failure()
        return False

    try:
        pool.load(model)
        breaker.record_success()
        return True
    except RuntimeError:
        breaker.record_failure()
        return False

def morphology_mst_cost(morph: Morphology) -> float:
    """
    Build a 3‑D point set from the morphology dimensions and compute the
    Minimum Spanning Tree cost. This provides a geometry‑based scalar that can
    be used for further weighting if desired.
    """
    points = np.array([
        [0, 0, 0],
        [morph.length, 0, 0],
        [0, morph.width, 0],
        [0, 0, morph.height],
        [morph.length, morph.width, morph.height],
    ])
    return min_spanning_tree_cost(points)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create sample morphology
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.2)

    # Random data vector for entropy
    rng = np.random.default_rng(42)
    data_vec = rng.integers(0, 10, size=100)

    # Hybrid similarity between two frequencies
    score = hybrid_similarity(
        morph=morph,
        data_vector=data_vec,
        freq_ref=5.0,
        freq_model=5.5,
        duration=0.5,
    )
    print(f"Hybrid similarity score: {score:.4f}")

    # Model loading demo
    pool = ModelPool(ram_ceiling_mb=8000)
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    success = load_model_hybrid(
        pool=pool,
        breaker=breaker,
        model=TIER_T2_REASONING,
        morph=morph,
        data_vector=data_vec,
        risk_threshold=2.0,
    )
    print(f"Model load successful: {success}, Circuit breaker open: {breaker.open}")

    # Geometry MST cost
    mst = morphology_mst_cost(morph)
    print(f"MST cost for morphology: {mst:.3f}")