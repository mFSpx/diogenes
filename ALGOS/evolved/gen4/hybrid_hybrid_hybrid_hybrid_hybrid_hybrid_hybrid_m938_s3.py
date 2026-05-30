# DARWIN HAMMER — match 938, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (gen3)
# born: 2026-05-29T23:31:42Z

"""
Hybrid Endpoint Morphology–Circuit–SHAP Fusion

Parents:
- hybrid_hybrid_hybrid_endpoi_shap_attribution_m45_s0.py (circuit‑breaker + SHAP weighting)
- hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s7.py (morphology indices, recovery priority, model pool)

Mathematical bridge:
The circuit‑breaker health score `h ∈ [0,1]` is used as a multiplicative scaling factor for the
morphology‑derived recovery priority `p ∈ [0,1]`. The product `h·p` becomes a *dynamic weight*
that modulates any downstream attribution, here instantiated as a SHAP‑style kernel weight
`w(S) = (|S|!·(M‑|S|‑1)!)/M!` (the classic Shapley kernel). Thus the unified system blends
failure‑aware reliability (circuit state) with shape‑driven stability (righting‑time index)
and distributes the combined influence across feature subsets via the SHAP kernel.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core data structures (shared between parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A – Circuit breaker (health scoring)
# ----------------------------------------------------------------------
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

    def health_score(self) -> float:
        """
        Normalised health score in [0, 1].
        1.0 → fully healthy (no failures, circuit closed)
        0.0 → circuit open (failure_threshold reached)
        Linear interpolation between.
        """
        if self.open:
            return 0.0
        return max(0.0, min(1.0, (self.failure_threshold - self.failures) / self.failure_threshold))

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Parent B – Morphology indices and recovery priority
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2*height). Larger → flatter object."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """Physical proxy for how long an entity needs to right itself."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority ∈ [0,1] derived from righting time.
    Higher righting time → lower priority.
    """
    return max(0.0, min(1.0, 1.0 - righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Fusion – Hybrid functions
# ----------------------------------------------------------------------
def shapley_kernel_weight(subset_size: int, feature_count: int) -> float:
    """
    Classic Shapley kernel weight:
        w(S) = (|S|! * (M-|S|-1)!) / M!
    where M is total number of features.
    """
    if not (0 <= subset_size <= feature_count):
        raise ValueError("subset_size must be between 0 and feature_count")
    # factorials via gamma to stay in float domain for larger M
    from math import gamma

    return gamma(subset_size + 1) * gamma(feature_count - subset_size) / gamma(feature_count + 1)


def weighted_recovery(m: Morphology, cb: EndpointCircuitBreaker) -> float:
    """
    Combine morphology‑derived recovery priority with circuit‑breaker health.
    h = health_score ∈ [0,1]
    p = recovery_priority ∈ [0,1]
    Return h·p.
    """
    h = cb.health_score()
    p = recovery_priority(m)
    return h * p


def shap_weighted_score(
    subset: Tuple[int, ...],
    total_features: int,
    m: Morphology,
    cb: EndpointCircuitBreaker,
) -> float:
    """
    Compute a SHAP‑style attribution for a given feature subset, where the
    underlying value is the hybrid health‑recovery metric.
    """
    w = shapley_kernel_weight(len(subset), total_features)
    v = weighted_recovery(m, cb)
    return w * v


# ----------------------------------------------------------------------
# Auxiliary – Model pool that respects circuit health
# ----------------------------------------------------------------------
class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier


class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier, cb: EndpointCircuitBreaker) -> None:
        """
        Load a model only if the circuit breaker permits operation and RAM allows it.
        """
        if not cb.allow():
            raise RuntimeError(f"Circuit open – cannot load model {model.name}")
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create morphology instance
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Initialise circuit breaker and simulate a couple of successes/failures
    cb = EndpointCircuitBreaker(failure_threshold=4)
    cb.record_success()
    cb.record_failure()
    cb.record_failure()  # now failures = 2, health = 0.5

    # Compute hybrid metrics
    hybrid_val = weighted_recovery(morph, cb)
    print(f"Weighted recovery (health·priority): {hybrid_val:.4f}")

    # Demonstrate SHAP‑style weighting for a dummy feature subset
    total_feats = 5
    subset = (0, 2)  # arbitrary indices
    shap_score = shap_weighted_score(subset, total_feats, morph, cb)
    print(f"SHAP‑weighted score for subset {subset}: {shap_score:.6f}")

    # Model pool interaction
    pool = ModelPool(ram_ceiling_mb=2000)
    model_a = ModelTier(name="tiny-gpt", ram_mb=500, tier="small")
    model_b = ModelTier(name="big-gpt", ram_mb=1800, tier="large")
    pool.load(model_a, cb)  # should succeed
    try:
        pool.load(model_b, cb)  # may raise due to RAM limit
    except RuntimeError as e:
        print(f"Expected load failure: {e}")

    # Show loaded models
    print("Loaded models:", list(pool.loaded.keys()))
    sys.exit(0)