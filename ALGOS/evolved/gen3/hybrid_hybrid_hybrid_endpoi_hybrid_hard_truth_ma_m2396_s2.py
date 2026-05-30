# DARWIN HAMMER — match 2396, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s0.py (gen2)
# parent_b: hybrid_hard_truth_math_model_pool_m8_s0.py (gen1)
# born: 2026-05-29T23:42:22Z

"""Hybrid Endpoint-Circuit + Krampus Brainmap + Hard Truth Math + Model Pool Fusion

Parents:
- hybrid_hybrid_endpoint_circ_hybrid_krampus_brain_m24_s0.py
- hybrid_hard_truth_math_model_pool_m8_s0.py

Mathematical Bridge:
The health state of an EndpointCircuitBreaker provides a scalar *health factor* H ∈ [0,1].
Morphology yields a *curvature* C = sphericity × flatness (dimension‑less).
The product W = H·C is used as a *global weight* that modulates the influence of
stylometric feature vectors **f** (derived from FUNCTION_CATS) on model‑selection
decisions.  Concretely, the hybrid score for a candidate model *m* is

    S_m = W · ‖f‖₂ · (ram_m / R_max)⁻¹

where R_max is the RAM ceiling of the ModelPool.  Models with larger S_m are
prioritized for loading, while the circuit‑breaker can veto loading when it is
open.  This fuses the circuit‑breaker / morphology topology with the stylometry /
model‑pool topology into a single optimisation problem.

The module implements the combined mathematics and provides three public
functions that demonstrate the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (circuit breaker + morphology)
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
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def health_factor(self) -> float:
        """Return a normalized health factor H ∈ [0,1]."""
        if self.failure_threshold == 0:
            return 0.0
        return max(0.0, 1.0 - self.failures / self.failure_threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    geom_mean = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return geom_mean / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """
    Simple flatness: ratio of the smallest dimension to the average of the other two.
    Returns a dimension‑less number ∈ (0,1].
    """
    dims = sorted([length, width, height])
    smallest, mid, largest = dims
    if (mid + largest) == 0:
        raise ValueError("invalid dimensions for flatness")
    return smallest / ((mid + largest) / 2.0)


def curvature_score(morph: Morphology) -> float:
    """
    Combined curvature C = sphericity × flatness.
    Normalised to [0,1] by clipping.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flt = flatness_index(morph.length, morph.width, morph.height)
    c = sph * flt
    return max(0.0, min(1.0, c))


def global_weight(cb: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    W = H · C, where H is the circuit‑breaker health factor and C is curvature.
    Result is guaranteed to be in [0,1].
    """
    H = cb.health_factor()
    C = curvature_score(morph)
    return max(0.0, min(1.0, H * C))


# ----------------------------------------------------------------------
# Parent B components (stylometry + model pool)
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was'nt weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def extract_stylometry_features(text: str) -> np.ndarray:
    """
    Produce a vector f where each component is the relative frequency of a
    FUNCTION_CATS category in the supplied text.
    """
    tokens = [t.lower().strip(".,!?;:()[]{}\"'") for t in text.split()]
    total = len(tokens) or 1
    freqs = []
    for cat_words in FUNCTION_CATS.values():
        count = sum(1 for t in tokens if t in cat_words)
        freqs.append(count / total)
    return np.array(freqs, dtype=float)


def stylometry_norm(features: np.ndarray) -> float:
    """Return the Euclidean norm of the feature vector, normalised to [0,1]."""
    norm = np.linalg.norm(features)
    # The theoretical maximum for a unit‑sum vector of length N is sqrt(N)
    max_norm = math.sqrt(len(features))
    return max(0.0, min(1.0, norm / max_norm))


class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier  # e.g. "T1", "T2", "T3"

    def __repr__(self) -> str:
        return f"ModelTier(name={self.name!r}, ram_mb={self.ram_mb}, tier={self.tier!r})"


class ModelPool:
    """
    Manages loading/unloading of models under a RAM ceiling.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return self._used() + model.ram_mb <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> bool:
        """
        Attempt to load a model. Returns True on success, False otherwise.
        If insufficient RAM, tries to evict the lowest‑tier loaded model(s).
        """
        if self.is_loaded(model.name):
            return True  # already present

        if self.can_load(model):
            self.loaded[model.name] = model
            return True

        # eviction strategy: remove loaded models of higher tier number first
        # (e.g., T3 > T2 > T1)
        tier_order = {"T1": 1, "T2": 2, "T3": 3}
        # sort currently loaded models by descending tier (evict highest tier first)
        evict_candidates = sorted(
            self.loaded.values(),
            key=lambda m: tier_order.get(m.tier, 99),
            reverse=True,
        )
        freed = 0
        evicted = []
        for cand in evict_candidates:
            freed += cand.ram_mb
            evicted.append(cand.name)
            if self._used() - freed + model.ram_mb <= self.ram_ceiling_mb:
                break

        if self._used() - freed + model.ram_mb > self.ram_ceiling_mb:
            # still not enough RAM after eviction
            return False

        for name in evicted:
            del self.loaded[name]
        self.loaded[model.name] = model
        return True

    def unload(self, name: str) -> bool:
        if name in self.loaded:
            del self.loaded[name]
            return True
        return False

    def status(self) -> Tuple[int, int]:
        """Return (used_ram_mb, remaining_ram_mb)."""
        used = self._used()
        return used, max(0, self.ram_ceiling_mb - used)


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def compute_hybrid_score(
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    text: str,
) -> float:
    """
    Core hybrid metric:
        S = W · ‖f‖₂
    where W = H·C (global weight) and f are stylometry features.
    """
    W = global_weight(cb, morph)
    f = extract_stylometry_features(text)
    f_norm = stylometry_norm(f)
    return W * f_norm


def rank_models_for_text(
    models: List[ModelTier],
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    text: str,
) -> List[Tuple[ModelTier, float]]:
    """
    Produce a list of (model, priority) sorted descending.
    Priority = S / (ram_mb / R_max)   (higher priority = more likely to load)
    """
    S = compute_hybrid_score(cb, morph, text)
    R_max = max(1, max(m.ram_mb for m in models))  # avoid division by zero
    ranked = []
    for m in models:
        ram_factor = m.ram_mb / R_max
        priority = S / (ram_factor if ram_factor != 0 else 1)
        ranked.append((m, priority))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def load_optimal_models(
    pool: ModelPool,
    models: List[ModelTier],
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    text: str,
) -> List[ModelTier]:
    """
    Load models into the pool according to hybrid priority while respecting
    the circuit‑breaker state. Returns the list of models that ended up loaded.
    """
    if not cb.allow():
        # Circuit open → refuse to load any model
        return []

    ranked = rank_models_for_text(models, cb, morph, text)
    loaded_models = []
    for model, _priority in ranked:
        if pool.load(model):
            loaded_models.append(model)
        # Stop early if we have exhausted RAM
        if pool._used() >= pool.ram_ceiling_mb:
            break
    return loaded_models


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise circuit breaker with a low threshold to demonstrate health factor dynamics
    cb = EndpointCircuitBreaker(failure_threshold=5)
    for _ in range(2):
        cb.record_failure()  # 2 failures → health factor = 0.6

    # Define a simple morphology
    morph = Morphology(length=2.0, width=1.0, height=1.5, mass=3.0)

    # Sample text for stylometry extraction
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog, but it does not "
        "always happen. However, sometimes it does."
    )

    # Define a pool and a set of candidate models
    pool = ModelPool(ram_ceiling_mb=4000)
    candidate_models = [
        ModelTier(name="tiny", ram_mb=500, tier="T1"),
        ModelTier(name="small", ram_mb=1200, tier="T2"),
        ModelTier(name="medium", ram_mb=2000, tier="T2"),
        ModelTier(name="large", ram_mb=3000, tier="T3"),
    ]

    # Run hybrid loading
    loaded = load_optimal_models(pool, candidate_models, cb, morph, sample_text)

    # Output results
    used, remaining = pool.status()
    print("CircuitBreaker health factor:", cb.health_factor())
    print("Morphology curvature:", curvature_score(morph))
    print("Hybrid score (S):", compute_hybrid_score(cb, morph, sample_text))
    print("Loaded models:", loaded)
    print(f"RAM used: {used} MB / {pool.ram_ceiling_mb} MB (remaining {remaining} MB)")

    # Verify that the pool respects the RAM ceiling
    assert used <= pool.ram_ceiling_mb, "RAM usage exceeds ceiling"
    # Verify that no model is loaded when the circuit is open
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()  # now failures == threshold → circuit opens
    loaded_when_closed = load_optimal_models(pool, candidate_models, cb, morph, sample_text)
    assert not loaded_when_closed, "Models loaded despite open circuit"

    print("Smoke test completed successfully.")