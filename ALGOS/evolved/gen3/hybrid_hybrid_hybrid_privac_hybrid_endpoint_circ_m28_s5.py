# DARWIN HAMMER — match 28, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:25:23Z

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import exp
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str

# Example tiers (mirroring parent A)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Parent A – probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """
    Simple Laplace mechanism: sum(values) + Laplace(0, sensitivity/epsilon).
    """
    total = float(np.sum(list(values)))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise

# ----------------------------------------------------------------------
# Parent B – circuit‑breaker primitives
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

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

# ----------------------------------------------------------------------
# Parent B – morphology utilities
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """
    Ratio of the geometric mean of dimensions to the longest dimension.
    Returns a value in (0,1].
    """
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """
    Simple flatness: 1 - (min dimension / max dimension).
    Zero for a perfect cube, approaching 1 for extremely flat objects.
    """
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return 1.0 - (min(length, width, height) / max(length, width, height))

# ----------------------------------------------------------------------
# Hybrid entities
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntity:
    """Combines tier, privacy risk, and morphology."""
    tier: ModelTier
    uqis: int          # unique quasi‑identifiers
    total_records: int
    morphology: Morphology

    @property
    def risk(self) -> float:
        return reconstruction_risk_score(self.uqis, self.total_records)

    @property
    def ram_mb(self) -> int:
        return self.tier.ram_mb

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------

def compute_effective_load(models: Iterable[ModelEntity],
                           alpha: float = 0.3,
                           beta: float = 0.2) -> float:
    """
    Compute the morphology‑adjusted expected VRAM load:

        L̂ = Σ_i ( r_i · m_i · g_i )
    where
        g_i = 1 + α·(1‑s_i) + β·(1‑f_i)
        s_i = sphericity_index(...)
        f_i = flatness_index(...)

    Parameters
    ----------
    models: iterable of ModelEntity
    alpha, beta: non‑negative shape‑weight coefficients

    Returns
    -------
    float – effective load in MiB
    """
    total = 0.0
    for m in models:
        s = sphericity_index(m.morphology.length,
                             m.morphology.width,
                             m.morphology.height)
        f = flatness_index(m.morphology.length,
                           m.morphology.width,
                           m.morphology.height)
        g = 1.0 + alpha * (1.0 - s) + beta * (1.0 - f)
        total += m.risk * m.ram_mb * g
    return total

def admit_models(models: List[ModelEntity],
                 vram_budget: float,
                 breaker: EndpointCircuitBreaker,
                 alpha: float = 0.3,
                 beta: float = 0.2) -> List[ModelEntity]:
    """
    Attempt to admit a list of models under a VRAM budget using the circuit
    breaker.  Models are processed in order; if adding a model would exceed the
    budget, the breaker records a failure and the model is rejected.
    Successful admission resets the breaker.

    Returns the list of admitted models.
    """
    admitted: List[ModelEntity] = []
    current_load = 0.0

    for model in models:
        # compute marginal contribution
        s = sphericity_index(model.morphology.length,
                             model.morphology.width,
                             model.morphology.height)
        f = flatness_index(model.morphology.length,
                           model.morphology.width,
                           model.morphology.height)
        g = 1.0 + alpha * (1.0 - s) + beta * (1.0 - f)
        marginal_load = model.risk * model.ram_mb * g

        if current_load + marginal_load > vram_budget:
            breaker.record_failure()
            continue

        admitted.append(model)
        current_load += marginal_load
        breaker.record_success()

    return admitted

def dp_privacy_aggregate_risks(models: Iterable[ModelEntity],
                              epsilon: float = 1.0,
                              sensitivity: float = 1.0) -> float:
    """
    Compute the differential privacy aggregate of model risks.

    Parameters
    ----------
    models: iterable of ModelEntity
    epsilon: privacy budget
    sensitivity: sensitivity of the query

    Returns
    -------
    float – DP aggregate risk
    """
    risks = [model.risk for model in models]
    return dp_aggregate(risks, epsilon, sensitivity)

# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # create some example models
    models = [
        ModelEntity(TIER_T1_QWEN_0_5B, 10, 100, Morphology(1.0, 2.0, 3.0, 10.0)),
        ModelEntity(TIER_T2_REASONING, 20, 200, Morphology(4.0, 5.0, 6.0, 20.0)),
        ModelEntity(TIER_T3_QWEN_7B, 30, 300, Morphology(7.0, 8.0, 9.0, 30.0)),
    ]

    # create a circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # admit models under a VRAM budget
    vram_budget = 10000.0
    admitted_models = admit_models(models, vram_budget, breaker)

    # compute DP aggregate risk
    epsilon = 1.0
    sensitivity = 1.0
    dp_risk = dp_privacy_aggregate_risks(admitted_models, epsilon, sensitivity)

    print("Admitted models:", admitted_models)
    print("DP aggregate risk:", dp_risk)