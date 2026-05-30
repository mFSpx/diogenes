# DARWIN HAMMER — match 28, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:25:23Z

"""Hybrid VRAM‑Privacy‑Morphology Scheduler

Parents:
- hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py (risk → expected VRAM)
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (circuit breaker & morphology)

Mathematical bridge:
Each model *i* has a reconstruction risk 𝑟_i ∈[0,1] (probability of access) and a VRAM demand m_i (MiB).
We also compute a morphology scaling factor s_i from its geometric description
(sphericity_index). The weighted expected VRAM load becomes

    Ê[VRAM] = Σ_i ( r_i · s_i · m_i )

A Laplace‑DP aggregate of the risks, 𝑅̂ = DP_aggregate({r_i}), supplies a privacy‑aware
budget that the circuit‑breaker monitors.  The hybrid planner admits a model if

    r_i·s_i·m_i ≤ (VRAM_budget – current_load)   and   breaker.allow()

Thus the algorithm fuses probabilistic risk, deterministic memory, geometric
scaling, differential‑privacy aggregation and a failure‑aware circuit breaker
into a single decision engine.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a logical entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class ModelSpec:
    """Combined specification used by the hybrid scheduler."""
    tier: ModelTier
    morphology: Morphology
    unique_quasi_identifiers: int
    total_records: int


# ----------------------------------------------------------------------
# Parent A – risk & DP utilities
# ----------------------------------------------------------------------


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Simple Laplace differential‑privacy sum aggregator.
    Returns Σ values + Laplace(0, sensitivity/epsilon) noise.
    """
    total = sum(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


# ----------------------------------------------------------------------
# Parent B – circuit breaker & morphology utilities
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

    def as_dict(self) -> Dict[str, Any]:
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
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    longest = max(length, width, height)
    return geo_mean / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (shortest dimension) / (longest dimension)."""
    dims = sorted([length, width, height])
    return dims[0] / dims[-1]


# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical fusion)
# ----------------------------------------------------------------------


def morphology_scaling(morph: Morphology) -> float:
    """
    Compute a scalar weight from morphology.
    We combine sphericity (favoring compact shapes) and flatness (penalizing
    extremely flat shapes) into a single factor in (0, 1].
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    # Normalise: sphericity already ≤1, flatness ≤1; multiply and clip.
    weight = max(0.01, min(1.0, sph * flat))
    return weight


def expected_vram_load(models: List[ModelSpec]) -> float:
    """
    Compute Σ (risk_i * morphology_weight_i * ram_i).
    This is the fused expected VRAM load used for budgeting.
    """
    total = 0.0
    for spec in models:
        risk = reconstruction_risk_score(
            spec.unique_quasi_identifiers, spec.total_records
        )
        weight = morphology_scaling(spec.morphology)
        total += risk * weight * spec.tier.ram_mb
    return total


def hybrid_admission_decision(
    candidate: ModelSpec,
    current_models: List[ModelSpec],
    vram_budget: int,
    breaker: EndpointCircuitBreaker,
    epsilon: float = 1.0,
) -> bool:
    """
    Decide whether to admit *candidate* into memory.

    1. Verify circuit‑breaker permits operation.
    2. Compute DP‑aggregated risk of the *candidate* plus already admitted models.
    3. Compute expected VRAM after admission using the fused formula.
    4. Admit if the expected load stays within the VRAM budget.
    """
    if not breaker.allow():
        breaker.record_failure()
        return False

    # DP‑aggregate risk of the whole set (including candidate)
    risks = [
        reconstruction_risk_score(m.unique_quasi_identifiers, m.total_records)
        for m in current_models
    ] + [
        reconstruction_risk_score(
            candidate.unique_quasi_identifiers, candidate.total_records
        )
    ]
    dp_risk = dp_aggregate(risks, epsilon=epsilon)

    # Use dp_risk as a scaling factor (privacy‑budget awareness)
    scaling_factor = max(0.0, min(1.0, dp_risk / len(risks)))  # average noisy risk

    # Expected VRAM with candidate
    projected_models = current_models + [candidate]
    raw_expected = expected_vram_load(projected_models)
    adjusted_expected = raw_expected * scaling_factor

    if adjusted_expected <= vram_budget:
        breaker.record_success()
        return True
    else:
        breaker.record_failure()
        return False


def simulate_hybrid_workflow(
    model_pool: List[ModelSpec],
    vram_budget: int,
    breaker: EndpointCircuitBreaker,
    epsilon: float = 1.0,
) -> List[ModelSpec]:
    """
    Greedy simulation: iterate over *model_pool* and admit models
    when the hybrid decision permits them.
    Returns the list of admitted models.
    """
    admitted: List[ModelSpec] = []
    for spec in model_pool:
        if hybrid_admission_decision(spec, admitted, vram_budget, breaker, epsilon):
            admitted.append(spec)
    return admitted


# ----------------------------------------------------------------------
# Example data for testing
# ----------------------------------------------------------------------


# Define a few tiers
T1 = ModelTier("tiny-model", 256, "T1")
T2 = ModelTier("mid-model", 2048, "T2")
T3 = ModelTier("large-model", 8192, "T3")

# Example model specifications
EXAMPLE_POOL: List[ModelSpec] = [
    ModelSpec(
        tier=T1,
        morphology=Morphology(1.0, 1.0, 1.0, 0.5),
        unique_quasi_identifiers=10,
        total_records=1000,
    ),
    ModelSpec(
        tier=T2,
        morphology=Morphology(2.0, 1.5, 1.0, 2.0),
        unique_quasi_identifiers=200,
        total_records=5000,
    ),
    ModelSpec(
        tier=T3,
        morphology=Morphology(4.0, 2.0, 1.5, 5.0),
        unique_quasi_identifiers=800,
        total_records=8000,
    ),
    ModelSpec(
        tier=T2,
        morphology=Morphology(1.2, 0.9, 0.8, 1.0),
        unique_quasi_identifiers=5,
        total_records=300,
    ),
]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed random seed for reproducibility of DP noise
    np.random.seed(42)

    breaker = EndpointCircuitBreaker(failure_threshold=2)
    budget_mb = 10000  # 10 GiB VRAM budget

    admitted_models = simulate_hybrid_workflow(EXAMPLE_POOL, budget_mb, breaker, epsilon=0.5)

    print("Admitted models:")
    for m in admitted_models:
        print(f" - {m.tier.name} ({m.tier.ram_mb} MiB)")

    print("\nFinal circuit‑breaker state:")
    print(json.dumps(breaker.as_dict(), indent=2))

    print("\nExpected VRAM load after admission:")
    print(f"{expected_vram_load(admitted_models):.2f} MiB")