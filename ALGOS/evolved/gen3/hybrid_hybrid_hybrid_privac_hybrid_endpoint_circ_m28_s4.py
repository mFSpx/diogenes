# DARWIN HAMMER — match 28, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# born: 2026-05-29T23:25:23Z

"""Hybrid VRAM‑Privacy‑Circuit‑Morphology Scheduler

Parents:
- PARENT ALGORITHM A: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2.py
  Provides reconstruction_risk_score, dp_aggregate and the expected VRAM
  computation E[VRAM] = Σ r_i·m_i.

- PARENT ALGORITHM B: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py
  Provides an EndpointCircuitBreaker (failure‑counter) and geometric
  morphology utilities (sphericity_index, flatness_index).

Mathematical bridge:
Each model i is characterised by a privacy risk r_i ∈ [0,1] and a memory
consumption m_i (MiB).  The expected VRAM load L = Σ r_i·m_i is a linear
form.  Morphology supplies dimension‑based shape factors s_i (sphericity)
and f_i (flatness).  We treat these shape factors as *resource modifiers*:
the effective load becomes L̂ = Σ (r_i·m_i·g_i) where g_i = 1 + α·(1‑s_i) + β·(1‑f_i).
The circuit‑breaker monitors whether the cumulative effective load exceeds a
hard VRAM budget B; on breach it records a failure, otherwise a success.
The DP‑aggregate of the raw risks is also emitted for downstream privacy‑
preserving reporting.

The module implements this fused decision engine with three public functions:
    compute_effective_load(models, α, β)
    admit_models(models, vram_budget, breaker, α, β)
    dp_privacy_aggregate_risks(models, epsilon, sensitivity)
"""

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
        marginal = model.risk * model.ram_mb * g

        if not breaker.allow():
            # circuit open – reject without changing state
            continue

        if current_load + marginal <= vram_budget:
            admitted.append(model)
            current_load += marginal
            breaker.record_success()
        else:
            breaker.record_failure()
            # do not add the model; continue to next candidate

    return admitted


def dp_privacy_aggregate_risks(models: Iterable[ModelEntity],
                               epsilon: float = 1.0,
                               sensitivity: float = 1.0) -> float:
    """
    Apply a Laplace DP aggregation to the raw privacy risks of the provided
    models.
    """
    risks = (m.risk for m in models)
    return dp_aggregate(risks, epsilon=epsilon, sensitivity=sensitivity)


# ----------------------------------------------------------------------
# Utility for pretty‑printing model summaries
# ----------------------------------------------------------------------


def summarize_models(models: Iterable[ModelEntity]) -> List[dict[str, Any]]:
    """Return a JSON‑serialisable list of model descriptors."""
    out = []
    for m in models:
        out.append({
            "name": m.tier.name,
            "tier": m.tier.tier,
            "ram_mb": m.ram_mb,
            "risk": round(m.risk, 4),
            "sphericity": round(sphericity_index(m.morphology.length,
                                                 m.morphology.width,
                                                 m.morphology.height), 4),
            "flatness": round(flatness_index(m.morphology.length,
                                             m.morphology.width,
                                             m.morphology.height), 4),
        })
    return out


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a few synthetic models
    models = [
        ModelEntity(
            tier=TIER_T1_QWEN_0_5B,
            uqis=120,
            total_records=1000,
            morphology=Morphology(length=1.2, width=1.0, height=0.8, mass=0.5),
        ),
        ModelEntity(
            tier=TIER_T2_REASONING,
            uqis=800,
            total_records=5000,
            morphology=Morphology(length=2.5, width=2.0, height=2.0, mass=2.0),
        ),
        ModelEntity(
            tier=TIER_T3_QWEN_7B,
            uqis=2500,
            total_records=3000,
            morphology=Morphology(length=3.0, width=1.5, height=1.5, mass=5.0),
        ),
        ModelEntity(
            tier=TIER_T2_TOOL,
            uqis=50,
            total_records=200,
            morphology=Morphology(length=0.9, width=0.9, height=0.9, mass=0.3),
        ),
    ]

    # Parameters
    VRAM_BUDGET = 8000.0  # MiB
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Compute effective load before admission
    load_before = compute_effective_load(models)
    print(f"Effective expected load (pre‑admission): {load_before:.2f} MiB")

    # Attempt admission
    admitted = admit_models(models, VRAM_BUDGET, breaker)
    print(f"Admitted {len(admitted)} models out of {len(models)}.")
    print("Circuit breaker state:", breaker.as_dict())

    # Effective load after admission
    load_after = compute_effective_load(admitted)
    print(f"Effective expected load (post‑admission): {load_after:.2f} MiB")

    # DP‑aggregated privacy risk
    dp_risk = dp_privacy_aggregate_risks(admitted, epsilon=0.5)
    print(f"DP‑aggregated risk (ε=0.5): {dp_risk:.4f}")

    # Pretty JSON summary
    summary = summarize_models(admitted)
    print(json.dumps(summary, indent=2))