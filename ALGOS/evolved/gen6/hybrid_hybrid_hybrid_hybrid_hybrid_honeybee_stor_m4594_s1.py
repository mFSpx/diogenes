# DARWIN HAMMER — match 4594, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s0.py (gen3)
# parent_b: hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s3.py (gen5)
# born: 2026-05-29T23:56:44Z

"""Hybrid Algorithm: Privacy‑Risk‑Weighted Store Dynamics with SSIM‑Guided Scoring
==========================================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s2``  
  Provides a probabilistic *reconstruction risk score* and an expected VRAM
  consumption computed as the dot‑product of the risk score with model memory
  footprints.

* **Parent B** – ``hybrid_honeybee_store_hybrid_hybrid_hybrid_m836_s3``  
  Defines a deterministic *store* dynamical system (Δstore = α·propensity –
  β·confidence) and a Structural Similarity Index (SSIM) that weights the store
  update to obtain a *hybrid score*.

Mathematical Bridge
-------------------
Both parents expose a scalar in the interval ``[0, 1]`` that quantifies
*confidence* in a decision:

* ``risk_score`` (Parent A) – probability that a model will be accessed.
* ``ssim_score`` (Parent B) – similarity between two feature vectors.

The fusion treats these scalars interchangeably: the risk score is used as a
weight for the store dynamics, while the SSIM score can additionally modulate
the same dynamics when textual similarity information is available.  The core
hybrid update therefore becomes  


Δstore = (α·propensity – β·confidence) · weight
storeₜ₊₁ = max(0, storeₜ + Δstore·dt)


where ``weight`` is either the reconstruction risk or the SSIM similarity (or a
product of both).  This single equation unifies the probabilistic privacy model
with the deterministic resource‑control model.

The module below implements the unified mathematics, exposing three public
functions that demonstrate the hybrid operation.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


# ----------------------------------------------------------------------
# Endpoint circuit breaker (utility from Parent A)
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


# ----------------------------------------------------------------------
# Parent A – probabilistic privacy primitives
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """
    Returns a probability that a record can be re‑identified.
    Implements a simple exponential decay model:

        risk = 1 - exp( - λ * (unique_qi / total_records) )

    where λ = 5 is a tunable sensitivity constant.
    """
    if total_records <= 0:
        return 0.0
    lam = 5.0
    proportion = unique_quasi_identifiers / total_records
    risk = 1.0 - math.exp(-lam * proportion)
    return max(0.0, min(1.0, risk))


def expected_vram_load(models: List[ModelTier],
                       risk_score: float) -> float:
    """
    Expected VRAM consumption (in MB) as a dot‑product between the risk
    probability and each model's memory footprint.
    """
    if not models:
        return 0.0
    ram_array = np.array([m.ram_mb for m in models], dtype=float)
    # Treat risk_score as a uniform probability for each model
    prob_vector = np.full_like(ram_array, fill_value=risk_score, dtype=float)
    expected = float(np.dot(prob_vector, ram_array))
    return expected


# ----------------------------------------------------------------------
# Parent B – store dynamics and SSIM primitives
# ----------------------------------------------------------------------
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
K1 = 0.01
K2 = 0.03
L = 255.0            # dynamic range for SSIM (8‑bit images)


def _mean_variance_covariance(x: np.ndarray,
                              y: np.ndarray) -> Tuple[float, float, float]:
    """Utility returning μ₁, μ₂, σ₁², σ₂² and covariance σ₁₂."""
    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    var_x = float(np.var(x, ddof=0))
    var_y = float(np.var(y, ddof=0))
    cov = float(np.mean((x - mu_x) * (y - mu_y)))
    return mu_x, mu_y, var_x, var_y, cov


def ssim_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    One‑dimensional Structural Similarity Index Measurement.
    The implementation follows the classic SSIM formula:

        ssim = ((2 μ₁ μ₂ + C₁)(2 σ₁₂ + C₂)) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))

    Returns a value in [0, 1] (higher → more similar).
    """
    if x.shape != y.shape:
        raise ValueError("Input vectors must have the same shape")
    mu1, mu2, var1, var2, cov = _mean_variance_covariance(x, y)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    numerator = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    ssim = numerator / denominator if denominator != 0 else 0.0
    return max(0.0, min(1.0, ssim))


def store_update(store: float,
                 propensity: float,
                 confidence: float,
                 dt: float = DT,
                 weight: float = 1.0) -> float:
    """
    Deterministic store dynamics from Parent B, weighted by an external scalar.
    ``weight`` may be a risk score, an SSIM score, or any factor in [0,1].

    Δstore = (α·propensity – β·confidence)·weight
    storeₜ₊₁ = max(0, store + Δstore·dt)
    """
    delta = (ALPHA * propensity - BETA * confidence) * weight
    new_store = max(0.0, store + delta * dt)
    return new_store


# ----------------------------------------------------------------------
# Hybrid Functions (the true fusion)
# ----------------------------------------------------------------------
def hybrid_risk_weighted_store(models: List[ModelTier],
                               unique_quasi_identifiers: int,
                               total_records: int,
                               initial_store: float,
                               propensity: float,
                               confidence: float) -> float:
    """
    Combines Parent A's risk estimation with Parent B's store dynamics.
    The risk score is used as the weighting factor for the store update.
    """
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    # The store update is weighted by the privacy risk (higher risk → larger
    # impact on the resource store).
    new_store = store_update(initial_store, propensity, confidence, DT, risk)
    return new_store


def hybrid_ssim_weighted_store(feature_vec_a: np.ndarray,
                               feature_vec_b: np.ndarray,
                               initial_store: float,
                               propensity: float,
                               confidence: float) -> float:
    """
    Uses the SSIM similarity between two feature vectors as the weighting
    factor for the store dynamics (pure Parent B formulation).
    """
    similarity = ssim_score(feature_vec_a, feature_vec_b)
    new_store = store_update(initial_store, propensity, confidence, DT, similarity)
    return new_store


def hybrid_combined_score(models: List[ModelTier],
                          unique_quasi_identifiers: int,
                          total_records: int,
                          feature_vec_a: np.ndarray,
                          feature_vec_b: np.ndarray,
                          propensity: float,
                          confidence: float) -> Tuple[float, float]:
    """
    Full fusion:
      * Compute privacy risk (Parent A).
      * Compute SSIM similarity (Parent B).
      * Blend the two weights multiplicatively to obtain a *combined* weight.
      * Apply the blended weight to the store dynamics.
    Returns a tuple ``(new_store, expected_vram)``.
    """
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    similarity = ssim_score(feature_vec_a, feature_vec_b)
    combined_weight = risk * similarity  # both in [0,1] → product also in [0,1]

    # For demonstration we start from a zero store.
    new_store = store_update(0.0, propensity, confidence, DT, combined_weight)

    # Expected VRAM load based on the same risk score.
    expected_vram = expected_vram_load(models, risk)

    return new_store, expected_vram


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few dummy models
    models = [
        ModelTier(name="tiny", ram_mb=256, tier="low"),
        ModelTier(name="small", ram_mb=1024, tier="mid"),
        ModelTier(name="large", ram_mb=4096, tier="high")
    ]

    # Parameters for the privacy side
    uid = 42
    total = 1000

    # Parameters for the store side
    propensity = 0.8
    confidence = 0.3

    # Random feature vectors (simulating extracted text embeddings)
    rng = np.random.default_rng(1234)
    vec1 = rng.integers(0, 256, size=64).astype(float)
    vec2 = rng.integers(0, 256, size=64).astype(float)

    # Run the three hybrid functions
    store1 = hybrid_risk_weighted_store(models, uid, total, initial_store=5.0,
                                       propensity=propensity, confidence=confidence)
    print(f"Risk‑weighted store: {store1:.3f}")

    store2 = hybrid_ssim_weighted_store(vec1, vec2, initial_store=5.0,
                                       propensity=propensity, confidence=confidence)
    print(f"SSIM‑weighted store: {store2:.3f}")

    store3, vram = hybrid_combined_score(models, uid, total, vec1, vec2,
                                         propensity, confidence)
    print(f"Combined store: {store3:.3f}, Expected VRAM: {vram:.1f} MB")

    # Demonstrate the circuit breaker (auxiliary utility)
    cb = EndpointCircuitBreaker(failure_threshold=2)
    for i in range(4):
        if i % 2 == 0:
            cb.record_success()
        else:
            cb.record_failure()
        print(f"Step {i}: allow={cb.allow()}, failures={cb.failures}")

    sys.exit(0)