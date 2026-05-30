# DARWIN HAMMER — match 1857, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s1.py (gen4)
# born: 2026-05-29T23:39:13Z

"""
Hybrid Algorithm: hybrid_hybrid_endpoint_circuit_shap_morphology_fusion.py

This module fuses the core topologies of two parent algorithms:

* Parent A – hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s0.py  
  (provides morphology‑driven righting‑time index, recovery priority and a
  structural‑similarity (SSIM) based evaluation).

* Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s1.py  
  (provides a circuit‑breaker health model and a SHAP‑style attribution
  framework).

**Mathematical bridge** – Both parents expose the *sphericity_index* derived
from the same morphological parameters (length, width, height).  The bridge
uses this index to couple the morphology‑driven *recovery_priority* (Parent A)
with the circuit‑breaker *health_score* (Parent B).  The resulting
*hybrid_priority* scales SHAP attributions and SSIM‑based similarity measures,
producing a unified assessment that respects both physical stability and
service reliability.

The implementation below integrates the governing equations, provides three
core hybrid functions, and includes a smoke test.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Physical description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity used by both parents."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness metric from Parent A."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Morphology‑based stability metric (Parent A)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority derived from righting time (Parent A)."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Parent‑A specific endpoint definition
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EngineEndpoint:
    """Endpoint description used by Parent A."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


# ----------------------------------------------------------------------
# Parent‑B specific circuit‑breaker definition
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Simple circuit‑breaker model (Parent B)."""

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
        """Whether the endpoint is currently allowed to serve traffic."""
        return not self.open

    def health_score(self) -> float:
        """Linear health score in [0,1] (Parent B)."""
        return 1.0 - (self.failures / self.failure_threshold)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------


def hybrid_priority(endpoint: EngineEndpoint,
                    breaker: EndpointCircuitBreaker,
                    weight_sphericity: float = 0.5) -> float:
    """
    Combine morphology‑driven recovery priority with circuit‑breaker health.

    The sphericity index modulates the blend:
        hp = (w * RP + (1-w) * HS) * Sph

    where
        RP = recovery_priority(endpoint.morphology)
        HS = breaker.health_score()
        w = weight_sphericity (default 0.5)
        Sph = sphericity_index(...)
    """
    rp = recovery_priority(endpoint.morphology)
    hs = breaker.health_score()
    sph = sphericity_index(endpoint.morphology.length,
                          endpoint.morphology.width,
                          endpoint.morphology.height)
    blended = weight_sphericity * rp + (1.0 - weight_sphericity) * hs
    return blended * sph


def hybrid_ssim_state(x: np.ndarray,
                      y: np.ndarray,
                      morphology: Morphology) -> float:
    """
    SSIM‑like similarity between two state vectors, weighted by morphology.

    The classic SSIM formula:
        (2*mu_x*mu_y + C1) * (2* sigma_xy + C2) / ((mu_x^2+mu_y^2 + C1)*(sigma_x^2+sigma_y^2 + C2))

    We scale the contrast term by the sphericity index to embed the geometric
    bridge.
    """
    if x.shape != y.shape:
        raise ValueError("State vectors must have the same shape")

    C1 = 1e-4
    C2 = 9e-4

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.cov(x, y, bias=True)[0, 1]

    sph = sphericity_index(morphology.length,
                           morphology.width,
                           morphology.height)

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) * sph
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


def hybrid_shap_attribution(features: Dict[str, float],
                            endpoint: EngineEndpoint,
                            breaker: EndpointCircuitBreaker) -> Dict[str, float]:
    """
    Very lightweight SHAP‑style attribution.

    For each feature f we compute a baseline contribution proportional to the
    feature value.  The contribution is then scaled by the hybrid priority
    (which embeds both morphology and circuit‑breaker health).

    This mirrors Parent B's SHAP integration while respecting the physical
    stability from Parent A.
    """
    hp = hybrid_priority(endpoint, breaker)

    # Normalise absolute feature values to sum to 1 (if possible)
    abs_vals = np.array([abs(v) for v in features.values()], dtype=float)
    total = abs_vals.sum()
    if total == 0:
        norm = np.zeros_like(abs_vals)
    else:
        norm = abs_vals / total

    attribution = {}
    for (key, _), weight in zip(features.items(), norm):
        attribution[key] = weight * hp

    return attribution


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Construct a sample morphology
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=5.0)

    # Engine endpoint (Parent A)
    endpoint = EngineEndpoint(
        engine_id="eng-001",
        channel="alpha",
        residency="us-east",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="https://api.example.com/v1",
        capabilities=["predict", "explain"],
        morphology=morph,
    )

    # Circuit breaker (Parent B)
    breaker = EndpointCircuitBreaker(failure_threshold=4)
    # Simulate a couple of failures
    breaker.record_failure()
    breaker.record_failure()

    # Hybrid priority
    hp = hybrid_priority(endpoint, breaker)
    print(f"Hybrid priority: {hp:.4f}")

    # SSIM‑like similarity between two synthetic state vectors
    state_a = np.random.rand(10)
    state_b = np.random.rand(10)
    ssim_val = hybrid_ssim_state(state_a, state_b, morph)
    print(f"Hybrid SSIM similarity: {ssim_val:.4f}")

    # SHAP‑style attribution
    feats = {"feature1": 0.8, "feature2": -0.3, "feature3": 0.0}
    attribution = hybrid_shap_attribution(feats, endpoint, breaker)
    print("Hybrid SHAP attribution:")
    for k, v in attribution.items():
        print(f"  {k}: {v:.4f}")

    # Verify circuit‑breaker logic
    print(f"Circuit breaker allows traffic? {breaker.allow()}")
    # Record a success to reset
    breaker.record_success()
    print(f"After success – health score: {breaker.health_score():.2f}, allows: {breaker.allow()}")