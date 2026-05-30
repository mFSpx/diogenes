# DARWIN HAMMER — match 2857, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2313_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m160_s0.py (gen4)
# born: 2026-05-29T23:46:16Z

"""Hybrid Algorithm: Endpoint Circuit Breaker + Morphology + Certainty‑Weighted Geometric Product

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2313_s1.py (Endpoint Circuit Breaker + Morphology sphericity)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m160_s0.py (Hybrid Sheaf‑Certainty Cohomology + Geometric Product)

Mathematical Bridge:
The sphericity index 𝜎 (a scalar derived from a Morphology object) is used to
scale the epistemic confidence w = confidence_bps/10000 of a CertaintyFlag.
Both scalars modulate the failure threshold τ of the EndpointCircuitBreaker:

    τ = ⌈ τ₀ · (1 + (1‑σ)) · w ⌉

where τ₀ is the static base threshold.  The same confidence w also scales the
certainty‑weighted coboundary of a section, and the sphericity‑adjusted rotor
derived from morphology rotates an input vector via the hybrid geometric
product before the circuit‑breaker decision is taken.  This creates a single
closed‑form update that fuses the two parent topologies.

The module provides three core hybrid functions:
1. `calculate_dynamic_failure_threshold`
2. `rotate_vector_with_morphology_rotor`
3. `hybrid_process_step`
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Tuple

# ----------------------------------------------------------------------
# Parent A components (Morphology + Endpoint Circuit Breaker)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2 * (l * w + w * h + h * l)

    @property
    def sphericity(self) -> float:
        """Classic sphericity σ = π^{1/3} (6V)^{2/3} / A."""
        V = self.volume
        A = self.surface_area
        if A == 0:
            return 0.0
        return (math.pi ** (1.0 / 3.0) * (6 * V) ** (2.0 / 3.0)) / A


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.base_threshold = failure_threshold
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
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_threshold": self.base_threshold,
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

# ----------------------------------------------------------------------
# Parent B components (Certainty + Geometric Product)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points 0..10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", now_z())


def now_z() -> str:
    """Current UTC time in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def certainty_weighted_coboundary(section: np.ndarray, flag: CertaintyFlag) -> np.ndarray:
    """Scale a section by the epistemic confidence."""
    w = flag.confidence_bps / 10000.0
    return w * np.array(section)


def hybrid_geometric_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Hybrid geometric product: inner (dot) + outer (cross) components.
    Returns a vector of the same dimension (3‑D assumed for cross).
    """
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    dot = np.dot(x, y)
    # For dimensions >3 we fall back to zero outer part
    if x.size == 3:
        cross = np.cross(x, y)
        return dot * np.ones(3) + cross
    else:
        return np.full_like(x, dot)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def calculate_dynamic_failure_threshold(
    morph: Morphology,
    base_threshold: int,
    flag: CertaintyFlag,
) -> int:
    """
    Compute τ = ceil( τ₀ · (1 + (1‑σ)) · w )
    where σ = morph.sphericity ∈ [0,1] and w = confidence/10000.
    """
    sigma = morph.sphericity
    w = flag.confidence_bps / 10000.0
    raw = base_threshold * (1.0 + (1.0 - sigma)) * w
    return max(1, int(math.ceil(raw)))


def rotate_vector_with_morphology_rotor(vec: np.ndarray, morph: Morphology) -> np.ndarray:
    """
    Build a rotor from morphology dimensions and rotate `vec` using the hybrid
    geometric product.  The rotor is a 3‑D vector (l, w, h) normalised.
    """
    rotor = np.array([morph.length, morph.width, morph.height], dtype=float)
    norm = np.linalg.norm(rotor)
    if norm == 0:
        return vec.copy()
    rotor /= norm
    # Rotate via hybrid product: r·v + r×v
    return hybrid_geometric_product(rotor, vec)


def hybrid_process_step(
    morph: Morphology,
    cb: EndpointCircuitBreaker,
    flag: CertaintyFlag,
    input_vec: np.ndarray,
    section: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """
    One hybrid iteration:
    1. Update the circuit breaker's failure threshold using morphology & certainty.
    2. Rotate the input vector with a morphology‑derived rotor.
    3. Apply certainty‑weighted coboundary to `section`.
    4. Compute a scalar decision metric = dot(rotated_vec, coboundary_section).
    5. If metric exceeds a simple adaptive bound, record a failure; else success.
    Returns (circuit_allows, rotated_vector).
    """
    # 1. dynamic threshold
    cb.failure_threshold = calculate_dynamic_failure_threshold(
        morph, cb.base_threshold, flag
    )

    # 2. rotate input
    rotated = rotate_vector_with_morphology_rotor(input_vec, morph)

    # 3. weighted coboundary
    weighted_section = certainty_weighted_coboundary(section, flag)

    # 4. decision metric (scalar)
    metric = float(np.dot(rotated, weighted_section))

    # 5. simple adaptive bound: mean magnitude of weighted_section
    bound = np.mean(np.abs(weighted_section)) * 0.5

    if metric > bound:
        cb.record_failure()
    else:
        cb.record_success()

    return cb.allow(), rotated


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic objects
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)
    cb = EndpointCircuitBreaker(failure_threshold=4)
    flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=7500,
        authority_class="lab",
        rationale="test run",
    )
    # Random 3‑D vector and section
    rng = np.random.default_rng(42)
    vec = rng.normal(size=3)
    sec = rng.normal(size=3)

    # Run a few hybrid steps
    for step in range(5):
        allowed, rotated = hybrid_process_step(morph, cb, flag, vec, sec)
        print(
            f"Step {step+1}: allowed={allowed}, "
            f"threshold={cb.failure_threshold}, failures={cb.failures}"
        )
        # mutate vector slightly to simulate changing input
        vec = rotated * (0.9 + 0.2 * rng.random())
    print("Final circuit breaker state:", cb.as_dict())