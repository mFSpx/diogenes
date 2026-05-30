# DARWIN HAMMER — match 2625, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0.py (gen4)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s3.py (gen2)
# born: 2026-05-29T23:43:18Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Any, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Geometric Algebra utilities for Cl(2,0)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GA2DMultivector:
    """A simple 2‑D Euclidean Clifford (geometric) algebra multivector.

    Components are ordered as (scalar, e1, e2, e12).
    """
    s: float = 0.0   # scalar
    e1: float = 0.0  # vector e1
    e2: float = 0.0  # vector e2
    e12: float = 0.0 # bivector e12

    def __add__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        return GA2DMultivector(
            self.s + other.s,
            self.e1 + other.e1,
            self.e2 + other.e2,
            self.e12 + other.e12,
        )

    def __sub__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        return GA2DMultivector(
            self.s - other.s,
            self.e1 - other.e1,
            self.e2 - other.e2,
            self.e12 - other.e12,
        )

    def __mul__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        """Geometric product (Cl(2,0))."""
        a0, a1, a2, a12 = self.s, self.e1, self.e2, self.e12
        b0, b1, b2, b12 = other.s, other.e1, other.e2, other.e12

        # scalar part
        s = a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12
        # vector e1
        e1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
        # vector e2
        e2 = a0 * b2 + a2 * b0 - a1 * b12 + a12 * b1
        # bivector e12
        e12 = a0 * b12 + a12 * b0 + a1 * b2 - a2 * b1

        return GA2DMultivector(s, e1, e2, e12)

    def reverse(self) -> "GA2DMultivector":
        """Reversion changes sign of bivector part."""
        return GA2DMultivector(self.s, self.e1, self.e2, -self.e12)

    def norm(self) -> float:
        """Euclidean norm of the vector part (e1, e2)."""
        return math.hypot(self.e1, self.e2)

    def distance_to(self, other: "GA2DMultivector") -> float:
        """Euclidean distance between the vector parts of two multivectors."""
        return math.hypot(self.e1 - other.e1, self.e2 - other.e2)


def point_to_mv(point: np.ndarray) -> GA2DMultivector:
    """Convert a 2‑D numpy point to a pure vector multivector."""
    if point.shape != (2,):
        raise ValueError("point must be a 2‑element array")
    return GA2DMultivector(0.0, float(point[0]), float(point[1]), 0.0)


# ----------------------------------------------------------------------
# Domain specific data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int  # a positive integer weighting factor


# ----------------------------------------------------------------------
# Circuit‑breaker implementation
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold: int = failure_threshold
        self.failures: int = 0
        self.open: bool = False
        self.last_event_at: str = ""

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


# ----------------------------------------------------------------------
# Core mathematics – path signature, Voronoi, stress
# ----------------------------------------------------------------------


def path_signature(vector: np.ndarray, morphology: Morphology) -> float:
    """
    Compute a scalar path signature from a vector and a morphology.

    The signature is defined as exp( - (‖v‖² / mass) ),
    where ‖v‖ is the Euclidean norm of the input vector.
    """
    if morphology.mass <= 0.0:
        raise ValueError("Morphology mass must be positive")
    distance_sq = float(np.dot(vector, vector))
    normalized = distance_sq / morphology.mass
    return math.exp(-normalized)


def voronoi_region(
    query: np.ndarray,
    seeds: Sequence[np.ndarray],
) -> int:
    """
    Assign the query point to the nearest seed using Euclidean distance.
    Returns the index of the closest seed.
    """
    if not seeds:
        raise ValueError("At least one seed point is required for Voronoi partitioning")
    query_mv = point_to_mv(query)
    min_idx = -1
    min_dist = float("inf")
    for idx, seed in enumerate(seeds):
        seed_mv = point_to_mv(seed)
        d = query_mv.distance_to(seed_mv)
        if d < min_dist:
            min_dist = d
            min_idx = idx
    return min_idx


def compute_stress(
    signature: float,
    temporal_motif: TemporalMotif,
) -> float:
    """
    Derive a stress measure from the path signature and temporal motif.

    The stress is defined as:
        stress = signature * (len(pattern) * support)
    """
    weight = len(temporal_motif.pattern) * temporal_motif.support
    return signature * weight


# ----------------------------------------------------------------------
# Fusion entry point – deeper integration of the two parent systems
# ----------------------------------------------------------------------


def fused_geometric_voronoi_circuit_breaker(
    vector: np.ndarray,
    morphology: Morphology,
    temporal_motif: TemporalMotif,
    seed_points: Sequence[np.ndarray],
    failure_threshold: int = 3,
) -> Tuple[float, EndpointCircuitBreaker, int]:
    """
    Fully integrated routine:

    1. Convert the input vector to a GA multivector.
    2. Determine its Voronoi region among ``seed_points``.
    3. Compute a path signature that blends Euclidean distance,
       bivector magnitude (via a simple wedge with a fixed direction),
       and the morphology mass.
    4. Derive a stress value using the temporal motif.
    5. Update and return an ``EndpointCircuitBreaker`` based on the stress.

    Returns:
        signature (float): the scalar path signature.
        circuit_breaker (EndpointCircuitBreaker): updated state.
        region_idx (int): index of the Voronoi region the vector belongs to.
    """
    # 1. GA representation of the input
    mv = point_to_mv(vector)

    # 2. Voronoi assignment
    region_idx = voronoi_region(vector, seed_points)

    # 3. Enriched signature:
    #    - Euclidean part (as in original)
    #    - Bivector part: wedge the vector with a fixed unit direction (e1)
    #      to capture orientation information.
    unit_e1 = GA2DMultivector(0.0, 1.0, 0.0, 0.0)
    bivector = mv * unit_e1 - unit_e1 * mv  # 2 * (v ∧ e1) = 2 * (v1*e2 - v2*e1) e12
    biv_norm = abs(bivector.e12)  # magnitude of the bivector component

    # Combine Euclidean norm and bivector magnitude
    euclidean_norm = mv.norm()
    combined_metric = euclidean_norm**2 + biv_norm**2

    if morphology.mass <= 0.0:
        raise ValueError("Morphology mass must be positive")
    normalized_metric = combined_metric / morphology.mass
    signature = math.exp(-normalized_metric)

    # 4. Stress calculation
    stress = compute_stress(signature, temporal_motif)

    # 5. Circuit‑breaker update
    circuit_breaker = EndpointCircuitBreaker(failure_threshold)
    if stress > failure_threshold:
        circuit_breaker.record_failure()
    else:
        circuit_breaker.record_success()

    return signature, circuit_breaker, region_idx


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Random query vector
    query_vec = np.random.rand(2)

    # Example morphology
    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.0)

    # Temporal motif with a non‑trivial pattern
    motif = TemporalMotif(pattern=("α", "β", "γ"), support=2)

    # Seed points for Voronoi partitioning
    seeds = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    sig, cb, region = fused_geometric_voronoi_circuit_breaker(
        query_vec, morph, motif, seeds, failure_threshold=4
    )

    print("Query vector:", query_vec)
    print("Voronoi region index:", region)
    print("Path signature:", sig)
    print("Circuit‑breaker state:", cb.as_dict())