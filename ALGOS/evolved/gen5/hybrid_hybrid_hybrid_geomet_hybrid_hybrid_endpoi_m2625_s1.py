# DARWIN HAMMER — match 2625, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_semant_m431_s0.py (gen4)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s3.py (gen2)
# born: 2026-05-29T23:43:18Z

import math
import random
import sys
import pathlib
from typing import Tuple, List, Dict
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – morphology and TemporalMotif
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
    support: int


# ----------------------------------------------------------------------
# Parent B – endpoint_circuit_breaker and path_signature
# ----------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at
        }


def path_signature(vector: np.ndarray, morphology: Morphology) -> np.ndarray:
    """
    Compute the path signature of a vector using the given morphology.

    Args:
    vector (np.ndarray): Input vector.
    morphology (Morphology): Morphology to use for computing the path signature.

    Returns:
    np.ndarray: Path signature of the input vector.
    """
    # Compute the Euclidean squared distance between the input vector and the origin
    distance = np.linalg.norm(vector) ** 2
    # Normalize the distance by the morphology's mass
    normalized_distance = distance / morphology.mass
    # Compute the path signature using the normalized distance
    signature = np.exp(-normalized_distance)
    return np.array([signature])


def hybrid_geometric_temporal_motif_fusion(
    vector: np.ndarray, morphology: Morphology, temporal_motif: TemporalMotif,
    failure_threshold: int = 3
) -> Tuple[np.ndarray, EndpointCircuitBreaker]:
    """
    Fuse the geometric algebra and Voronoi partitioning with the circuit-breaker state logic.

    Args:
    vector (np.ndarray): Input vector.
    morphology (Morphology): Morphology to use for computing the Voronoi partitioning.
    temporal_motif (TemporalMotif): Temporal motif to use for computing the stress measure.
    failure_threshold (int): Threshold for the circuit-breaker.

    Returns:
    Tuple[np.ndarray, EndpointCircuitBreaker]: Path signature of the input vector and the circuit-breaker state.
    """
    # Compute the path signature of the input vector
    signature = path_signature(vector, morphology)
    # Compute the stress measure using the weighted signature norm
    stress = np.sum(signature * temporal_motif.support)
    # Update the circuit-breaker state based on the stress measure
    circuit_breaker = EndpointCircuitBreaker(failure_threshold)
    if stress > failure_threshold:
        circuit_breaker.record_failure()
    else:
        circuit_breaker.record_success()
    return signature, circuit_breaker


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Create a random vector
    vector = np.random.rand(2)
    # Create a morphology
    morphology = Morphology(length=1.0, width=1.0, height=1.0, mass=1.0)
    # Create a temporal motif
    temporal_motif = TemporalMotif(pattern=("A", "B"), support=1)
    # Create a circuit-breaker
    circuit_breaker = EndpointCircuitBreaker(failure_threshold=3)
    # Fuse the geometric algebra and Voronoi partitioning with the circuit-breaker state logic
    signature, circuit_breaker_state = hybrid_geometric_temporal_motif_fusion(
        vector, morphology, temporal_motif
    )
    # Print the result
    print("Path signature:", signature)
    print("Circuit-breaker state:", circuit_breaker_state.as_dict())