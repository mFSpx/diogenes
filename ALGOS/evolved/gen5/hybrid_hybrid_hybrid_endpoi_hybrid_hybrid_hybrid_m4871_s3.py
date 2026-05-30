# DARWIN HAMMER — match 4871, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_pherom_m1028_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_ternary_route_m812_s0.py (gen4)
# born: 2026-05-29T23:58:40Z

import sys
import math
import random
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – geometry & circuit breaker
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z‑suffix form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple circuit breaker whose threshold can be updated dynamically."""

    def __init__(self, failure_threshold: int = 3):
        self.base_threshold = failure_threshold
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
        """Return True if the breaker is closed (requests are allowed)."""
        return not self.open

    def adapt_threshold(self, scaling_factor: float, confidence: float) -> None:
        """
        Adjust the failure threshold according to the hybrid confidence score.

        τ = τ₀·(1 + λ·(1‑γ))

        Parameters
        ----------
        scaling_factor : λ  – non‑negative scalar that controls adaptation strength.
        confidence     : γ  – hybrid confidence in [0, 1].
        """
        if scaling_factor < 0:
            raise ValueError("scaling_factor must be non‑negative")
        new_thr = self.base_threshold * (1.0 + scaling_factor * (1.0 - confidence))
        # enforce integer threshold >=1
        self.failure_threshold = max(1, int(round(new_thr)))
        # recompute open state based on current failure count
        self.open = self.failures >= self.failure_threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_threshold": self.base_threshold,
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an object."""
    length: float
    width: float
    height: float
    mass: float  # mass is retained for possible extensions


def sphericity_index(length: float, width: float, height: float) -> float:
    """(V)^{1/3} / L where V = L·W·H."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """(L + W) / (2·H)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Simplified righting‑time proxy used in the original parent.
    Returns a dimensionless time‑scale estimate.
    """
    # The original code was truncated; we implement a plausible formulation.
    inertia = m.mass * (m.length ** 2 + m.width ** 2) / 12.0
    restoring = b * m.mass * 9.81 * neck_lever
    return math.sqrt(inertia / (k * restoring + 1e-9))


def compute_shape_vector(morph: Morphology) -> np.ndarray:
    """
    Return a 3‑element vector [sphericity, flatness, righting_time] normalized to [0,1].
    """
    s = sphericity_index(morph.length, morph.width, morph.height)
    f = flatness_index(morph.length, morph.width, morph.height)
    r = righting_time_index(morph)
    vec = np.array([s, f, r], dtype=float)

    # Normalise each component to [0,1] using simple min‑max heuristics.
    # Reasonable bounds are chosen empirically.
    bounds = np.array([[0.0, 1.0],   # sphericity theoretically ≤1
                       [0.0, 2.0],   # flatness can exceed 1 for very flat objects
                       [0.0, 10.0]])  # righting time (seconds) typical range
    mins = bounds[:, 0]
    maxs = bounds[:, 1]
    norm = (vec - mins) / (maxs - mins + 1e-12)
    return np.clip(norm, 0.0, 1.0)


# ----------------------------------------------------------------------
# Parent B – deterministic feature extraction
# ----------------------------------------------------------------------


def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit integer hash derived from SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a reproducible pseudo‑random feature vector from *text*.
    The same input always yields the same output across runs.
    """
    seed = _deterministic_hash(text) % (2 ** 32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """
    Reduce the full feature set to a compact master vector.
    The selection mirrors the original implementation.
    """
    f = extract_full_features(text)
    # Pick three features that conceptually align with the three shape indices.
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "symmetry_ratio": f["telemetry_agent_symmetry_ratio"],
        "entropy": f["psyche_poetic_entropy"],
    }


def master_vector_to_array(master: Dict[str, float]) -> np.ndarray:
    """Convert the master dict into a deterministic ordered NumPy array."""
    order = ["visceral_ratio", "symmetry_ratio", "entropy"]
    return np.array([master[k] for k in order], dtype=float)


# ----------------------------------------------------------------------
# Hybrid layer – mathematical bridge
# ----------------------------------------------------------------------


def hybrid_confidence(shape_vector: np.ndarray, master_vector: np.ndarray) -> float:
    """
    Compute a dimensionless confidence score γ by projecting ϕ onto w.

    Parameters
    ----------
    shape_vector  : 3‑element vector describing object morphology.
    master_vector : 3‑element vector describing text features.

    Returns
    -------
    γ : dimensionless confidence score in [0, 1].
    """
    # Ensure inputs are normalised
    assert np.all(shape_vector >= 0) and np.all(shape_vector <= 1)
    assert np.all(master_vector >= 0) and np.all(master_vector <= 1)

    # Compute linear weighted composition
    confidence = np.dot(shape_vector, master_vector)

    # Map to [0, 1] range using sigmoid
    return 1 / (1 + math.exp(-confidence))


def adapt_circuit_breaker(circuit_breaker: EndpointCircuitBreaker,
                          scaling_factor: float, confidence: float) -> None:
    circuit_breaker.adapt_threshold(scaling_factor, confidence)


def improved_hybrid_endpoint_circuit_bayesian_ternary_router(
    morphology: Morphology, text: str, base_threshold: int = 3,
    scaling_factor: float = 1.0) -> Tuple[EndpointCircuitBreaker, float]:
    shape_vector = compute_shape_vector(morphology)
    master_vector = master_vector_to_array(extract_master_vector(text))
    confidence = hybrid_confidence(shape_vector, master_vector)

    circuit_breaker = EndpointCircuitBreaker(base_threshold)
    adapt_circuit_breaker(circuit_breaker, scaling_factor, confidence)

    return circuit_breaker, confidence


if __name__ == "__main__":
    # Example usage
    morphology = Morphology(10.0, 5.0, 2.0, 1.0)
    text = "example text"
    circuit_breaker, confidence = improved_hybrid_endpoint_circuit_bayesian_ternary_router(
        morphology, text)
    print(circuit_breaker.as_dict())
    print(f"Confidence: {confidence:.4f}")