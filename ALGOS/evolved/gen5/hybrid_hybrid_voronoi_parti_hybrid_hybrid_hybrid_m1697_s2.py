# DARWIN HAMMER — match 1697, survivor 2
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m905_s2.py (gen4)
# born: 2026-05-29T23:38:24Z

import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
BASE_TAU: float = 1.0          # baseline liquid time constant (unused in this module)
ALPHA: float = 5.0             # gating steepness (unused in this module)
LAMBDA: float = 0.7            # VFE weighting factor (unused in this module)
MINHASH_K: int = 64            # number of hash functions for MinHash (unused in this module)
MAX64: int = (1 << 64) - 1     # mask for 64‑bit hashing (unused in this module)

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC time in ISO‑8601 Z‑suffix format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Core domain objects
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical description of a 3‑D object.

    All dimensions must be strictly positive.
    """
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Convenient tuple of the three spatial dimensions."""
        return self.length, self.width, self.height


# ----------------------------------------------------------------------
# Geometric indices (Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimension‑less sphericity measure in (0, 1].

    The index is the ratio of the geometric mean of the three edges to the
    longest edge.  It equals 1 for a perfect cube.
    """
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest


def flatness_index(length: float, width: float, height: float) -> float:
    """Return a dimension‑less flatness measure ≥ 1.

    It is the ratio of the longest edge to the shortest edge.
    """
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive")
    return max(length, width, height) / min(length, width, height)


# ----------------------------------------------------------------------
# Statistical similarity (Parent B)
# ----------------------------------------------------------------------
def pearson_similarity(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation transformed to a similarity in [0, 1].

    ``x`` and ``y`` must be equal‑length, non‑empty numeric sequences.
    The classic Pearson correlation lies in [-1, 1]; we map it linearly
    to the unit interval so that 1 → 1 (identical) and -1 → 0 (maximally
    dissimilar).
    """
    if len(x) != len(y):
        raise ValueError("Sequences must have the same length")
    if len(x) == 0:
        raise ValueError("Sequences must be non‑empty")

    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)

    # Guard against zero variance which would produce NaN.
    if np.allclose(xv, xv.mean()) or np.allclose(yv, yv.mean()):
        return 0.0

    r = np.corrcoef(xv, yv)[0, 1]  # Pearson correlation
    return (r + 1.0) / 2.0        # map to [0, 1]


# ----------------------------------------------------------------------
# Weight vector generation – refined (previously could be negative)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised, strictly non‑negative weight vector for a given weekday.

    ``dow`` follows the ISO convention (0 = Monday … 6 = Sunday).  The
    vector is generated from a raised cosine (i.e. shifted sinusoid) that
    guarantees positivity, then normalised to sum to 1.
    """
    if not (0 <= dow <= 6):
        raise ValueError("dow must be in the range 0‑6")

    n = len(groups)
    # Shifted cosine yields values in [0, 1]
    raw = np.array([0.5 * (1 + math.cos(2 * math.pi * (i - dow) / n))
                    for i in range(n)], dtype=float)

    # Avoid division by zero in pathological cases (should never happen)
    total = raw.sum()
    if total == 0:
        raise RuntimeError("Generated weight vector sums to zero")
    return raw / total


# ----------------------------------------------------------------------
# Deep integration – hybrid metric
# ----------------------------------------------------------------------
def _normalise_index(value: float, v_min: float, v_max: float) -> float:
    """Scale ``value`` to the unit interval given known bounds.

    Values outside the bounds are clipped to the nearest bound.
    """
    if v_min >= v_max:
        raise ValueError("v_min must be smaller than v_max")
    clipped = max(min(value, v_max), v_min)
    return (clipped - v_min) / (v_max - v_min)


def hybrid_metric(
    morphology: Morphology,
    similarity: float,
    *,
    sphericity_bounds: Tuple[float, float] = (0.0, 1.0),
    flatness_bounds: Tuple[float, float] = (1.0, 10.0),
) -> float:
    """
    Compute a *deeply fused* metric that reflects geometry **and** statistical
    similarity.

    The geometric part consists of a normalized sphericity and flatness term.
    Both terms are scaled to [0, 1] using plausible bounds and then combined
    with a similarity‑dependent gating function:

        metric = (1 - α)·geom + α·similarity

    where α = similarity (i.e. the more similar the data, the more the
    similarity term dominates).  The result is guaranteed to lie in [0, 1].
    """
    if not (0.0 <= similarity <= 1.0):
        raise ValueError("similarity must be in the unit interval")

    sphericity = sphericity_index(*morphology.dimensions)
    flatness = flatness_index(*morphology.dimensions)

    s_norm = _normalise_index(sphericity, *sphericity_bounds)
    f_norm = _normalise_index(flatness, *flatness_bounds)

    geom = (s_norm + f_norm) / 2.0  # average of the two normalized geometric scores

    # Gating: similarity acts both as a weight and as an additive term.
    return (1.0 - similarity) * geom + similarity


# ----------------------------------------------------------------------
# Decision hygiene – refined variational free‑energy (VFE) estimate
# ----------------------------------------------------------------------
def decision_hygiene(
    morphology: Morphology,
    similarity: float,
    *,
    vfe_exponent: float = 1.0 / 3.0,
) -> float:
    """
    Evaluate decision‑hygiene quality.

    The VFE proxy is the geometric mean of the three spatial dimensions,
    optionally raised to a configurable exponent.  It is then modulated by
    the similarity measure, yielding a value in the same range as the
    similarity (i.e. [0, 1] after normalisation).

    The geometric mean is normalised against a user‑supplied maximum
    ``vfe_max`` (default 10.0) to keep the output bounded.
    """
    if not (0.0 <= similarity <= 1.0):
        raise ValueError("similarity must be in the unit interval")

    vfe_raw = (morphology.length * morphology.width * morphology.height) ** vfe_exponent
    vfe_max = 10.0  # heuristic upper bound for typical objects
    vfe_norm = min(vfe_raw / vfe_max, 1.0)

    return vfe_norm * similarity


# ----------------------------------------------------------------------
# Endpoint circuit breaker simulation – more principled
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Simple failure‑count circuit breaker.

    The breaker opens after ``failure_threshold`` consecutive failures.
    ``record_success`` resets the counter, while ``record_failure`` increments it.
    """
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
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
        """Return ``True`` if the breaker is closed (i.e. requests are allowed)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def simulate_endpoint_circuit_breaker(
    morphology: Morphology,
    similarity: float,
    trials: int = 20,
    success_threshold: float = 0.6,
) -> dict:
    """
    Run a Monte‑Carlo style simulation of the circuit breaker.

    For each trial we compute ``hybrid_metric``; if it exceeds
    ``success_threshold`` we count a success, otherwise a failure.
    The breaker state after the last trial is returned.
    """
    if trials <= 0:
        raise ValueError("trials must be a positive integer")
    if not (0.0 <= success_threshold <= 1.0):
        raise ValueError("success_threshold must be in [0, 1]")

    breaker = EndpointCircuitBreaker()
    for _ in range(trials):
        metric = hybrid_metric(morphology, similarity)
        if metric >= success_threshold:
            breaker.record_success()
        else:
            breaker.record_failure()
        # Early exit if the breaker is open – further trials would be blocked.
        if not breaker.allow():
            break
    return breaker.as_dict()


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example morphology
    morphology = Morphology(length=1.2, width=0.9, height=1.5, mass=2.3)

    # Example statistical sequences – in a real scenario these would be
    # derived from model outputs, embeddings, etc.
    seq_a = [0.12, 0.45, 0.33, 0.78, 0.61]
    seq_b = [0.10, 0.47, 0.35, 0.80, 0.60]

    similarity = pearson_similarity(seq_a, seq_b)

    print("Pearson‑derived similarity :", similarity)
    print("Hybrid metric               :", hybrid_metric(morphology, similarity))
    print("Decision hygiene            :", decision_hygiene(morphology, similarity))
    print("Circuit breaker state       :", simulate_endpoint_circuit_breaker(morphology, similarity))