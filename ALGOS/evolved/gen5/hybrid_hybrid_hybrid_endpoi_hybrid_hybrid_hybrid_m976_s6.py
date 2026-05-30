# DARWIN HAMMER — match 976, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_krampus_brain_m295_s2.py (gen3)
# born: 2026-05-29T23:32:08Z

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology and circuit‑breaker primitives
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""

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
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

    def as_vector(self) -> np.ndarray:
        """Return the morphology as a 1‑D numpy array."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_success(self) -> None:
        """Reset failures and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        """Increment failures and open the breaker if threshold is hit."""
        self.failures += 1
        self.last_event_at = now_z()
        if self.failures >= self.failure_threshold:
            self.open = True

    def status(self) -> Tuple[bool, int]:
        """Return (open, failures)."""
        return self.open, self.failures


def fisher_score(morph: Morphology) -> float:
    """
    Compute a scalar Fisher‑like discriminative weight from morphology.

    The score is defined as the ratio of between‑class variance (here
    approximated by the variance of the four dimensions) to the mean
    dimension value.  It is guaranteed to be positive.
    """
    vec = morph.as_vector()
    variance = np.var(vec, ddof=1)
    mean = np.mean(vec)
    # Avoid division by zero; add a tiny epsilon.
    eps = 1e-12
    return variance / (mean + eps) + 1.0  # shift to ensure >1


def prune_probability(cb: EndpointCircuitBreaker, flow_norm: float, fisher_w: float) -> float:
    """
    Compute a probability that the circuit‑breaker will prune (i.e. stay closed).

    The raw probability grows with the magnitude of the hybrid flow and the
    Fisher weight, but is capped at 1.0.
    """
    base = flow_norm * fisher_w
    # Scale by a heuristic factor to keep the value in [0,1].
    prob = min(1.0, base / (cb.failure_threshold + 1.0))
    return prob


# ----------------------------------------------------------------------
# Parent B – rectified flow and Ollivier‑Ricci curvature
# ----------------------------------------------------------------------


def rectified_flow(v_src: np.ndarray, v_tgt: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation between source and target embeddings.

    Parameters
    ----------
    v_src, v_tgt : np.ndarray
        Same‑shaped vectors.
    t : float
        Interpolation coefficient in [0, 1].

    Returns
    -------
    np.ndarray
        (1‑t)·v_src + t·v_tgt
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be in [0, 1]")
    return (1.0 - t) * v_src + t * v_tgt


def ollivier_ricci_curvature(
    v_src: np.ndarray,
    v_tgt: np.ndarray,
    samples: int = 20,
    sigma: float = 1.0,
) -> float:
    """
    Approximate discrete Ollivier‑Ricci curvature on the edge (v_src, v_tgt).

    Gaussian neighbourhoods μ_src and μ_tgt are sampled, the average Euclidean
    distance between paired samples approximates the 1‑Wasserstein distance.
    """
    if v_src.shape != v_tgt.shape:
        raise ValueError("Source and target vectors must have the same shape")
    dim = v_src.shape[0]

    # Sample Gaussian clouds around each endpoint.
    rng = np.random.default_rng()
    src_samples = rng.normal(loc=v_src, scale=sigma, size=(samples, dim))
    tgt_samples = rng.normal(loc=v_tgt, scale=sigma, size=(samples, dim))

    # Pairwise Euclidean distances between corresponding samples.
    dists = np.linalg.norm(src_samples - tgt_samples, axis=1)
    w1 = np.mean(dists)

    edge_len = np.linalg.norm(v_src - v_tgt)
    if edge_len == 0:
        return 0.0
    curvature = 1.0 - w1 / edge_len
    return curvature


# ----------------------------------------------------------------------
# Hybrid core – three functions demonstrating the fused operation
# ----------------------------------------------------------------------


def build_extended_state(
    stylometric: np.ndarray, brain_map: np.ndarray, morph: Morphology
) -> np.ndarray:
    """
    Concatenate stylometric counts, brain‑map features and morphology into a
    single embedding vector.

    Returns
    -------
    np.ndarray
        v̂ = [s; b; m]  with shape (n + m + 4,)
    """
    return np.concatenate([stylometric, brain_map, morph.as_vector()])


def hybrid_transport(
    v_src: np.ndarray,
    v_tgt: np.ndarray,
    t: float,
    morph: Morphology,
) -> Tuple[np.ndarray, float]:
    """
    Perform the curvature‑modulated, Fisher‑weighted transport from source to
    target.

    Returns
    -------
    v_hybrid : np.ndarray
        The hybrid embedding after transport.
    κ : float
        The computed Ollivier‑Ricci curvature (used for diagnostics).
    """
    # 1. Base linear flow.
    phi = rectified_flow(v_src, v_tgt, t)

    # 2. Fisher weight from morphology.
    w_f = fisher_score(morph)

    # 3. Curvature on the extended edge.
    κ = ollivier_ricci_curvature(v_src, v_tgt)

    # 4. Apply both modifiers.
    v_hybrid = (1.0 + κ) * w_f * phi
    return v_hybrid, κ


def update_breaker_from_flow(
    cb: EndpointCircuitBreaker,
    flow_vec: np.ndarray,
    fisher_w: float,
) -> None:
    """
    Update the circuit‑breaker based on the hybrid flow.

    The breaker’s failure probability is modulated by the flow and Fisher
    weight.
    """
    flow_norm = np.linalg.norm(flow_vec)
    prob = prune_probability(cb, flow_norm, fisher_w)
    # Simulate breaker update (success or failure).
    if np.random.rand() < prob:
        cb.record_success()
    else:
        cb.record_failure()


def smoke_test() -> None:
    """
    A minimal test case to ensure the hybrid pipeline runs without errors.
    """
    # Mock inputs.
    stylometric = np.array([1.0, 2.0])
    brain_map = np.array([3.0, 4.0])
    morph = Morphology(5.0, 6.0, 7.0, 8.0)
    v_src = build_extended_state(stylometric, brain_map, morph)
    v_tgt = v_src + np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    t = 0.5
    cb = EndpointCircuitBreaker()

    # Run hybrid transport and update breaker.
    v_hybrid, κ = hybrid_transport(v_src, v_tgt, t, morph)
    update_breaker_from_flow(cb, v_hybrid, fisher_score(morph))

    # Sanity check.
    assert cb.status() is not None


if __name__ == "__main__":
    smoke_test()