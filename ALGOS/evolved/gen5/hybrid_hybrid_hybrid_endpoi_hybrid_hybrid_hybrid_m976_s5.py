# DARWIN HAMMER — match 976, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_krampus_brain_m295_s2.py (gen3)
# born: 2026-05-29T23:32:08Z

"""Hybrid Morphology‑Circuit‑Flow Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – ``hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2.py``  
  Provides a ``Morphology`` description, a ``EndpointCircuitBreaker`` primitive
  and a *Fisher score* that weights the breaker’s failure‑probability.

* **Parent B** – ``hybrid_hybrid_hybrid_hard_t_hybrid_krampus_brain.py``  
  Supplies a joint embedding ``v = [s; b]`` (stylometric + brain‑map), a
  rectified linear flow ``Φ_t`` and a discrete **Ollivier‑Ricci curvature**
  ``κ`` that modulates the transport.

**Mathematical bridge**

1. The morphological parameters ``(length, width, height, mass)`` are
   treated as a 4‑dimensional *feature* vector ``m``.  Concatenating ``m`` to
   the joint embedding yields an extended state  

   ``v̂ = [s; b; m] ∈ ℝ^{n+m+4}``.

2. A *Fisher weight* ``w_f = fisher_score(m)`` (scalar) rescales the
   rectified flow, i.e. the raw transport ``Φ_t`` becomes ``w_f·Φ_t``.

3. The Ollivier‑Ricci curvature ``κ(v̂_src, v̂_tgt)`` is computed on the
   extended edge and multiplies the flow as in the original Parent B:

   ``v_hybrid = (1 + κ)·w_f·Φ_t(v̂_src, v̂_tgt)``.

4. The resulting hybrid vector drives the ``EndpointCircuitBreaker``:
   its failure‑probability is pruned by ``prune_probability`` using the
   curvature‑adjusted flow magnitude.

The three core functions below implement this pipeline, and a small
smoke‑test demonstrates a complete run without external dependencies."""

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
    morph: Morphology,
) -> None:
    """
    Use the norm of the hybrid flow to probabilistically update the circuit
    breaker.  A successful (low‑probability) outcome records a success,
    otherwise a failure.
    """
    flow_norm = np.linalg.norm(flow_vec)
    w_f = fisher_score(morph)
    prob = prune_probability(cb, flow_norm, w_f)

    # Random draw: success with probability (1‑prob), failure otherwise.
    if random.random() < (1.0 - prob):
        cb.record_success()
    else:
        cb.record_failure()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


def _smoke_test() -> None:
    # Synthetic stylometric (n=5) and brain‑map (m=7) vectors.
    rng = np.random.default_rng(42)
    s_src = rng.poisson(lam=3, size=5).astype(float)
    s_tgt = rng.poisson(lam=5, size=5).astype(float)

    b_src = rng.normal(loc=0.0, scale=1.0, size=7)
    b_tgt = rng.normal(loc=1.0, scale=1.0, size=7)

    # Morphology instance.
    morph = Morphology(length=2.3, width=1.1, height=0.8, mass=4.5)

    # Build extended state vectors.
    v_src = build_extended_state(s_src, b_src, morph)
    v_tgt = build_extended_state(s_tgt, b_tgt, morph)

    # Circuit breaker.
    cb = EndpointCircuitBreaker(failure_threshold=3)

    # Perform hybrid transport at t=0.4.
    t = 0.4
    v_hybrid, curvature = hybrid_transport(v_src, v_tgt, t, morph)

    # Update breaker based on the resulting flow.
    update_breaker_from_flow(cb, v_hybrid, morph)

    # Simple diagnostics.
    open_state, failures = cb.status()
    print(f"Curvature κ = {curvature:.4f}")
    print(f"Hybrid norm = {np.linalg.norm(v_hybrid):.4f}")
    print(f"CircuitBreaker open={open_state}, failures={failures}")


if __name__ == "__main__":
    _smoke_test()