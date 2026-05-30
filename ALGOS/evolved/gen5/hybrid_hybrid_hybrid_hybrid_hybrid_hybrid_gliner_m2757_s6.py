# DARWIN HAMMER — match 2757, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s2.py (gen4)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s0.py (gen2)
# born: 2026-05-29T23:45:47Z

"""
Hybrid Fusion Module
====================

This module combines the core topologies of the two parent algorithms:

* **Parent A** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s2.py`
  - Provides `EndpointCircuitBreaker` (failure counter) and the geometric
    `sphericity_index` based on a `Morphology` description.
* **Parent B** – `hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s0.py`
  - Provides a span extraction model, a 2‑D embedding of spans
    (x = start, y = length) and a Gini coefficient computed on the span
    lengths.

**Mathematical Bridge**

Both parents expose a *scalar quality* that can be used as a weight:

* **Health weight** – `EndpointCircuitBreaker.health_score()` returns a value in
  `[0, 1]` reflecting the circuit‑breaker state.
* **Shape weight** – `sphericity_index(morphology)` returns a value in `[0, 1]`
  describing how close the entity is to a sphere.

These two weights are multiplied to obtain a *global morphology weight* `w_m`.
The Gini coefficient `G` (inequality of span lengths) lives in `[0, 1]`.
We map `G` to a *coherence factor* `c = 1 - G` (high coherence ↔ low inequality).

The final hybrid metric is defined as:


HybridScore = w_m * c * Σ_i (span_i.score)


Thus the circuit‑breaker health, the geometric sphericity, and the
distributional coherence of extracted spans are fused into a single
scalar that can drive downstream decisions (e.g., recovery priority,
resource allocation, etc.).
"""

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared from parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Span:
    """Extracted text span with confidence."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


# ----------------------------------------------------------------------
# Parent A – Circuit breaker and sphericity
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        """Reset failure count and close the breaker."""
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        """Increment failure count; open the breaker if threshold exceeded."""
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    def health_score(self) -> float:
        """
        Return a health weight in [0, 1].
        - 1.0 when the breaker is closed (healthy)
        - linearly decreasing to 0.0 as failures approach the threshold
        """
        if not self.open:
            return 1.0
        # When open, map failures to a decreasing weight
        excess = self.failures - self.failure_threshold
        # Clamp to [0, failure_threshold] for smooth decay
        decay = max(0, self.failure_threshold - excess) / self.failure_threshold
        return decay


def sphericity_index(morph: Morphology) -> float:
    """
    Ratio of the geometric mean of the three spatial dimensions to the longest dimension.
    The result lies in (0, 1]; a perfect sphere (all dimensions equal) yields 1.
    """
    dims = np.array([morph.length, morph.width, morph.height], dtype=float)
    geom_mean = np.prod(dims) ** (1.0 / 3.0)
    longest = np.max(dims)
    if longest == 0:
        return 0.0
    return geom_mean / longest


# ----------------------------------------------------------------------
# Parent B – Span embedding and Gini coefficient
# ----------------------------------------------------------------------


def embed_spans(spans: List[Span]) -> np.ndarray:
    """
    Map each span to a 2‑D point: (x = start, y = length).
    Returns an (N, 2) NumPy array.
    """
    points = [(s.start, s.end - s.start) for s in spans]
    return np.asarray(points, dtype=float)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array of non‑negative numbers.
    The coefficient is 0 for perfect equality and approaches 1 for maximal inequality.
    """
    if values.size == 0:
        return 0.0
    # Ensure non‑negative
    values = np.abs(values)
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    # The Gini formula based on the Lorenz curve
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def morphology_weight(morph: Morphology, breaker: EndpointCircuitBreaker) -> float:
    """
    Combine sphericity and circuit‑breaker health into a single weight.
    """
    sph = sphericity_index(morph)
    health = breaker.health_score()
    return sph * health


def coherence_factor(spans: List[Span]) -> float:
    """
    Derive a coherence factor from the Gini coefficient of span lengths.
    High coherence (balanced lengths) → factor close to 1.
    """
    if not spans:
        return 0.0
    lengths = np.array([s.end - s.start for s in spans], dtype=float)
    gini = gini_coefficient(lengths)
    return 1.0 - gini  # invert: lower inequality → higher factor


def hybrid_score(
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
    spans: List[Span],
) -> float:
    """
    Compute the unified hybrid metric:

        HybridScore = (sphericity * health) * (1 - Gini) * Σ(span.score)

    Returns a non‑negative float.
    """
    w_m = morphology_weight(morph, breaker)
    c = coherence_factor(spans)
    total_span_score = sum(s.score for s in spans)
    return w_m * c * total_span_score


def summarize_hybrid(
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
    spans: List[Span],
) -> Dict[str, float]:
    """
    Produce a dictionary with intermediate components and the final score.
    Useful for debugging or downstream consumption.
    """
    sph = sphericity_index(morph)
    health = breaker.health_score()
    w_m = sph * health
    gini = gini_coefficient(np.array([s.end - s.start for s in spans], dtype=float))
    c = 1.0 - gini
    total_span_score = sum(s.score for s in spans)
    final = w_m * c * total_span_score
    return {
        "sphericity": sph,
        "health": health,
        "morphology_weight": w_m,
        "gini": gini,
        "coherence_factor": c,
        "total_span_score": total_span_score,
        "hybrid_score": final,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a sample morphology
    sample_morph = Morphology(length=2.0, width=1.5, height=1.5, mass=3.0)

    # Initialise a circuit breaker and simulate some failures
    breaker = EndpointCircuitBreaker(failure_threshold=4)
    for _ in range(2):
        breaker.record_failure()
    # At this point the breaker is still closed (health = 1.0)

    # Generate synthetic spans
    sample_spans = [
        Span(start=0, end=5, text="alpha", label="Operator", score=0.9),
        Span(start=10, end=15, text="beta", label="Rainmaker", score=0.85),
        Span(start=20, end=22, text="γ", label="Paladin / God-Mode", score=0.95),
        Span(start=30, end=45, text="delta", label="DIOGENES", score=0.7),
    ]

    # Compute hybrid metric
    result = summarize_hybrid(sample_morph, breaker, sample_spans)

    # Pretty‑print the result
    for key, value in result.items():
        print(f"{key}: {value:.4f}")