# DARWIN HAMMER — match 2757, survivor 8
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s2.py (gen4)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s0.py (gen2)
# born: 2026-05-29T23:45:47Z

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A core structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(morph: Morphology) -> float:
    """
    Ratio of the geometric mean of the three spatial dimensions to the longest dimension.

    S = (l·w·h)^{1/3} / max(l,w,h)

    Returns 0 when any dimension is non‑positive.
    """
    dims = np.array([morph.length, morph.width, morph.height], dtype=float)
    if np.any(dims <= 0):
        return 0.0
    geo_mean = np.prod(dims) ** (1.0 / 3.0)
    longest = np.max(dims)
    return geo_mean / longest


class EndpointCircuitBreaker:
    """
    Simple failure counter that opens after a configurable threshold.
    The health score H ∈ [0,1] is defined as H = 1 - failures/threshold,
    clipped to the interval [0,1].
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = max(1, failure_threshold)
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

    @property
    def health(self) -> float:
        """Health score H = 1 - failures/threshold, bounded in [0,1]."""
        h = 1.0 - (self.failures / self.failure_threshold)
        return max(0.0, min(1.0, h))


# ----------------------------------------------------------------------
# Parent B core structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """A textual extraction with positional metadata."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

    @property
    def length(self) -> int:
        return self.end - self.start


def embed_spans(spans: List[Span]) -> np.ndarray:
    """
    Convert a list of Span objects into a 2‑D embedding:
    column 0 -> start position (x), column 1 -> span length (y).
    """
    if not spans:
        return np.empty((0, 2), dtype=int)
    starts = np.array([s.start for s in spans], dtype=int)
    lengths = np.array([s.length for s in spans], dtype=int)
    return np.column_stack((starts, lengths))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a 1‑D array of non‑negative values.
    G = (1/(n·μ)) * Σ_i Σ_j |x_i - x_j| / (2·n²·μ)
    Simplified implementation using sorted values.
    Returns 0 for empty or constant arrays.
    """
    if values.size == 0:
        return 0.0
    if np.all(values == values[0]):
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    sum_vals = cumulative[-1]
    if sum_vals == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_vals) / n
    return max(0.0, min(1.0, gini))


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def compute_unified_metric(
    morph: Morphology,
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
) -> Dict[str, float]:
    """
    Core hybrid function that fuses the topologies of both parents.

    Steps:
    1. Compute sphericity S from the Morphology.
    2. Derive health score H from the circuit breaker.
    3. Embed spans and obtain the length distribution L.
    4. Compute Gini coefficient G on L.
    5. Combine them: U = H * S * (1 - G).

    Returns a dictionary with the intermediate values and the final unified score.
    """
    S = sphericity_index(morph)
    H = breaker.health

    if not spans:
        return {
            "sphericity": S,
            "health": H,
            "gini": 0.0,
            "unified_metric": 0.0,
        }

    embedding = embed_spans(spans)
    lengths = embedding[:, 1] if embedding.size else np.array([], dtype=int)
    G = gini_coefficient(lengths)

    unified = H * S * (1.0 - G)

    return {
        "sphericity": S,
        "health": H,
        "gini": G,
        "unified_metric": unified,
    }


def weighted_span_scores(
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
) -> List[Dict[str, Any]]:
    """
    Produce a SHAP‑like attribution where each span's original confidence score is
    modulated by the hybrid health‑sphericity factor H·S.
    Returns a list of dictionaries with original and weighted scores.
    """
    factor = breaker.health * sphericity_index(morph)
    results = []
    for sp in spans:
        weighted = sp.score * factor
        results.append(
            {
                "span": sp,
                "original_score": sp.score,
                "weighted_score": weighted,
            }
        )
    return results


def simulate_circuit_breaker(
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
    threshold_score: float = 0.5,
) -> None:
    """
    Simple simulation: a span with score < threshold is considered a failure;
    otherwise a success. Updates the breaker state accordingly.
    """
    for sp in spans:
        if sp.score < threshold_score:
            breaker.record_failure()
        else:
            breaker.record_success()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a sample morphology
    morph = Morphology(length=3.0, width=2.0, height=1.5, mass=10.0)

    # Initialise a circuit breaker with a low threshold to see state changes
    breaker = EndpointCircuitBreaker(failure_threshold=4)

    # Create a synthetic list of spans
    sample_spans = [
        Span(start=0, end=10, text="alpha", label="Operator", score=0.92),
        Span(start=15, end=25, text="beta", label="Rainmaker", score=0.47),
        Span(start=30, end=45, text="gamma", label="Paladin / God-Mode", score=0.78),
        Span(start=50, end=58, text="delta", label="Psyche / State-Collapse", score=0.33),
    ]

    # Run the simulation that may open/close the breaker
    simulate_circuit_breaker(sample_spans, breaker, threshold_score=0.5)

    # Compute the unified metric
    result = compute_unified_metric(morph, sample_spans, breaker)
    print("Unified metric components:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")

    # Compute weighted span scores
    weighted = weighted_span_scores(sample_spans, breaker, morph)
    print("\nWeighted scores:")
    for w in weighted:
        print(f"  Span: {w['span'].text}, Original: {w['original_score']:.2f}, Weighted: {w['weighted_score']:.2f}")