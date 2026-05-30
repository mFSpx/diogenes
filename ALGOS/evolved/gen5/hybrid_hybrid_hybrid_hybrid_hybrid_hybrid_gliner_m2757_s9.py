# DARWIN HAMMER — match 2757, survivor 9
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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(morph: Morphology) -> float:
    dims = np.array([morph.length, morph.width, morph.height], dtype=float)
    if np.any(dims <= 0):
        return 0.0
    geo_mean = np.prod(dims) ** (1.0 / 3.0)
    longest = np.max(dims)
    return geo_mean / longest

class EndpointCircuitBreaker:
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
        h = 1.0 - (self.failures / self.failure_threshold)
        return max(0.0, min(1.0, h))

@dataclass(frozen=True)
class Span:
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
    if not spans:
        return np.empty((0, 2), dtype=int)
    starts = np.array([s.start for s in spans], dtype=int)
    lengths = np.array([s.length for s in spans], dtype=int)
    return np.column_stack((starts, lengths))

def gini_coefficient(values: np.ndarray) -> float:
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

def compute_unified_metric(
    morph: Morphology,
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
) -> Dict[str, float]:
    S = sphericity_index(morph)
    H = breaker.health

    embedding = embed_spans(spans)
    lengths = embedding[:, 1] if embedding.size else np.array([], dtype=int)
    G = gini_coefficient(lengths)

    unified = H * S * (1.0 - G)
    harmonic_mean = 2 * H * S / (H + S) if H + S > 0 else 0
    unified_improved = harmonic_mean * (1.0 - G)

    return {
        "sphericity": S,
        "health": H,
        "gini": G,
        "unified_metric": unified,
        "unified_metric_improved": unified_improved,
    }

def weighted_span_scores(
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
) -> List[Dict[str, Any]]:
    factor = breaker.health * sphericity_index(morph)
    harmonic_mean = 2 * breaker.health * sphericity_index(morph) / (breaker.health + sphericity_index(morph)) if breaker.health + sphericity_index(morph) > 0 else 0
    results = []
    for sp in spans:
        weighted = sp.score * factor
        weighted_improved = sp.score * harmonic_mean
        results.append(
            {
                "span": sp,
                "original_score": sp.score,
                "weighted_score": weighted,
                "weighted_score_improved": weighted_improved,
            }
        )
    return results

def simulate_circuit_breaker(
    spans: List[Span],
    breaker: EndpointCircuitBreaker,
    threshold_score: float = 0.5,
) -> None:
    for sp in spans:
        if sp.score < threshold_score:
            breaker.record_failure()
        else:
            breaker.record_success()

if __name__ == "__main__":
    morph = Morphology(length=3.0, width=2.0, height=1.5, mass=10.0)
    breaker = EndpointCircuitBreaker(failure_threshold=4)
    sample_spans = [
        Span(start=0, end=10, text="alpha", label="Operator", score=0.92),
        Span(start=15, end=25, text="beta", label="Rainmaker", score=0.47),
        Span(start=30, end=45, text="gamma", label="Paladin / God-Mode", score=0.78),
        Span(start=50, end=58, text="delta", label="Psyche / State-Collapse", score=0.33),
    ]
    simulate_circuit_breaker(sample_spans, breaker, threshold_score=0.5)
    result = compute_unified_metric(morph, sample_spans, breaker)
    print("Unified metric components:")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    weighted = weighted_span_scores(sample_spans, breaker, morph)
    print("\nWeighted span scores:")
    for w in weighted:
        print(f"  Original score: {w['original_score']:.4f}, Weighted score: {w['weighted_score']:.4f}, Weighted score improved: {w['weighted_score_improved']:.4f}")