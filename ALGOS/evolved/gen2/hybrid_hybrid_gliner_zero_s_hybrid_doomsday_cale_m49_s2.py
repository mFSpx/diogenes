# DARWIN HAMMER — match 49, survivor 2
# gen: 2
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s2.py (gen1)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s4.py (gen1)
# born: 2026-05-29T23:26:37Z

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"

def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def parse_labels(raw: str | None) -> List[str]:
    if raw is None:
        return DEFAULT_LABELS
    return [label.strip() for label in raw.split(",")]

def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    """
    Compute weekday indices for vectorised (year, month, day) arrays using
    Tomohiko Sakamoto's algorithm.  
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)

def gini_coefficient(values: np.ndarray) -> float:
    """
    Return the Gini coefficient of a 1‑D array of non‑negative numbers.
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)

def calculate_spatial_coherence(spans: List[Span]) -> float:
    # Calculate spatial coherence using minimum-cost tree with a more efficient algorithm
    points = np.array([(span.start, span.end - span.start) for span in spans])
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.mean(distances)

def calculate_hybrid_score(spans: List[Span]) -> float:
    # Calculate Gini coefficient of span lengths
    lengths = np.array([span.end - span.start for span in spans])
    gini = gini_coefficient(lengths)

    # Calculate spatial coherence
    spatial_coherence = calculate_spatial_coherence(spans)

    # Combine spatial coherence and Gini coefficient with a more meaningful formula
    hybrid_score = spatial_coherence * (1 + gini) / (1 + np.log(len(spans)))
    return hybrid_score

def calculate_weighted_spans(spans: List[Span], hybrid_score: float) -> List[Tuple[float, Span]]:
    # Calculate weighted spans using confidence scores and hybrid score
    weighted_spans = [(span.score * hybrid_score, span) for span in spans]
    return weighted_spans

def evaluate_extraction(spans: List[Span]) -> Dict[str, Any]:
    hybrid_score = calculate_hybrid_score(spans)
    weighted_spans = calculate_weighted_spans(spans, hybrid_score)
    return {
        "hybrid_score": hybrid_score,
        "weighted_spans": weighted_spans,
    }

if __name__ == "__main__":
    spans = [
        Span(0, 10, "example text", "Operator", 0.9),
        Span(15, 25, "another example", "Rainmaker", 0.8),
        Span(30, 40, "more text", "Paladin / God-Mode", 0.7),
    ]
    result = evaluate_extraction(spans)
    print(result)