# DARWIN HAMMER — match 49, survivor 1
# gen: 2
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s2.py (gen1)
# parent_b: hybrid_doomsday_calendar_gini_coefficient_m49_s4.py (gen1)
# born: 2026-05-29T23:26:37Z

"""
Hybrid algorithm that fuses the geometric embedding of 
`hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s2.py` 
with the Gini coefficient calculation of 
`hybrid_doomsday_calendar_gini_coefficient_m49_s4.py`.

The mathematical bridge is established by embedding the 
extracted spans from the GLiNER zero-shot extractor 
into a spatial representation, where each span is a 
point in 2D space with coordinates (start, length). 
The Gini coefficient is then used to evaluate the 
inequality of the span lengths, providing a measure 
of the diversity of the extracted information.

The hybrid algorithm calculates a weighted score that 
combines the spatial coherence of the extracted spans 
with the Gini coefficient of their lengths, 
yielding a unified metric that rewards both 
high-confidence spans and diverse, well-connected layouts.
"""

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

def calculate_hybrid_score(spans: List[Span]) -> float:
    # Calculate spatial coherence using minimum-cost tree
    points = [(span.start, span.end - span.start) for span in spans]
    distances = np.linalg.norm(np.array(points)[:, np.newaxis] - np.array(points), axis=2)
    tree_cost = np.sum(distances)

    # Calculate Gini coefficient of span lengths
    lengths = np.array([span.end - span.start for span in spans])
    gini = gini_coefficient(lengths)

    # Combine spatial coherence and Gini coefficient
    hybrid_score = tree_cost * (1 + gini)
    return hybrid_score

def calculate_weighted_spans(spans: List[Span]) -> List[Tuple[float, Span]]:
    # Calculate weighted spans using confidence scores and hybrid score
    hybrid_score = calculate_hybrid_score(spans)
    weighted_spans = [(span.score / hybrid_score, span) for span in spans]
    return weighted_spans

def evaluate_extraction(spans: List[Span]) -> Dict[str, Any]:
    hybrid_score = calculate_hybrid_score(spans)
    weighted_spans = calculate_weighted_spans(spans)
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