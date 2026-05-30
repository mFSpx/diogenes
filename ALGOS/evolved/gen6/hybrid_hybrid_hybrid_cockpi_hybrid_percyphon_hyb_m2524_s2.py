# DARWIN HAMMER — match 2524, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_cockpit_metri_hybrid_hybrid_hybrid_m608_s2.py (gen5)
# parent_b: hybrid_percyphon_hybrid_endpoint_circ_m45_s1.py (gen2)
# born: 2026-05-29T23:42:46Z

import numpy as np
import math
import random
import sys
import pathlib
from typing import List, Callable, Any
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def words(text: str) -> List[str]:
    """Split a string into lowercase alphabetic words."""
    return [word for word in (text or "").lower().split() if word.isalpha()]

# ----------------------------------------------------------------------
# Metric calculations (original “cockpit_metrics”)
# ----------------------------------------------------------------------
def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Proportion of claims that are backed by evidence."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are truly OK."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))

# ----------------------------------------------------------------------
# Stylometry (original “hybrid_hybrid_hybrid_hard_truth_ma_kan_m27_s4”)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "ours"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at", "before",
        "behind", "below", "between", "by", "during", "for", "from", "in",
        "into", "of", "off", "on", "onto", "over", "through", "to", "under",
        "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would"
    },
    "conjunction": {}
}

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        return 0.0
    surface_area = 2 * (length * width + width * height + height * length)
    volume = length * width * height
    return (np.pi ** (1/3)) * (6 * volume) ** (2/3) / surface_area

def flatness_index(length: float, width: float) -> float:
    if min(length, width) <= 0:
        return 0.0
    return min(length / width, width / length)

def calculate_hybrid_metric(morphology: Morphology, claims_with_evidence: int, total_claims_emitted: int, displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    flatness = flatness_index(morphology.length, morphology.width)
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    cockpit_honest = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    return sphericity ** 0.25 * anti_slop ** 0.25 * cockpit_honest ** 0.25 * flatness ** 0.25

def hybrid_operation(morphology: Morphology, claims_with_evidence: int, total_claims_emitted: int, displayed_ok: int, unknown_displayed_as_ok: int) -> None:
    hybrid_metric = calculate_hybrid_metric(morphology, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok)
    print(f"Hybrid Metric: {hybrid_metric}")

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

class EndpointCircuitBreaker:
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
        return not self.open

def integrate_circuit_breaker_with_hybrid_metric(circuit_breaker: EndpointCircuitBreaker, morphology: Morphology, claims_with_evidence: int, total_claims_emitted: int, displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    if not circuit_breaker.allow():
        return 0.0
    return calculate_hybrid_metric(morphology, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok)

if __name__ == "__main__":
    morphology = Morphology(10.0, 5.0, 2.0)
    claims_with_evidence = 100
    total_claims_emitted = 200
    displayed_ok = 50
    unknown_displayed_as_ok = 20

    circuit_breaker = EndpointCircuitBreaker()
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    print(integrate_circuit_breaker_with_hybrid_metric(circuit_breaker, morphology, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok))