# DARWIN HAMMER — match 1035, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py (gen3)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py (gen1)
# born: 2026-05-29T23:32:28Z

"""Hybrid module integrating:
- Parent A: regex‑based evidence/planning feature extraction (hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py)
- Parent B: morphology‑driven recovery priority and circuit breaker logic (hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s1.py)

Mathematical bridge:
The recovery priority `ρ = recovery_priority(morphology)` (∈[0,1]) derived from the
righting‑time index of the morphology is used as a probabilistic weight for the
feature‑count vector `v = [evidence_count, planning_count]` extracted by the
regexes of Parent A. The expected weighted vector is

    𝔼[v] = ρ · v

which blends deterministic textual evidence with a morphology‑dependent confidence.
All hybrid operations are built on this expectation, e.g. audit scores are cosine
similarities between `𝔼[v]` and a reference vector, and the circuit‑breaker
threshold is dynamically scaled by `ρ`."""
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple, Dict

import numpy as np
import re

# ----------------------------------------------------------------------
# Parent A – regex feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def _extract_counts(text: str) -> Dict[str, int]:
    """Return raw counts of evidence‑related and planning‑related tokens."""
    evidence = len(EVIDENCE_RE.findall(text))
    planning = len(PLANNING_RE.findall(text))
    return {"evidence": evidence, "planning": planning}


def feature_vector(text: str) -> np.ndarray:
    """Convert raw counts to a 2‑element NumPy column vector."""
    cnts = _extract_counts(text)
    return np.array([cnts["evidence"], cnts["planning"]], dtype=float).reshape(2, 1)


# ----------------------------------------------------------------------
# Parent B – morphology & recovery priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized confidence (0‑1) that the system will recover, based on morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


class EndpointCircuitBreaker:
    """Simple circuit breaker whose failure threshold adapts to recovery priority."""

    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None):
        self.base_threshold = failure_threshold
        self.morphology = morphology
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def _now_z(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @property
    def dynamic_threshold(self) -> int:
        """Threshold scaled by recovery priority (higher priority → higher tolerance)."""
        if self.morphology is None:
            return self.base_threshold
        rho = recovery_priority(self.morphology)
        # Scale linearly between 0.5× and 2× the base threshold
        scale = 0.5 + 1.5 * rho
        return max(1, int(round(self.base_threshold * scale)))

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = self._now_z()
        if self.failures >= self.dynamic_threshold:
            self.open = True

    def reset(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()


# ----------------------------------------------------------------------
# Hybrid operations (mathematical fusion)
# ----------------------------------------------------------------------
def expected_feature_vector(text: str, morph: Morphology) -> np.ndarray:
    """
    Compute the expected (weighted) feature vector:
        𝔼[v] = ρ · v
    where ρ = recovery_priority(morph) and v is the raw count vector from Parent A.
    """
    v = feature_vector(text)                # shape (2,1)
    rho = recovery_priority(morph)          # scalar ∈ [0,1]
    return rho * v


def hybrid_audit_score(text: str, morph: Morphology, reference: np.ndarray = None) -> float:
    """
    Cosine similarity between the expected feature vector and a reference vector.
    If no reference is supplied, use a unit vector [1, 1]^T.
    """
    ev = expected_feature_vector(text, morph)  # (2,1)
    if reference is None:
        reference = np.ones((2, 1), dtype=float)
    # Flatten for cosine computation
    ev_flat = ev.ravel()
    ref_flat = reference.ravel()
    dot = float(np.dot(ev_flat, ref_flat))
    norm_ev = float(np.linalg.norm(ev_flat))
    norm_ref = float(np.linalg.norm(ref_flat))
    if norm_ev == 0 or norm_ref == 0:
        return 0.0
    return dot / (norm_ev * norm_ref)


def hybrid_circuit_breaker_step(
    breaker: EndpointCircuitBreaker,
    text: str,
    success_condition: bool,
) -> Tuple[bool, float]:
    """
    Perform one evaluation step:
    * Update the breaker using the dynamic threshold derived from morphology.
    * Return a tuple (circuit_open, audit_score) where `audit_score` is the hybrid
      audit similarity for the supplied text.
    """
    if success_condition:
        breaker.record_success()
    else:
        breaker.record_failure()
    score = hybrid_audit_score(text, breaker.morphology)
    return breaker.open, score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The audit confirmed the evidence and source. "
        "We plan the next steps in the checklist and timeline."
    )
    # Example morphology resembling a compact robot
    morph = Morphology(length=0.5, width=0.4, height=0.3, mass=2.0)

    # Instantiate a circuit breaker with the morphology
    breaker = EndpointCircuitBreaker(failure_threshold=3, morphology=morph)

    # Simulate a series of events
    for i in range(5):
        # Arbitrary success condition: even iterations succeed
        success = (i % 2 == 0)
        open_state, audit = hybrid_circuit_breaker_step(breaker, sample_text, success)
        print(
            f"Step {i+1}: success={success}, breaker_open={open_state}, "
            f"audit_score={audit:.4f}, dynamic_threshold={breaker.dynamic_threshold}"
        )
    sys.exit(0)