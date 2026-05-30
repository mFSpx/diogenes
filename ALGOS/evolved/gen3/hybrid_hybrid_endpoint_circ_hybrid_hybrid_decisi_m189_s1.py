# DARWIN HAMMER — match 189, survivor 1
# gen: 3
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py (gen1)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s0.py (gen2)
# born: 2026-05-29T23:26:02Z

"""Hybrid Endpoint Decision Hygiene Module.

Parents:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4 (Algorithm A)
- hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s0 (Algorithm B)

Mathematical Bridge:
Algorithm A provides a *recovery priority* p ∈ [0,1] derived from morphology via
    p = min(1, righting_time_index / max_index).
Algorithm B produces a raw decision feature vector v ∈ ℝ⁹ from regex counts,
which is linearly transformed by positive/negative weight matrices to a score
vector s.

The hybrid combines them by scaling each component of s with the recovery
priority p, yielding a morphology‑aware decision vector ŝ = p·s.
Shannon entropy H(ŝ) of the normalized |ŝ| then quantifies the diversity of
the morphology‑adjusted decision landscape.

The module implements:
1. Morphology‑based priority computation (A).
2. Textual feature extraction and weighted scoring (B).
3. Hybrid scoring and entropy evaluation (bridge).

"""

import math
import re
import sys
import random
import pathlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


# ---------------------------- Algorithm A ---------------------------------

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


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting time index to a unit interval."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ---------------------------- Algorithm B ---------------------------------

_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Compile regexes for each feature
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
_OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|"
    r"delivered|finalized|closed)\b",
    re.I,
)
_IMPULSIVE_RE = re.compile(r"\b(?:impulse|rash|spontaneous|react|snap|unplanned)\b", re.I)
_SCARCITY_RE = re.compile(r"\b(?:scarcity|rare|limited|few|shortage|exhausted)\b", re.I)
_RISK_RE = re.compile(r"\b(?:risk|danger|hazard|threat|exposure|vulnerability)\b", re.I)

_REGEX_MAP = {
    "evidence": _EVIDENCE_RE,
    "planning": _PLANNING_RE,
    "delay": _DELAY_RE,
    "support": _SUPPORT_RE,
    "boundary": _BOUNDARY_RE,
    "outcome": _OUTCOME_RE,
    "impulsive": _IMPULSIVE_RE,
    "scarcity": _SCARCITY_RE,
    "risk": _RISK_RE,
}


def extract_feature_counts(text: str) -> np.ndarray:
    """Return raw occurrence counts for each feature in _FEATURE_ORDER."""
    counts = []
    lowered = text.lower()
    for feat in _FEATURE_ORDER:
        regex = _REGEX_MAP[feat]
        matches = regex.findall(lowered)
        counts.append(len(matches))
    return np.array(counts, dtype=np.int64)


def weighted_decision_vector(counts: np.ndarray) -> np.ndarray:
    """Apply positive and negative weight matrices to raw counts."""
    pos = _POSITIVE_WEIGHTS * counts
    neg = _NEGATIVE_WEIGHTS * counts
    return pos - neg  # positive influence minus negative influence


# ---------------------------- Hybrid Bridge -------------------------------

def hybrid_score(text: str, morphology: Morphology) -> np.ndarray:
    """
    Compute morphology‑aware decision scores.

    Steps:
    1. Extract raw feature counts from `text`.
    2. Transform counts to a decision vector via `weighted_decision_vector`.
    3. Compute recovery priority `p` from `morphology`.
    4. Scale the decision vector by `p` to obtain the hybrid score.
    """
    counts = extract_feature_counts(text)
    decision_vec = weighted_decision_vector(counts).astype(float)
    p = recovery_priority(morphology)
    return p * decision_vec


def shannon_entropy_from_scores(scores: np.ndarray) -> float:
    """
    Compute Shannon entropy of the absolute score distribution.

    The scores are first converted to a probability distribution by taking
    absolute values, normalising, and then applying the classic entropy formula:
        H = - Σ_i p_i log2(p_i)
    """
    abs_vals = np.abs(scores)
    total = np.sum(abs_vals)
    if total == 0:
        return 0.0
    probs = abs_vals / total
    # Guard against log2(0) by filtering zero probabilities
    nonzero = probs[probs > 0]
    return -float(np.sum(nonzero * np.log2(nonzero)))


def evaluate_endpoint(text: str, morphology: Morphology) -> Dict[str, float]:
    """
    Produce a compact report for a given endpoint description.

    Returns a dictionary containing:
        - priority: recovery priority p
        - raw_score: sum of unscaled decision vector
        - hybrid_score: sum of morphology‑scaled decision vector
        - entropy: Shannon entropy of the hybrid score vector
    """
    p = recovery_priority(morphology)
    raw_vec = weighted_decision_vector(extract_feature_counts(text)).astype(float)
    hybrid_vec = p * raw_vec
    return {
        "priority": p,
        "raw_score": float(np.sum(raw_vec)),
        "hybrid_score": float(np.sum(hybrid_vec)),
        "entropy": shannon_entropy_from_scores(hybrid_vec),
    }


# ---------------------------- Supporting Classes ---------------------------

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker (Parent A)."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open


class HybridEngineEndpointPool:
    """
    Pool that stores EngineEndpoint objects and evaluates them with the hybrid
    decision hygiene algorithm. Endpoints that fail the circuit breaker are
    automatically excluded from `available_endpoints`.
    """

    def __init__(self, failure_threshold: int = 3):
        self.endpoints: Dict[str, EngineEndpoint] = {}
        self.breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.failure_threshold = failure_threshold

    def add_endpoint(self, ep: EngineEndpoint) -> None:
        self.endpoints[ep.engine_id] = ep
        self.breakers[ep.engine_id] = EndpointCircuitBreaker(self.failure_threshold)

    def record_result(self, engine_id: str, success: bool) -> None:
        breaker = self.breakers.get(engine_id)
        if not breaker:
            return
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def available_endpoints(self) -> List[EngineEndpoint]:
        """Return endpoints whose circuit breaker is closed (allowing traffic)."""
        return [
            ep
            for eid, ep in self.endpoints.items()
            if self.breakers[eid].allow()
        ]

    def rank_by_hybrid_score(self, text: str) -> List[Dict]:
        """
        Rank currently available endpoints according to their hybrid_score sum.
        Returns a list of dictionaries with endpoint id and score.
        """
        ranked = []
        for ep in self.available_endpoints():
            vec = hybrid_score(text, ep.morphology)
            ranked.append({"engine_id": ep.engine_id, "score": float(np.sum(vec))})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


# ---------------------------- Smoke Test ----------------------------------

if __name__ == "__main__":
    # Create a dummy morphology
    demo_morph = Morphology(length=2.0, width=1.5, height=0.5, mass=3.0)

    # Sample text containing various cues
    sample_text = (
        "The evidence was verified and the plan was checklist based. "
        "We will wait tomorrow before proceeding, but the team has support. "
        "There is a risk of scarcity and impulsive decisions."
    )

    # Compute hybrid score vector and entropy
    vec = hybrid_score(sample_text, demo_morph)
    ent = shannon_entropy_from_scores(vec)
    report = evaluate_endpoint(sample_text, demo_morph)

    print("Hybrid score vector:", vec)
    print("Entropy:", ent)
    print("Endpoint evaluation report:", report)

    # Demonstrate pool usage
    pool = HybridEngineEndpointPool()
    pool.add_endpoint(
        EngineEndpoint(
            engine_id="engine_001",
            channel="channel_a",
            residency="local",
            runtime="python3.11",
            resource_class="standard",
            always_on=True,
            endpoint="http://localhost:8000",
            capabilities=["compute", "io"],
            morphology=demo_morph,
        )
    )
    # Simulate a failure then success
    pool.record_result("engine_001", success=False)
    pool.record_result("engine_001", success=True)

    ranking = pool.rank_by_hybrid_score(sample_text)
    print("Ranking of available endpoints:", ranking)