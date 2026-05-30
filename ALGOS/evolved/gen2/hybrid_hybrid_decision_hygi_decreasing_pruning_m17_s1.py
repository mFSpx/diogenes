# DARWIN HAMMER — match 17, survivor 1
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:22:55Z

"""
This module represents a novel HYBRID algorithm, mathematically fusing the core topologies of 
hybrid_decision_hygiene_shannon_entropy_m12_s4.py and decreasing_pruning.py into a single unified system.
The bridge between the two parents lies in the application of Shannon entropy to the feature vectors 
extracted by the decision-hygiene algorithm, and the use of a decreasing-rate pruning schedule to 
select the most informative features. This allows for a more efficient and effective decision-making 
process, by pruning away less relevant features and focusing on those with the highest information content.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = __import__(re).compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    __import__(re).I,
)
PLANNING_RE = __import__(re).compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    __import__(re).I,
)
DELAY_RE = __import__(re).compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    __import__(re).I,
)
SUPPORT_RE = __import__(re).compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    __import__(re).I,
)
BOUNDARY_RE = __import__(re).compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    __import__(re).I,
)
OUTCOME_RE = __import__(re).compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    __import__(re).I,
)
IMPULSIVE_RE = __import__(re).compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    __import__(re).I,
)
SCARCITY_RE = __import__(re).compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    __import__(re).I,
)
RISK_RE = __import__(re).compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    __import__(re).I,
)

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

# ----------------------------------------------------------------------
# Parent B – decreasing-rate pruning schedule
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

def prune_edges(edges: list, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> list:
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using parent-A regexes."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def feature_vector(text: str) -> np.ndarray:
    """Return a 9-element ``numpy`` array of counts ordered as _FEATURE_ORDER."""
    c = _raw_counts(text)
    return np.array(
        [
            c["evidence_count"],
            c["planning_count"],
            c["delay_count"],
            c["support_count"],
            c["boundary_count"],
            c["outcome_count"],
            c["impulsive_count"],
            c["scarcity_count"],
            c["risk_count"],
        ],
        dtype=np.int64,
    )

def shannon_entropy(observations: Iterable[float | Any], is_distribution: bool = False) -> float:
    """Return Shannon entropy (bits) of a discrete distribution."""
    xs = list(observations)
    if not xs:
        return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        cnt = Counter(xs)
        total = float(sum(cnt.values()))
        probs = [v / total for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0.0)

def entropy_from_counts(vector: np.ndarray) -> float:
    """Treat *vector* as raw observations and return its Shannon entropy (bits)."""
    if vector.sum() == 0:
        return 0.0
    probs = vector[vector > 0] / float(vector.sum())
    return -float(np.sum(pr * math.log2(pr) for pr in probs if pr > 0.0))

def prune_features(vector: np.ndarray, t: float, lam: float = 1.0, alpha: float = 0.2, seed: int | str | None = None) -> np.ndarray:
    """Prune features based on their entropy and the decreasing-rate pruning schedule."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return np.array([x for x in vector if rng.random() >= p or entropy_from_counts(np.array([x])) > 0.5])

def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
    """Compute the original decision-hygiene score and a textual label."""
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))

    # label logic (unchanged)
    risk_present = vector[6] > 0 or vector[7] > 0 or vector[8] > 0  # impulsive, scarcity, risk counts
    if risk_present and raw_score < 2500:
        label = "critical_risk_or_pain_signal"
    elif raw_score >= 7000:
        label = "high_decision_hygiene"
    elif raw_score >= 3000:
        label = "improving_decision_hygiene"
    elif raw_score <= -2500:
        label = "strained_decision_context"
    else:
        label = "neutral_or_unclear"
    return raw_score, label

if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    vector = feature_vector(text)
    pruned_vector = prune_features(vector, 1.0)
    score, label = hygiene_score(pruned_vector)
    print(f"Hygiene score: {score}, Label: {label}")