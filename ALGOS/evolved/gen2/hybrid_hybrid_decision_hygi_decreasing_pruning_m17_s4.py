# DARWIN HAMMER — match 17, survivor 4
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:22:55Z

import re
import math
import random
import sys
from pathlib import Path
from collections import Counter
import numpy as np

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
    re.I,
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

def _raw_counts(text: str) -> dict[str, int]:
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

def hygiene_score(vector: np.ndarray) -> tuple[int, str]:
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))

    risk_present = any(vector[idx] > 0 for idx in (6, 7, 8))
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

def shannon_entropy(vector: np.ndarray) -> float:
    if vector.sum() == 0:
        return 0.0
    probs = vector[vector > 0].astype(float) / float(vector.sum())
    return -float(np.sum(probs * np.log2(probs)))

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non-negative")
    return min(1.0, lam * math.exp(-alpha * t))

def entropy_normalisation_factor(vector: np.ndarray) -> float:
    k = np.count_nonzero(vector)
    if k <= 1:
        return 1.0
    h = shannon_entropy(vector)
    h_max = math.log2(k)
    return 1.0 + (h / h_max)

def hybrid_prune_probability(t: float, vector: np.ndarray, lam: float = 1.0, alpha: float = 0.2) -> float:
    base_p = prune_probability(t, lam, alpha)
    gamma = entropy_normalisation_factor(vector)
    return min(1.0, base_p / gamma)

def hybrid_score(text: str, t: float, lam: float = 1.0, alpha: float = 0.2) -> tuple[float, str]:
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)
    p_hybrid = hybrid_prune_probability(t, vec, lam, alpha)
    hybrid_score = raw_score * (1 - p_hybrid)
    return hybrid_score, label

def improved_hybrid_score(text: str, t: float, lam: float = 1.0, alpha: float = 0.2) -> tuple[float, str]:
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)
    h = shannon_entropy(vec)
    h_max = math.log2(np.count_nonzero(vec))
    p_hybrid = prune_probability(t, lam, alpha) / (1 + h / h_max)
    hybrid_score = raw_score * (1 - p_hybrid)
    return hybrid_score, label

def improved_hybrid_prune_probability(t: float, vector: np.ndarray, lam: float = 1.0, alpha: float = 0.2) -> float:
    base_p = prune_probability(t, lam, alpha)
    h = shannon_entropy(vector)
    h_max = math.log2(np.count_nonzero(vector))
    return min(1.0, base_p / (1 + h / h_max))

def main():
    text = "This is a test text with evidence and planning."
    t = 1.0
    score, label = improved_hybrid_score(text, t)
    print(f"Hybrid Score: {score}, Label: {label}")

if __name__ == "__main__":
    main()