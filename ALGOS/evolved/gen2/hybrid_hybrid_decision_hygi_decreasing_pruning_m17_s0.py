# DARWIN HAMMER — match 17, survivor 0
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s4.py (gen1)
# parent_b: decreasing_pruning.py (gen0)
# born: 2026-05-29T23:22:55Z

"""
This module implements a hybrid algorithm that combines the decision-hygiene scoring from 'hybrid_decision_hygiene_shannon_entropy_m12_s4.py' 
with the decreasing-rate pruning schedule from 'decreasing_pruning.py'. The mathematical bridge between the two structures is the use of 
Shannon entropy to weigh the importance of different features in the decision-hygiene scoring. The pruning schedule is used to dynamically 
adjust the weights based on the time step 't'.

The hybrid algorithm integrates the governing equations of both parents by using the prune_probability function to adjust the weights 
used in the hygiene_score function. This allows the algorithm to adapt to changing conditions over time.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Iterable, List, Tuple

# Parent A – regexes and raw count extraction
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)

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

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non-negative')
    return min(1.0, lam * math.exp(-alpha * t))

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

def hygiene_score(vector: np.ndarray, t: float) -> Tuple[int, str]:
    positive_weights = _POSITIVE_WEIGHTS * (1 - prune_probability(t))
    negative_weights = _NEGATIVE_WEIGHTS * prune_probability(t)
    positive = int(np.dot(vector, positive_weights))
    negative = int(np.dot(vector, negative_weights))
    raw_score = max(-10000, min(10000, positive - negative))

    risk_present = vector[6] > 0 or vector[7] > 0 or vector[8] > 0
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

def entropy_from_counts(vector: np.ndarray) -> float:
    if vector.sum() == 0:
        return 0.0
    probs = vector[vector > 0] / float(vector.sum())
    return -float(np.sum(probs * np.log2(probs)))

def hybrid_score(text: str, t: float) -> Tuple[int, str, float]:
    vector = feature_vector(text)
    score, label = hygiene_score(vector, t)
    entropy = entropy_from_counts(vector)
    return score, label, entropy

if __name__ == "__main__":
    text = "This is a sample text with some evidence and planning."
    t = 1.0
    score, label, entropy = hybrid_score(text, t)
    print(f"Score: {score}, Label: {label}, Entropy: {entropy}")