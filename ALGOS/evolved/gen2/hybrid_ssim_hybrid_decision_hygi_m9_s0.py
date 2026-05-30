# DARWIN HAMMER — match 9, survivor 0
# gen: 2
# parent_a: ssim.py (gen0)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# born: 2026-05-29T23:22:30Z

"""
This module defines a novel hybrid algorithm that combines the structural similarity 
index measurement (ssim) from ssim.py with the decision hygiene scoring and 
Shannon entropy calculation from hybrid_decision_hygiene_shannon_entropy_m12_s5.py.

The mathematical bridge between these two parents lies in the application of ssim to 
compare the similarity between feature vectors extracted from text, and then using 
the result as a weighting factor in the calculation of the hybrid score. This allows 
for a more nuanced evaluation of decision hygiene based on the similarity between 
the input text and a set of reference texts.
"""

import numpy as np
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple
import random
import sys

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
_TOTAL_ABS_WEIGHTS = np.abs(_POSITIVE_WEIGHTS) + np.abs(_NEGATIVE_WEIGHTS)

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


def ssim(x: list[float], y: list[float], dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    vx = sum((v - mx) ** 2 for v in x) / n
    vy = sum((v - my) ** 2 for v in y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y)) / n
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def _raw_counts(text: str) -> dict[str, int]:
    txt = text or ""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(txt)),
        "planning_count": len(PLANNING_RE.findall(txt)),
        "delay_count": len(DELAY_RE.findall(txt)),
        "support_count": len(SUPPORT_RE.findall(txt)),
        "boundary_count": len(BOUNDARY_RE.findall(txt)),
        "outcome_count": len(OUTCOME_RE.findall(txt)),
        "impulsive_count": len(IMPULSIVE_RE.findall(txt)),
        "scarcity_count": len(SCARCITY_RE.findall(txt)),
        "risk_count": len(RISK_RE.findall(txt)),
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


def _weighted_entropy(vector: np.ndarray, weights: np.ndarray = _TOTAL_ABS_WEIGHTS) -> float:
    if vector.sum() == 0:
        return 0.0

    weighted = vector.astype(np.float64) * weights.astype(np.float64)
    nonzero = weighted[weighted > 0]
    if nonzero.size == 0:
        return 0.0

    probs = nonzero / nonzero.sum()
    return -float(np.sum(probs * np.log2(probs)))


def hybrid_score(text: str, reference_text: str, *, beta: float = 0.5, pos_weights: np.ndarray = _POSITIVE_WEIGHTS, neg_weights: np.ndarray = _NEGATIVE_WEIGHTS, entropy_weights: np.ndarray = _TOTAL_ABS_WEIGHTS) -> Tuple[float, dict[str, Any]]:
    vector = feature_vector(text)
    reference_vector = feature_vector(reference_text)
    similarity = ssim(vector.tolist(), reference_vector.tolist())
    hygiene_score_val, label = hygiene_score(vector, pos_weights, neg_weights)
    entropy = _weighted_entropy(vector, entropy_weights)
    hybrid_val = beta * hygiene_score_val + (1 - beta) * entropy * similarity
    return hybrid_val, {"hygiene_score": hygiene_score_val, "label": label, "entropy": entropy, "similarity": similarity}


def hygiene_score(vector: np.ndarray, pos_weights: np.ndarray = _POSITIVE_WEIGHTS, neg_weights: np.ndarray = _NEGATIVE_WEIGHTS) -> Tuple[int, str]:
    positive = int(np.dot(vector, pos_weights))
    negative = int(np.dot(vector, neg_weights))
    raw_score = max(-10_000, min(10_000, positive - negative))

    if vector[8] > 0 and raw_score < 2500:
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
    text = "This is a test text with some evidence and planning."
    reference_text = "This is a reference text with some evidence and planning."
    hybrid_val, details = hybrid_score(text, reference_text)
    print(f"Hybrid score: {hybrid_val}")
    print(f"Details: {details}")