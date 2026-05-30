# DARWIN HAMMER — match 12, survivor 3
# gen: 1
# parent_a: decision_hygiene.py (gen0)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:18:35Z

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regexes and raw count extraction
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Parent B – Shannon entropy implementation 
# ----------------------------------------------------------------------


def shannon_entropy(observations: Iterable[float | Any], is_distribution: bool = False) -> float:
    """Return Shannon entropy (bits) of a discrete distribution.

    If *is_distribution* is False, *observations* are treated as raw samples;
    otherwise they are interpreted as probabilities that already sum to 1.
    """
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


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------


def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using parent‑A regexes."""
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
    """Return a 9‑element ``numpy`` array of counts ordered as _FEATURE_ORDER."""
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


def hygiene_score(vector: np.ndarray) -> Tuple[int, str]:
    """Compute the original decision‑hygiene score and a textual label.

    The calculation mirrors ``score_features`` from parent A.
    """
    positive = int(np.dot(vector, _POSITIVE_WEIGHTS))
    negative = int(np.dot(vector, _NEGATIVE_WEIGHTS))
    raw_score = max(-10000, min(10000, positive - negative))

    risk_present = vector[8] > 0  
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
    """Treat *vector* as raw observations and return its Shannon entropy (bits).

    Zero entries are ignored in the probability normalisation.
    """
    if vector.sum() == 0:
        return 0.0
    non_zero_vector = vector[vector > 0]
    if non_zero_vector.sum() == 0:
        return 0.0
    probs = non_zero_vector / float(non_zero_vector.sum())
    return -float(np.sum(probs * np.log2(probs)))


def hybrid_score(text: str) -> Tuple[float, dict[str, Any]]:
    """Combine hygiene score with entropy to produce a hybrid metric.

    Returns ``(hybrid_score, details)`` where *details* contains the raw
    hygiene score, entropy, normalised entropy and the original label.
    """
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)
    ent = entropy_from_counts(vec)

    h_max = math.log2(len(_FEATURE_ORDER))
    norm_ent = ent / h_max if h_max > 0 else 0.0

    hybrid = raw_score * (1.0 + norm_ent)

    details = {
        "raw_score": raw_score,
        "label": label,
        "entropy_bits": round(ent, 4),
        "norm_entropy": round(norm_ent, 4),
        "hybrid_score": round(hybrid, 2),
    }
    return hybrid, details


def monthly_hybrid(rows: List[dict[str, Any]]) -> List[dict[str, Any]]:
    monthly_results = {}
    for row in rows:
        date = row['occurred_at'][:7]  # Get year and month
        if date not in monthly_results:
            monthly_results[date] = []
        hybrid, details = hybrid_score(row['text'])
        monthly_results[date].append(details)

    aggregated_results = []
    for date, results in monthly_results.items():
        avg_hybrid_score = sum([r['hybrid_score'] for r in results]) / len(results)
        avg_raw_score = sum([r['raw_score'] for r in results]) / len(results)
        avg_entropy_bits = sum([r['entropy_bits'] for r in results]) / len(results)
        avg_norm_entropy = sum([r['norm_entropy'] for r in results]) / len(results)

        aggregated_results.append({
            'date': date,
            'avg_hybrid_score': round(avg_hybrid_score, 2),
            'avg_raw_score': round(avg_raw_score, 2),
            'avg_entropy_bits': round(avg_entropy_bits, 4),
            'avg_norm_entropy': round(avg_norm_entropy, 4),
        })

    return aggregated_results