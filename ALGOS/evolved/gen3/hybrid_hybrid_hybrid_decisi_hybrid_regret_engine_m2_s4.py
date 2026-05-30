# DARWIN HAMMER — match 2, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s3.py (gen2)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s0.py (gen2)
# born: 2026-05-29T23:26:17Z

"""Hybrid Decision-Regret Engine
================================

This module fuses the *text‑feature decision* logic of
``hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s3.py`` with the
*regret‑weighted strategy* and *Gini‑based disparity* calculations of
``hybrid_regret_engine_hybrid_doomsday_cale_m19_s0.py``.

Mathematical Bridge
-------------------
* Parent A extracts a vector **c** of regex‑based feature counts and
  applies separate positive **p** and negative **n** weight vectors,
  yielding a raw utility vector **u = p·c – n·c**.
* Parent B treats a set of actions with utilities **uᵢ** and computes a
  regret‑weighted soft‑max distribution  
  **πᵢ = exp(uᵢ – max(u)) / Σⱼ exp(uⱼ – max(u))**.
* The *Gini coefficient* of the resulting probability distribution quantifies
  the regret (inequality) among features.

The hybrid algorithm therefore:
1. Builds **u** from text‑feature counts (A).
2. Generates a regret‑weighted probability vector **π** (B).
3. Computes the Gini coefficient of **π** and of a weekday‑distribution
   (B) and combines them to a final decision score.

The three core functions below demonstrate this pipeline.
"""

from __future__ import annotations

import re
import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction & weighting
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


def _raw_counts(text: str) -> Dict[str, int]:
    """Count occurrences of each feature regex in *text*."""
    return {
        "evidence": len(EVIDENCE_RE.findall(text or "")),
        "planning": len(PLANNING_RE.findall(text or "")),
        "delay": len(DELAY_RE.findall(text or "")),
        "support": len(SUPPORT_RE.findall(text or "")),
        "boundary": len(BOUNDARY_RE.findall(text or "")),
        "outcome": len(OUTCOME_RE.findall(text or "")),
        "impulsive": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity": len(SCARCITY_RE.findall(text or "")),
        "risk": len(RISK_RE.findall(text or "")),
    }


def _utility_vector(counts: Dict[str, int]) -> np.ndarray:
    """Convert raw counts into a signed utility vector using A's weight scheme."""
    c = np.array([counts[f] for f in _FEATURE_ORDER], dtype=np.float64)
    pos = _POSITIVE_WEIGHTS.astype(np.float64)
    neg = _NEGATIVE_WEIGHTS.astype(np.float64)
    return pos * c - neg * c


# ----------------------------------------------------------------------
# Parent B – regret‑weighted softmax and Gini calculations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


def regret_weighted_softmax(values: np.ndarray) -> np.ndarray:
    """Regret‑weighted softmax as described in Parent B."""
    if values.size == 0:
        return np.array([])
    best = np.max(values)
    shifted = np.exp(values - best)  # numerical stability
    total = shifted.sum()
    return shifted / (total if total != 0 else 1.0)


def gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a non‑negative list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


def weekday_distribution(year: int, month: int, num_days: int) -> List[int]:
    """Return a list of weekday indices (0=Mon … 6=Sun) for the given month."""
    return [(dt.date(year, month, day).weekday()) for day in range(1, num_days + 1)]


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_actions_from_counts(counts: Dict[str, int]) -> List[Action]:
    """
    Translate regex counts into ``Action`` objects.
    The expected value of each action is the signed utility from Parent A.
    """
    utilities = _utility_vector(counts)
    actions = [
        Action(id=feat, expected_value=val)
        for feat, val in zip(_FEATURE_ORDER, utilities)
    ]
    return actions


def hybrid_regret_distribution(text: str) -> np.ndarray:
    """
    Core hybrid operation:
    1. Extract feature counts from *text* (Parent A).
    2. Build ``Action`` objects (bridge).
    3. Apply regret‑weighted softmax to the expected values (Parent B).
    Returns a probability distribution over the feature set.
    """
    counts = _raw_counts(text)
    actions = build_actions_from_counts(counts)
    values = np.array([a.expected_value for a in actions], dtype=np.float64)
    return regret_weighted_softmax(values)


def hybrid_decision_score(
    text: str,
    year: int,
    month: int,
    num_days: int,
    alpha: float = 0.6,
) -> float:
    """
    Produce a single decision score that blends:
    * the Gini of the regret‑weighted feature distribution,
    * the Gini of the weekday distribution for the supplied calendar window.

    ``alpha`` controls the weight of the feature‑Gini (the rest goes to the
    weekday‑Gini).  The result lies in ``[0, 1]`` where values near 1 indicate
    a highly unequal (high‑regret) feature profile combined with an uneven
    weekday spread.
    """
    # Feature side
    prob_dist = hybrid_regret_distribution(text)
    feature_gini = gini_coefficient(prob_dist.tolist())

    # Temporal side
    weekdays = weekday_distribution(year, month, num_days)
    # Count occurrences of each weekday (0‑6)
    weekday_counts = [weekdays.count(d) for d in range(7)]
    weekday_gini = gini_coefficient(weekday_counts)

    # Blend the two Gini measures
    return alpha * feature_gini + (1 - alpha) * weekday_gini


def prune_features_by_threshold(text: str, threshold: float = 0.05) -> List[str]:
    """
    Return the list of feature names whose regret‑weighted probability
    exceeds *threshold*.  This mimics the “decreasing pruning” idea from
    Parent A while using the softmax probabilities from Parent B.
    """
    prob_dist = hybrid_regret_distribution(text)
    selected = [
        feat
        for feat, prob in zip(_FEATURE_ORDER, prob_dist.tolist())
        if prob >= threshold
    ]
    return selected


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have evidence that the plan was executed, but there is a delay. "
        "Please verify the document and check the outcome. "
        "If risk arises, call a friend."
    )
    # Example date: March 2024, 31 days
    year, month, days = 2024, 3, 31

    print("Raw feature counts:", _raw_counts(sample_text))
    print("Regret‑weighted distribution:", hybrid_regret_distribution(sample_text))
    print("Hybrid decision score:", hybrid_decision_score(sample_text, year, month, days))
    print("Pruned features (threshold=0.1):", prune_features_by_threshold(sample_text, 0.1))
    sys.exit(0)