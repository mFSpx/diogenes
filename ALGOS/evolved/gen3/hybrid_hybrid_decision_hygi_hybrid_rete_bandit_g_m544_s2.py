# DARWIN HAMMER — match 544, survivor 2
# gen: 3
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s3.py (gen1)
# parent_b: hybrid_rete_bandit_gate_hybrid_workshare_all_m43_s1.py (gen2)
# born: 2026-05-29T23:29:35Z

"""Hybrid Decision‑Hygiene & Work‑Share Allocator

Parent A: `hybrid_decision_hygiene_shannon_entropy_m12_s3.py` – extracts
semantic feature counts from text via regexes and combines them with
positive/negative weight vectors using a dot‑product (matrix) operation.

Parent B: `hybrid_rete_bandit_gate_hybrid_workshare_all_m43_s1.py` – allocates
a total work‑unit budget between deterministic (rule‑based) and LLM‑based
lanes, using a simple bandit‑style adjustment that depends on the day of
the week (computed by the Doomsday calendar).

Mathematical bridge:
    1. The weighted score `S` from Parent A (scalar = w⁺·c⁺ − w⁻·c⁻) is fed
       into the deterministic‑target percentage `p_det` of Parent B.
    2. `p_det` is further modulated by a day‑of‑week factor `d ∈ [0,1]`
       produced by the Doomsday function, giving a final deterministic
       target `p = clip(p_base + α·S + β·d, 0, 100)`.
    3. The resulting `p` drives the work‑share allocation; the LLM lanes
       are distributed proportionally to the groups defined in Parent B.

The code below implements this fusion with three public functions:
`extract_features`, `compute_deterministic_pct`, and `hybrid_allocate`.
"""

from __future__ import annotations

import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction and weighted scoring
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

_REGEX_MAP = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def extract_features(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Count occurrences of each feature in ``text`` and return two NumPy
    vectors: positive counts and negative counts (impulsive, scarcity, risk).

    The vectors are ordered according to ``_FEATURE_ORDER``.
    """
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for idx, name in enumerate(_FEATURE_ORDER):
        regex = __REGEX_MAP[name]
        counts[idx] = len(regex.findall(text))
    # Positive features are the first six entries; negative are the last three.
    pos_counts = counts[:6]
    neg_counts = counts[6:]
    return pos_counts, neg_counts


def weighted_score(pos_counts: np.ndarray, neg_counts: np.ndarray) -> float:
    """
    Compute the scalar score S = w⁺·pos_counts − w⁻·neg_counts.
    """
    pos_score = np.dot(_POSITIVE_WEIGHTS[: len(pos_counts)], pos_counts)
    neg_score = np.dot(_NEGATIVE_WEIGHTS[: len(neg_counts)], neg_counts)
    return float(pos_score - neg_score)


# ----------------------------------------------------------------------
# Parent B – Doomsday calendar + work‑share allocation (bandit‑style)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round to six decimal places for consistency."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return a day‑of‑week index where Monday=0 … Sunday=6.
    The original code used a modulo trick; we keep the same semantics.
    """
    return (date(year, month, day).weekday() + 1) % 7


def _day_factor(dow: int) -> float:
    """Normalised factor ∈[0,1] derived from day‑of‑week."""
    return dow / 6.0  # 0 for Monday, 1 for Sunday


def allocate_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Core allocation routine from Parent B.
    Returns a dict containing deterministic and LLM lane allocations.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)

    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]

    return {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }


# ----------------------------------------------------------------------
# Hybrid bridge – from text score to allocation percentage
# ----------------------------------------------------------------------
def compute_deterministic_pct(
    score: float,
    base_pct: float = 80.0,
    alpha: float = 0.001,
    beta: float = 10.0,
    dow: int = 0,
) -> float:
    """
    Map the weighted text score to a deterministic target percentage.

    p = clip( base_pct + α·score + β·day_factor , 0 , 100 )

    * ``alpha`` scales the influence of the textual score.
    * ``beta`` scales the influence of the day‑of‑week (bandit‑style) factor.
    * ``dow`` is the day‑of‑week index from ``doomsday``.
    """
    day_factor = _day_factor(dow)
    raw = base_pct + alpha * score + beta * day_factor
    return max(0.0, min(100.0, raw))


def hybrid_allocate(
    text: str,
    total_units: float,
    date_tuple: Tuple[int, int, int] = (2026, 5, 29),
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid operation:

    1. Extract feature counts from ``text`` (Parent A).
    2. Compute a weighted score.
    3. Derive the deterministic percentage using the day‑of‑week (Parent B).
    4. Allocate workshare accordingly.

    Returns the allocation dictionary produced by ``allocate_workshare``.
    """
    # Step 1 – feature extraction
    pos_counts, neg_counts = extract_features(text)

    # Step 2 – weighted score
    score = weighted_score(pos_counts, neg_counts)

    # Step 3 – deterministic pct
    dow = doomsday(*date_tuple)
    det_pct = compute_deterministic_pct(score, dow=dow)

    # Step 4 – allocation
    allocation = allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=det_pct,
        groups=groups,
    )
    # Enrich the allocation with diagnostic info
    allocation["diagnostics"] = {
        "pos_counts": pos_counts.tolist(),
        "neg_counts": neg_counts.tolist(),
        "score": _pct(score),
        "day_of_week": dow,
        "deterministic_target_pct_computed": _pct(det_pct),
    }
    return allocation


# ----------------------------------------------------------------------
# Additional helper demonstrating a pure bandit‑style selector
# ----------------------------------------------------------------------
def bandit_select_allocation(
    total_units: float,
    date_tuple: Tuple[int, int, int],
    score: float,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    """
    Very simple epsilon‑greedy selector that occasionally (probability ``epsilon``)
    picks a random deterministic percentage, otherwise uses the deterministic
    percentage derived from the score and day‑of‑week.
    """
    if random.random() < epsilon:
        # Random exploration in the interval [50, 100]
        det_pct = random.uniform(50.0, 100.0)
    else:
        dow = doomsday(*date_tuple)
        det_pct = compute_deterministic_pct(score, dow=dow)

    return allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=det_pct,
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_TEXT = (
        "We have verified the source and have a clear plan. "
        "However, we need to wait for the budget and there is a risk of delay. "
        "Support from the team is essential, and we must set boundaries."
    )
    TOTAL_UNITS = 1000.0
    ALLOC = hybrid_allocate(SAMPLE_TEXT, TOTAL_UNITS, date_tuple=(2026, 5, 29))
    print("Hybrid Allocation Result:")
    for key, value in ALLOC.items():
        if key != "lanes":
            print(f"{key}: {value}")
        else:
            print("lanes:")
            for lane in value:
                print(f"  {lane}")

    # Demonstrate bandit selector
    pos, neg = extract_features(SAMPLE_TEXT)
    sc = weighted_score(pos, neg)
    B_ALLOC = bandit_select_allocation(TOTAL_UNITS, (2026, 5, 29), sc)
    print("\nBandit Allocation Deterministic %:", B_ALLOC["deterministic_target_pct"])