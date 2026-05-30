# DARWIN HAMMER — match 12, survivor 2
# gen: 1
# parent_a: decision_hygiene.py (gen0)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:18:35Z

"""Hybrid Decision Hygiene & Shannon Entropy Module.

This module fuses the *Decision Hygiene* algorithm (parent A) with the
*Shannon Entropy* calculation (parent B).  The mathematical bridge is
the **feature‑count vector** produced by the hygiene regexes.  The vector
is used in two ways:

1.  A weighted dot‑product yields the original hygiene *score*.
2.  The same vector, after normalisation to a probability distribution,
    feeds the Shannon entropy formula, giving a measure of *information
    diversity* across the decision‑making cues.

The final hybrid score multiplies the hygiene score by a factor that
depends on the normalized entropy (0 ≤ H/Hmax ≤ 1), thus rewarding
decisions that are both well‑scored and information‑rich.

Only the allowed standard‑library modules and ``numpy`` are used.
"""

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
# Parent B – Shannon entropy implementation (kept for reference)
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

    # label logic (unchanged)
    risk_present = vector[8] > 0  # risk count
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
    probs = vector[vector > 0] / float(vector.sum())
    return -float(np.sum(probs * np.log2(probs)))


def hybrid_score(text: str) -> Tuple[float, dict[str, Any]]:
    """Combine hygiene score with entropy to produce a hybrid metric.

    Returns ``(hybrid_score, details)`` where *details* contains the raw
    hygiene score, entropy, normalised entropy and the original label.
    """
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec)
    ent = entropy_from_counts(vec)

    # Normalised entropy: H / H_max where H_max = log2(k) with k = number of
    # possible features (9).  This yields a factor in [0,1].
    h_max = math.log2(len(_FEATURE_ORDER))
    norm_ent = ent / h_max if h_max > 0 else 0.0

    # Hybrid score: boost the raw hygiene score proportionally to the
    # information richness.  The factor (1 + norm_ent) keeps the sign of the
    # original score while rewarding diverse cue usage.
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
    """Aggregate hybrid scores per month.

    Each *row* must contain ``occurred_at`` (ISO‑like date string) and ``text``.
    The function returns a list of dictionaries with month, count, average
    raw score, average hybrid score and average entropy.
    """
    buckets: dict[str, List[Tuple[int, float, float]]] = {}
    for r in rows:
        ts = r.get("occurred_at")
        month_key = str(ts)[:7] if ts else "unknown"
        text = r.get("text", "")
        vec = feature_vector(text)
        raw, _ = hygiene_score(vec)
        ent = entropy_from_counts(vec)
        hybrid, _ = hybrid_score(text)
        buckets.setdefault(month_key, []).append((raw, hybrid, ent))

    result = []
    for month, values in sorted(buckets.items()):
        raw_vals = [v[0] for v in values]
        hybrid_vals = [v[1] for v in values]
        ent_vals = [v[2] for v in values]
        result.append(
            {
                "month": month,
                "n": len(values),
                "avg_raw_score": round(sum(raw_vals) / len(raw_vals), 2),
                "avg_hybrid_score": round(sum(hybrid_vals) / len(hybrid_vals), 2),
                "avg_entropy": round(sum(ent_vals) / len(ent_vals), 4),
            }
        )
    return result


def compare_halves_hybrid(rows: List[dict[str, Any]]) -> dict[str, Any]:
    """Compare early vs. late halves of dated rows using the hybrid metric.

    Mirrors ``compare_halves`` from parent A but works on the hybrid scores.
    """
    dated = [r for r in rows if r.get("occurred_at") is not None]
    dated.sort(key=lambda r: r["occurred_at"])
    if len(dated) < 2:
        return {"available": False, "reason": "need at least two dated decision signals"}

    half = len(dated) // 2
    early = dated[:half]
    late = dated[half:]

    def avg_hybrid(subset: List[dict[str, Any]]) -> float:
        scores = [hybrid_score(r.get("text", ""))[0] for r in subset]
        return sum(scores) / len(scores) if scores else 0.0

    early_avg = avg_hybrid(early)
    late_avg = avg_hybrid(late)
    delta = late_avg - early_avg

    interpretation = (
        "improved_hybrid_signal"
        if delta > 500
        else "roughly_flat"
        if abs(delta) <= 500
        else "degraded_hybrid_signal"
    )
    return {
        "available": True,
        "early_n": len(early),
        "late_n": len(late),
        "early_avg_hybrid": round(early_avg, 2),
        "late_avg_hybrid": round(late_avg, 2),
        "delta": round(delta, 2),
        "interpretation": interpretation,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I have evidence and a solid plan, but I feel a bit of panic.",
        "Need to verify sources, schedule the rollout, and ask a lawyer for support.",
        "I'm broke, can't afford to wait, and I'm scared I might hurt myself.",
    ]

    print("=== Hybrid Scores ===")
    for i, txt in enumerate(sample_texts, 1):
        hscore, details = hybrid_score(txt)
        print(f"[{i}] Hybrid score: {hscore:.2f}")
        for k, v in details.items():
            print(f"    {k}: {v}")

    # Build a tiny dataset for monthly aggregation
    rows = [
        {"occurred_at": "2023-01-15", "text": sample_texts[0]},
        {"occurred_at": "2023-01-20", "text": sample_texts[1]},
        {"occurred_at": "2023-02-05", "text": sample_texts[2]},
        {"occurred_at": "2023-02-18", "text": sample_texts[0]},
    ]

    print("\n=== Monthly Aggregation ===")
    for month_info in monthly_hybrid(rows):
        print(month_info)

    print("\n=== Half‑Comparison ===")
    print(compare_halves_hybrid(rows))

    sys.exit(0)