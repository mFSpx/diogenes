# DARWIN HAMMER — match 12, survivor 5
# gen: 1
# parent_a: decision_hygiene.py (gen0)
# parent_b: shannon_entropy.py (gen0)
# born: 2026-05-29T23:18:35Z

from __future__ import annotations

import datetime
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

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

# Positive contributions (desired cues)
_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
# Negative contributions (undesired cues)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Combined absolute weights used for entropy weighting
_TOTAL_ABS_WEIGHTS = np.abs(_POSITIVE_WEIGHTS) + np.abs(_NEGATIVE_WEIGHTS)


def _raw_counts(text: str) -> dict[str, int]:
    """Extract raw feature counts from *text* using the compiled regexes."""
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
    """Return a 9‑element ``numpy`` array of raw counts ordered as ``_FEATURE_ORDER``."""
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


def hygiene_score(
    vector: np.ndarray,
    pos_weights: np.ndarray = _POSITIVE_WEIGHTS,
    neg_weights: np.ndarray = _NEGATIVE_WEIGHTS,
) -> Tuple[int, str]:
    """Compute the original decision‑hygiene score and a textual label.

    The calculation mirrors the parent‑A algorithm but allows custom
    weight vectors for testing or adaptation.
    """
    positive = int(np.dot(vector, pos_weights))
    negative = int(np.dot(vector, neg_weights))
    raw_score = max(-10_000, min(10_000, positive - negative))

    # Label logic – unchanged semantics
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


def _weighted_entropy(
    vector: np.ndarray,
    weights: np.ndarray = _TOTAL_ABS_WEIGHTS,
) -> float:
    """Compute Shannon entropy on a *weight‑scaled* version of ``vector``.

    Zero‑weight entries are ignored. The function returns entropy in bits.
    """
    if vector.sum() == 0:
        return 0.0

    weighted = vector.astype(np.float64) * weights.astype(np.float64)
    nonzero = weighted[weighted > 0]
    if nonzero.size == 0:
        return 0.0

    probs = nonzero / nonzero.sum()
    return -float(np.sum(probs * np.log2(probs)))


def _normalised_entropy(entropy: float, k: int) -> float:
    """Normalise *entropy* by the theoretical maximum ``log2(k)``.

    If ``k`` is 0 or 1 the normalised value is defined as 0.
    """
    if k <= 1:
        return 0.0
    h_max = math.log2(k)
    return entropy / h_max


def hybrid_score(
    text: str,
    *,
    beta: float = 0.5,
    pos_weights: np.ndarray = _POSITIVE_WEIGHTS,
    neg_weights: np.ndarray = _NEGATIVE_WEIGHTS,
    entropy_weights: np.ndarray = _TOTAL_ABS_WEIGHTS,
) -> Tuple[float, dict[str, Any]]:
    """Combine hygiene score with a weighted entropy to produce a deeper hybrid metric.

    Parameters
    ----------
    text:
        Input text to be analysed.
    beta:
        Scaling factor controlling how strongly entropy influences the final score.
        ``beta=0`` yields the pure hygiene score; larger values increase the
        contribution of information diversity.
    pos_weights, neg_weights, entropy_weights:
        Optional custom weight vectors for experimentation.

    Returns
    -------
    (hybrid_score, details) where *details* contains intermediate values.
    """
    vec = feature_vector(text)
    raw_score, label = hygiene_score(vec, pos_weights, neg_weights)

    # Compute weighted entropy
    ent = _weighted_entropy(vec, entropy_weights)

    # Normalise by the number of *active* weighted categories
    active_categories = np.count_nonzero(vec * entropy_weights)
    norm_ent = _normalised_entropy(ent, active_categories)

    # Hybrid combination – additive scaling to preserve sign while rewarding diversity
    hybrid = raw_score * (1.0 + beta * norm_ent)

    details = {
        "raw_score": raw_score,
        "label": label,
        "entropy_bits": round(ent, 4),
        "norm_entropy": round(norm_ent, 4),
        "beta": beta,
        "hybrid_score": round(hybrid, 2),
    }
    return hybrid, details


def _parse_iso_date(date_str: str) -> datetime.date:
    """Parse an ISO‑like date string (YYYY‑MM‑DD or YYYY‑MM‑DDTHH:MM…) to a ``date``."""
    try:
        return datetime.datetime.fromisoformat(date_str).date()
    except ValueError:
        # Fallback: split on non‑digit characters and construct date
        parts = [int(p) for p in re.split(r"\D+", date_str) if p]
        if len(parts) >= 3:
            return datetime.date(parts[0], parts[1], parts[2])
        raise


def monthly_hybrid(
    rows: List[dict[str, Any]],
    *,
    beta: float = 0.5,
    date_key: str = "occurred_at",
    text_key: str = "text",
    aggregation_fn: Callable[[List[float]], float] = lambda scores: sum(scores) / len(scores) if scores else 0.0,
) -> List[dict[str, Any]]:
    """Aggregate hybrid scores per month.

    Parameters
    ----------
    rows:
        Iterable of mappings each containing a date string and a text field.
    beta:
        Passed through to :func:`hybrid_score`.
    date_key, text_key:
        Keys used to retrieve the date and text from each row.
    aggregation_fn:
        Function applied to the list of hybrid scores for a month (default: arithmetic mean).

    Returns
    -------
    List of dictionaries, each with ``year``, ``month`` and ``hybrid_score`` keys.
    """
    monthly: dict[Tuple[int, int], List[float]] = defaultdict(list)

    for row in rows:
        try:
            dt = _parse_iso_date(row[date_key])
            txt = row[text_key]
        except Exception:
            continue  # skip malformed rows

        score, _ = hybrid_score(txt, beta=beta)
        monthly[(dt.year, dt.month)].append(score)

    result: List[dict[str, Any]] = []
    for (year, month), scores in sorted(monthly.items()):
        agg = aggregation_fn(scores)
        result.append(
            {
                "year": year,
                "month": month,
                "hybrid_score": round(agg, 2),
                "sample_size": len(scores),
            }
        )
    return result

# ----------------------------------------------------------------------
# Example usage (commented out – keep module import‑friendly)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     sample_text = "I have evidence and a plan, but I feel impulsive and scarce."
#     print(hybrid_score(sample_text))
#     rows = [
#         {"occurred_at": "2023-07-15", "text": sample_text},
#         {"occurred_at": "2023-07-20", "text": "We verified the source and scheduled the rollout."},
#     ]
#     print(monthly_hybrid(rows))