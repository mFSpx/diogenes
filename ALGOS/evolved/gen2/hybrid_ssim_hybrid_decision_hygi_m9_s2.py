# DARWIN HAMMER — match 9, survivor 2
# gen: 2
# parent_a: ssim.py (gen0)
# parent_b: hybrid_decision_hygiene_shannon_entropy_m12_s5.py (gen1)
# born: 2026-05-29T23:22:30Z

"""Hybrid similarity and decision‑hygiene module.

Parent A (ssim.py) provides a statistical similarity measure based on
means, variances and covariance of two equal‑length sequences.
Parent B (hybrid_decision_hygiene_shannon_entropy_m12_s5.py) extracts a
9‑dimensional feature vector from text, scores it with linear
positive/negative weights, and computes a weight‑scaled Shannon entropy.

Both algorithms share the same statistical backbone: they rely on
first‑order (mean) and second‑order (variance / covariance) moments.
The fusion therefore treats the feature vectors of two texts as two
signals and evaluates their structural similarity with the SSIM
formula.  The resulting similarity index is then combined with the
individual hygiene scores and entropy values to obtain a single hybrid
metric that reflects both content similarity and decision‑hygiene
quality.

The module implements:
* ``ssim_vector`` – SSIM on numeric vectors.
* ``hygiene_and_entropy`` – hygiene score, raw entropy and normalised entropy.
* ``hybrid_text_similarity`` – full fusion of the two parents for a pair of
  texts, returning a detailed report.
"""

from __future__ import annotations

import math
import re
from typing import Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – regexes and raw count extraction
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
    """Return a 9‑element numpy array of raw counts ordered as ``_FEATURE_ORDER``."""
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
    """Linear decision‑hygiene score and textual label."""
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


def _weighted_entropy(vector: np.ndarray, weights: np.ndarray = _TOTAL_ABS_WEIGHTS) -> float:
    """Shannon entropy on a weight‑scaled version of ``vector`` (bits)."""
    if vector.sum() == 0:
        return 0.0

    weighted = vector.astype(np.float64) * weights.astype(np.float64)
    nonzero = weighted[weighted > 0]
    if nonzero.size == 0:
        return 0.0

    probs = nonzero / nonzero.sum()
    return -float(np.sum(probs * np.log2(probs)))


def _normalised_entropy(entropy: float, k: int) -> float:
    """Normalise *entropy* by the theoretical maximum ``log2(k)``."""
    if k <= 1:
        return 0.0
    return entropy / math.log2(k)


# ----------------------------------------------------------------------
# Parent A – SSIM for generic numeric sequences
# ----------------------------------------------------------------------
def ssim_vector(
    x: np.ndarray,
    y: np.ndarray,
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two equal‑length numeric vectors."""
    if x.shape != y.shape:
        raise ValueError("vectors must have the same shape")
    if x.size == 0:
        raise ValueError("vectors must not be empty")

    n = x.size
    mx = float(x.mean())
    my = float(y.mean())
    vx = float(((x - mx) ** 2).mean())
    vy = float(((y - my) ** 2).mean())
    cov = float(((x - mx) * (y - my)).mean())

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hygiene_and_entropy(text: str) -> dict[str, Any]:
    """Compute hygiene score, raw entropy and normalised entropy for *text*."""
    vec = feature_vector(text)
    score, label = hygiene_score(vec)
    raw_ent = _weighted_entropy(vec)
    norm_ent = _normalised_entropy(raw_ent, k=vec.size)
    return {
        "vector": vec,
        "hygiene_score": score,
        "hygiene_label": label,
        "raw_entropy": raw_ent,
        "norm_entropy": norm_ent,
    }


def hybrid_score(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    hy_a: dict[str, Any],
    hy_b: dict[str, Any],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Fuse SSIM similarity with hygiene & entropy information.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        Feature vectors for the two texts.
    hy_a, hy_b : dict
        Outputs of ``hygiene_and_entropy`` for each text.
    alpha : float
        Weight of the SSIM component (0‑1).
    beta : float
        Weight of the average normalised entropy component (0‑1).

    Returns
    -------
    float
        Hybrid metric in the range roughly [-1, 1] (higher = better).
    """
    # Structural similarity (range ~[-1,1] but usually [0,1])
    sim = ssim_vector(vec_a.astype(np.float64), vec_b.astype(np.float64))

    # Average hygiene score, scaled to [-1,1] by dividing by max absolute value (10_000)
    avg_hyg = (hy_a["hygiene_score"] + hy_b["hygiene_score"]) / 2.0 / 10_000.0

    # Average normalised entropy (0‑1)
    avg_ent = (hy_a["norm_entropy"] + hy_b["norm_entropy"]) / 2.0

    # Combine: similarity is primary, hygiene nudges towards positive,
    # entropy penalises high uncertainty (we subtract it).
    combined = alpha * sim + (1 - alpha) * avg_hyg - beta * avg_ent
    return combined


def hybrid_text_similarity(
    text_a: str,
    text_b: str,
    *,
    alpha: float = 0.6,
    beta: float = 0.3,
) -> dict[str, Any]:
    """
    Full hybrid analysis for two pieces of text.

    Returns a dictionary containing:
        * feature vectors
        * individual hygiene reports
        * SSIM similarity
        * final hybrid score
    """
    vec_a = feature_vector(text_a)
    vec_b = feature_vector(text_b)

    hy_a = hygiene_and_entropy(text_a)
    hy_b = hygiene_and_entropy(text_b)

    similarity = ssim_vector(vec_a.astype(np.float64), vec_b.astype(np.float64))
    final_score = hybrid_score(vec_a, vec_b, hy_a, hy_b, alpha=alpha, beta=beta)

    return {
        "vector_a": vec_a,
        "vector_b": vec_b,
        "hygiene_a": hy_a,
        "hygiene_b": hy_b,
        "ssim_similarity": similarity,
        "hybrid_score": final_score,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = (
        "I have gathered evidence and verified the source. "
        "The plan is ready, and we will schedule the rollout tomorrow. "
        "Please keep the boundary clear and stay safe."
    )
    txt2 = (
        "Evidence was confirmed and the documentation is complete. "
        "We have a checklist and timeline, but we might need to pause "
        "until the risk assessment is done. Support from the team is available."
    )

    report = hybrid_text_similarity(txt1, txt2)
    print("SSIM similarity :", report["ssim_similarity"])
    print("Hybrid score    :", report["hybrid_score"])
    print("Hygiene A       :", report["hygiene_a"]["hygiene_label"], report["hygiene_a"]["hygiene_score"])
    print("Hygiene B       :", report["hygiene_b"]["hygiene_label"], report["hygiene_b"]["hygiene_score"])
    print("Entropy A (norm):", report["hygiene_a"]["norm_entropy"])
    print("Entropy B (norm):", report["hygiene_b"]["norm_entropy"])