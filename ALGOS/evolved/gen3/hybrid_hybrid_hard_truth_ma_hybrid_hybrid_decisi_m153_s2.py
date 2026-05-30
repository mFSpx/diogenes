# DARWIN HAMMER — match 153, survivor 2
# gen: 3
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# born: 2026-05-29T23:27:18Z

"""Hybrid algorithm merging linguistic function similarity (Parent A) with regex‑based feature weighting (Parent B).

Mathematical bridge:
Both parents transform raw text into a numeric vector representation.
- Parent A: a probability‑like vector **f** ∈ ℝ^{|FUNCTION_CATS|} via `lsm_vector`.
- Parent B: a count vector **c** ∈ ℝ^{|_FEATURE_ORDER|} via regex matches, then weighted by **w** = POSITIVE_WEIGHTS − NEGATIVE_WEIGHTS.

The fusion constructs a block‑concatenated vector **h** = [**f**, **c**·**w**] and evaluates similarity between two texts by the inner product of their normalized **h** vectors. This yields a single scalar that respects both linguistic function overlap and domain‑specific feature agreement.

The module provides three public functions:
* `lsm_vector(text)` – inherits Parent A.
* `feature_vector(text)` – inherits Parent B (regex counts → weighted vector).
* `combined_similarity(text_a, text_b, alpha=0.5)` – computes the hybrid similarity,
  blending the LSM similarity and the weighted feature similarity via a convex
  combination controlled by `alpha`.

"""

import re
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Linguistic function categories
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def words(text: str) -> List[str]:
    """Tokenise lower‑case alphabetic words (including simple contractions)."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> Dict[str, float]:
    """Return the proportion of each FUNCTION_CATS category in *text*."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }


def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Compute per‑category similarity and overall average.
    Score per category = 1 - |a-b|/(a+b+ε) ∈ [0,1].
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail


# ----------------------------------------------------------------------
# Parent B – Regex feature extraction and weighted scoring
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
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorch(ed)?|burn it|quit|give up)\b",
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
_WEIGHT_VECTOR = _POSITIVE_WEIGHTS.astype(float) - _NEGATIVE_WEIGHTS.astype(float)


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


def _raw_counts(text: str) -> np.ndarray:
    """Return raw match counts for each feature in _FEATURE_ORDER."""
    counts = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for idx, feat in enumerate(_FEATURE_ORDER):
        regex = _REGEX_MAP[feat]
        counts[idx] = len(regex.findall(text))
    return counts


def feature_vector(text: str) -> np.ndarray:
    """
    Transform *text* into a weighted feature vector:
        v = counts(text) ⊙ _WEIGHT_VECTOR
    where ⊙ denotes element‑wise multiplication.
    """
    raw = _raw_counts(text)
    weighted = raw.astype(float) * _WEIGHT_VECTOR
    return weighted


def _cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Safe cosine similarity (returns 0 when either norm is zero)."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def combined_similarity(
    text_a: str,
    text_b: str,
    alpha: float = 0.5,
) -> float:
    """
    Hybrid similarity between *text_a* and *text_b*.

    1. Compute LSM similarity `s_lsm` via `lsm_score`.
    2. Compute weighted‑feature cosine similarity `s_feat` via `feature_vector`.
    3. Return convex combination:  alpha * s_lsm + (1‑alpha) * s_feat.

    Parameters
    ----------
    alpha : float in [0,1]
        Relative importance of linguistic similarity versus domain‑specific features.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")

    vec_a = lsm_vector(text_a)
    vec_b = lsm_vector(text_b)
    s_lsm, _ = lsm_score(vec_a, vec_b)

    feat_a = feature_vector(text_a)
    feat_b = feature_vector(text_b)
    s_feat = _cosine_similarity(feat_a, feat_b)

    # Normalise feature similarity to [0,1] (cosine already lies in [-1,1] but our vectors are non‑negative)
    s_feat = (s_feat + 1.0) / 2.0  # map [-1,1] → [0,1]

    return alpha * s_lsm + (1.0 - alpha) * s_feat


def similarity_matrix(texts: List[str], alpha: float = 0.5) -> np.ndarray:
    """
    Build an N×N symmetric matrix of hybrid similarities for a list of texts.
    """
    n = len(texts)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            sim = combined_similarity(texts[i], texts[j], alpha=alpha)
            mat[i, j] = sim
            mat[j, i] = sim
    return mat


def rank_by_similarity(reference: str, candidates: List[str], alpha: float = 0.5) -> List[Tuple[str, float]]:
    """
    Rank *candidates* by descending hybrid similarity to *reference*.
    Returns a list of (candidate, score) tuples.
    """
    scored = [(c, combined_similarity(reference, c, alpha=alpha)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


if __name__ == "__main__":
    # Simple smoke test
    txt1 = "I verified the document and sent the proof to my friend."
    txt2 = "We plan to check the source tomorrow and maybe wait."
    txt3 = "I feel unsafe, need support, and want to protect my privacy."
    texts = [txt1, txt2, txt3]

    print("Hybrid similarity matrix:")
    print(similarity_matrix(texts, alpha=0.6))

    print("\nRanking txt1 against others:")
    for cand, score in rank_by_similarity(txt1, texts[1:], alpha=0.6):
        print(f"{cand[:30]:30} -> {score:.4f}")

    # Ensure no exception on empty input
    assert similarity_matrix([], alpha=0.5).size == 0
    print("\nSmoke test passed.")