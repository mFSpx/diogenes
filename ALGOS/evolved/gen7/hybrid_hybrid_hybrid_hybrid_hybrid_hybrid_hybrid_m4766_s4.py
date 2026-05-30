# DARWIN HAMMER — match 4766, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_decisi_m516_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m2238_s3.py (gen6)
# born: 2026-05-29T23:58:05Z

"""Hybrid algorithm combining linguistic style metrics (Parent A) with Clifford geometric algebra
operations (Parent B).

Mathematical bridge:
- Parent A yields a normalized feature vector `v = [v0, v1, …, v8]` derived from regex
  counts in a text (evidence, planning, …, risk).
- Parent B defines a geometric product on multivectors whose basis blades are indexed by
  integers.  By mapping each feature index `i` to the 1‑vector basis `e_i`, the feature
  vector can be interpreted as a multivector `F = Σ vi·e_i`.
- The weighted feature extraction (`_POSITIVE_WEIGHTS` / `_NEGATIVE_WEIGHTS`) supplies
  an additional multivector `W = Σ wi·e_i`.  The geometric product `P = F ⨂ W`
  fuses the two topologies: the scalar part of `P` encodes the alignment between the
  linguistic style and the weighted features, while higher‑grade blades capture
  cross‑feature interactions.

The module provides three high‑level functions:
1. `extract_feature_counts(text)` – regex based counting.
2. `build_multivector_from_counts(counts, weights)` – creates a multivector from
   counts and a weight array.
3. `hybrid_score(text, use_negative=False)` – computes the scalar component of the
   geometric product between the linguistic style multivector and the weighted feature
   multivector (positive or negative weights)."""

import re
import math
import random
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# ----------------------------------------------------------------------
# Parent A – linguistic feature extraction
# ----------------------------------------------------------------------
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

# Compile regexes for each feature
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact)\b", re.I)
_OUTCOME_RE = re.compile(r"\b(?:result|outcome|conclusion|success|failure|impact)\b", re.I)
_IMPULSIVE_RE = re.compile(r"\b(?:impulse|impulsive|rash|spontaneous|hasty)\b", re.I)
_SCARCITY_RE = re.compile(r"\b(?:scarcity|limited|rare|few|shortage)\b", re.I)
_RISK_RE = re.compile(r"\b(?:risk|danger|hazard|threat|peril|exposure)\b", re.I)

_REGEX_MAP = {
    "evidence": _EVIDENCE_RE,
    "planning": _PLANNING_RE,
    "delay": _DELAY_RE,
    "support": _SUPPORT_RE,
    "boundary": _BOUNDARY_RE,
    "outcome": _OUTCOME_RE,
    "impulsive": _IMPULSIVE_RE,
    "scarcity": _SCARCITY_RE,
    "risk": _RISK_RE,
}


def extract_feature_counts(text: str) -> dict:
    """Return a dict mapping each feature name to the number of regex matches in *text*."""
    counts = {}
    for feat, regex in _REGEX_MAP.items():
        matches = regex.findall(text)
        counts[feat] = len(matches)
    return counts


def linguistic_style_vector(text: str) -> np.ndarray:
    """
    Normalized linguistic style vector `v_i = count_i / total_counts`.
    If no features are found, returns a zero vector.
    """
    counts = extract_feature_counts(text)
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(_FEATURE_ORDER), dtype=float)
    vec = np.array([counts[feat] for feat in _FEATURE_ORDER], dtype=float) / total
    return vec


# ----------------------------------------------------------------------
# Parent B – Clifford (geometric) algebra utilities
# ----------------------------------------------------------------------
def _sorted_blade_and_sign(indices: list[int]) -> tuple[tuple[int, ...], int]:
    """
    Sort a list of basis indices and return the sorted tuple together with the sign
    (+1 or -1) resulting from the required swaps.  Duplicate indices cancel out
    (e_i * e_i = 1) and are removed.
    """
    counts = Counter(indices)
    reduced = [i for i, c in counts.items() if c % 2 == 1]

    sign = 1
    lst = list(reduced)
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return tuple(lst), sign


def _multiply_blades(
    blade_a: frozenset[int], blade_b: frozenset[int]
) -> tuple[frozenset[int], int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _sorted_blade_and_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(
    a: dict[frozenset[int], float], b: dict[frozenset[int], float]
) -> dict[frozenset[int], float]:
    """
    Full geometric product of two multivectors `a` and `b`.
    Each multivector is represented as a dict mapping a blade (frozenset of indices)
    to its scalar coefficient.
    """
    result: dict[frozenset[int], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    return result


# ----------------------------------------------------------------------
# Hybrid constructions
# ----------------------------------------------------------------------
def build_multivector_from_counts(
    counts: dict, weight_array: np.ndarray
) -> dict[frozenset[int], float]:
    """
    Create a multivector where each feature index `i` contributes a 1‑vector blade
    `e_i` with coefficient `c_i = weight_i * count_i`.
    """
    mv: dict[frozenset[int], float] = {}
    for i, feat in enumerate(_FEATURE_ORDER):
        coeff = float(weight_array[i] * counts.get(feat, 0))
        if coeff != 0.0:
            mv[frozenset({i})] = coeff
    return mv


def hybrid_multivector(
    text: str, use_negative: bool = False
) -> dict[frozenset[int], float]:
    """
    Compute the geometric product between the linguistic style multivector
    derived from *text* and the weighted feature multivector.
    `use_negative` selects the negative weight set; otherwise positive weights are used.
    """
    counts = extract_feature_counts(text)
    # linguistic style as a multivector (coefficients are the normalized probabilities)
    lsm_vec = linguistic_style_vector(text)
    lsm_mv: dict[frozenset[int], float] = {}
    for i, prob in enumerate(lsm_vec):
        if prob != 0.0:
            lsm_mv[frozenset({i})] = prob

    weights = _NEGATIVE_WEIGHTS if use_negative else _POSITIVE_WEIGHTS
    weight_mv = build_multivector_from_counts(counts, weights)

    return geometric_product(lsm_mv, weight_mv)


def hybrid_score(text: str, use_negative: bool = False) -> float:
    """
    Extract a scalar score from the hybrid multivector:
    - The scalar (grade‑0) component reflects direct alignment.
    - The magnitude of higher‑grade components is added as a penalty term.
    """
    mv = hybrid_multivector(text, use_negative=use_negative)
    scalar = mv.get(frozenset(), 0.0)
    # Sum absolute values of all non‑scalar blades as penalty
    penalty = sum(abs(v) for k, v in mv.items() if k)
    return scalar - 0.1 * penalty  # weighting of penalty is arbitrary but demonstrates fusion


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    The team prepared a detailed plan and checklist before the launch.
    Evidence was logged in the audit trail and verified by the reviewer.
    However, we decided to pause the rollout due to a risk of data loss.
    Support from the legal department was requested to handle the boundary issues.
    """
    print("Feature counts:", extract_feature_counts(sample_text))
    print("Linguistic style vector:", linguistic_style_vector(sample_text))
    print("Hybrid multivector (positive):", hybrid_multivector(sample_text, use_negative=False))
    print("Hybrid score (positive):", hybrid_score(sample_text, use_negative=False))
    print("Hybrid score (negative):", hybrid_score(sample_text, use_negative=True))