# DARWIN HAMMER — match 3337, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:49:20Z

"""Hybrid algorithm merging regex‑based semantic scoring (Parent A) with
hash‑seeded stylometry and linguistic‑category vectors (Parent B).

Mathematical bridge
-------------------
Both parents expose deterministic numeric representations of a text:

* **Parent A** → a 9‑dimensional count vector  *c*  (evidence, planning, …,
  risk) which is linearly weighted by positive/negative coefficient vectors
  *p* and *n*.
* **Parent B** → a 96‑dimensional pseudo‑random stylometry vector  *s*  and an
  8‑dimensional lexical‑category proportion vector  *l*.

The hybrid treats the concatenation  

    v = [c, s, l]   ∈ ℝ^{9+96+8}

as a single feature vector.  A deterministic weight vector  

    w = [p, w_s, w_l]   ∈ ℝ^{9+96+8}

is built by re‑using the original positive weights *p* and by generating
* w_s* and *w_l* from a SHA‑256 seed (guaranteeing reproducibility).  The
final scalar decision is the bilinear form

    score(text) = v · w .

Thus the core topologies of the two parents are fused through a common
linear algebraic interface (concatenation + dot‑product)."""

import re
import math
import random
import hashlib
from pathlib import Path
from typing import List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – regex feature extraction
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


def _regex_counts(text: str) -> np.ndarray:
    """Return a 9‑element array with raw match counts for each feature."""
    counts = []
    for name in _FEATURE_ORDER:
        regex = _REGEX_MAP[name]
        counts.append(len(regex.findall(text)))
    return np.array(counts, dtype=np.float64)


def regex_feature_vector(text: str) -> np.ndarray:
    """
    Produce the weighted regex feature vector used by Parent A:

        v_a =  p * c  –  n * c

    where *c* is the raw count vector, *p* the positive weight vector and *n*
    the negative weight vector.
    """
    c = _regex_counts(text)
    weighted = _POSITIVE_WEIGHTS * c - _NEGATIVE_WEIGHTS * c
    return weighted  # shape (9,)


# ----------------------------------------------------------------------
# Parent B – stylometry and linguistic‑category vectors
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_word(word: str) -> str:
    return word.strip(PUNCT).lower()


def words(text: str) -> List[str]:
    """Return a list of alphabetic tokens (lower‑cased, punctuation stripped)."""
    return [_clean_word(w) for w in text.split() if _clean_word(w).isalpha()]


def stylometry_vector(text: str) -> np.ndarray:
    """
    Deterministic 96‑dimensional pseudo‑random vector seeded by SHA‑256(text).
    Mirrors Parent B's `stylometry_features`.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96, dtype=np.float64)


def lsm_vector(text: str) -> np.ndarray:
    """
    Lexical‑category proportion vector.
    For each FUNCTION_CAT compute the fraction of tokens belonging to that
    category.  Length equals the number of categories (8).
    """
    toks = words(text)
    total = len(toks) if toks else 1  # avoid division by zero
    proportions = []
    for cat in FUNCTION_CATS.values():
        count = sum(1 for t in toks if t in cat)
        proportions.append(count / total)
    return np.array(proportions, dtype=np.float64)  # shape (8,)


# ----------------------------------------------------------------------
# Hybrid layer – linear combination of all modalities
# ----------------------------------------------------------------------
# Deterministic weight generation for the stylometry and LSM parts.
# The seed is derived from a constant string so that the weights are the same
# across runs but independent from the input text.
_WEIGHT_SEED = int.from_bytes(hashlib.sha256(b"hybrid_weight_seed").digest()[:8], "big")
_rng = np.random.default_rng(_WEIGHT_SEED)
_STYLOMETRY_WEIGHTS = _rng.normal(loc=0.0, scale=1.0, size=96).astype(np.float64)
_LSM_WEIGHTS = _rng.normal(loc=0.0, scale=1.0, size=8).astype(np.float64)


def hybrid_weight_vector() -> np.ndarray:
    """Concatenate the three sub‑weight vectors into a single (113,) vector."""
    return np.concatenate([_POSITIVE_WEIGHTS, _STYLOMETRY_WEIGHTS, _LSM_WEIGHTS])


def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Build the unified feature vector:

        v = [ regex_feature_vector , stylometry_vector , lsm_vector ]

    Shape: (113,)
    """
    a = regex_feature_vector(text)          # (9,)
    s = stylometry_vector(text)             # (96,)
    l = lsm_vector(text)                    # (8,)
    return np.concatenate([a, s, l])


def hybrid_score(text: str) -> float:
    """
    Compute the final scalar decision:

        score = v · w

    where *v* is the hybrid feature vector and *w* the hybrid weight vector.
    Positive scores indicate a predominance of “safe” cues (from Parent A)
    amplified by stylistic alignment; negative scores highlight risky/impulsive
    signals.
    """
    v = hybrid_feature_vector(text)
    w = hybrid_weight_vector()
    return float(np.dot(v, w))


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_vectors(text: str) -> None:
    """Print each sub‑vector and the combined score for a quick sanity check."""
    print("=== Regex weighted vector (9) ===")
    print(regex_feature_vector(text))
    print("\n=== Stylometry vector (96) – first 10 values ===")
    print(stylometry_vector(text)[:10])
    print("\n=== LSM proportion vector (8) ===")
    print(lsm_vector(text))
    print("\n=== Hybrid score ===")
    print(hybrid_score(text))


if __name__ == "__main__":
    SAMPLE = (
        "I plan to verify the source of the document tomorrow. "
        "If it looks good, I'll ship it tonight. "
        "But I'm scared of the risk and feel a bit impulsive."
    )
    demo_vectors(SAMPLE)