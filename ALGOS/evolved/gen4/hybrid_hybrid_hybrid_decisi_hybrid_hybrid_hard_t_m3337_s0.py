# DARWIN HAMMER — match 3337, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:49:20Z

"""
This module represents a novel hybrid algorithm that mathematically fuses the core topologies of 
'hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py' and 'hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py'.
The mathematical bridge between the two structures lies in the integration of stylometry features 
with the evidence extraction and weighting mechanisms. Specifically, the stylometry features are 
used to generate a weighted representation of the text, which is then used to inform the evidence 
extraction and weighting process.

Parent A: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py
Parent B: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py
"""

import re
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Union

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
# Parent B – stylometry / LSM utilities (simplified)
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
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def words(text: str) -> List[str]:
    """Lower‑case alphabetic tokens (apostrophe‑aware)."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96‑dimensional numeric representation of *text*.
    The implementation uses a SHA‑256 hash to seed a pseudo‑random generator,
    guaranteeing reproducibility without external corpora.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96)


def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the proportion of words belonging to each FUNCTION_CAT.
    """
    word_list = words(text)
    func_cat_counts = Counter()
    for word in word_list:
        for cat, words_in_cat in FUNCTION_CATS.items():
            if word in words_in_cat:
                func_cat_counts[cat] += 1
    return np.array([func_cat_counts[cat] / len(word_list) for cat in FUNCTION_CATS])


def _raw_counts(text: str) -> dict[str, int]:
    """
    Extract raw counts of evidence-related features from the input text.
    """
    raw_counts = {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
        "outcome": len(OUTCOME_RE.findall(text)),
        "impulsive": len(IMPULSIVE_RE.findall(text)),
        "scarcity": len(SCARCITY_RE.findall(text)),
        "risk": len(RISK_RE.findall(text)),
    }
    return raw_counts


def hybrid_evidence_extraction(text: str) -> np.ndarray:
    """
    Extract evidence-related features from the input text, using both regex-based extraction and stylometry features.
    """
    raw_counts = _raw_counts(text)
    stylometry_vec = stylometry_features(text)
    lsm_vec = lsm_vector(text)
    hybrid_features = np.concatenate([stylometry_vec, lsm_vec, np.array(list(raw_counts.values()))])
    return hybrid_features


def hybrid_weighting(features: np.ndarray) -> np.ndarray:
    """
    Apply weighting to the hybrid features, using a combination of positive and negative weights.
    """
    positive_weights = _POSITIVE_WEIGHTS
    negative_weights = _NEGATIVE_WEIGHTS
    weighted_features = features[:96] * positive_weights[:96] + features[96:] * negative_weights[96:]
    return weighted_features


def hybrid_inference(text: str) -> np.ndarray:
    """
    Perform hybrid inference on the input text, using both evidence extraction and stylometry features.
    """
    hybrid_features = hybrid_evidence_extraction(text)
    weighted_features = hybrid_weighting(hybrid_features)
    return weighted_features


if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    result = hybrid_inference(text)
    print(result)