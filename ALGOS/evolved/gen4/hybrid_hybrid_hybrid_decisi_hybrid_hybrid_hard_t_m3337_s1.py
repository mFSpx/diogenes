# DARWIN HAMMER — match 3337, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:49:20Z

"""
Module fusing hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py and 
hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py.

The mathematical bridge between the two parents lies in their use of 
weighted feature vectors and stylometry features. The first parent 
utilizes a weighted feature vector to represent text, while the second 
parent uses a stylometry feature vector. We fuse these by using the 
stylometry features as weights for the feature vector.

The governing equation for the fusion is:

Hybrid Feature Vector = (Stylometry Features) * (Weighted Feature Vector)

This allows us to leverage the strengths of both parents: the 
interpretable weighted features of the first parent and the 
deterministic, reproducible stylometry features of the second parent.
"""

import numpy as np
from pathlib import Path
import re
import math
import random
from collections import Counter
from typing import List, Tuple, Union
import hashlib
import sys

# Define regexes and raw count extraction from Parent A
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

# Define stylometry utilities from Parent B
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
    text_words = words(text)
    vector = np.zeros(len(FUNCTION_CATS))
    for cat, words_in_cat in FUNCTION_CATS.items():
        vector[list(FUNCTION_CATS.keys()).index(cat)] = sum(1 for w in text_words if w in words_in_cat) / len(text_words)
    return vector

def _raw_counts(text: str) -> dict[str, int]:
    counts = {
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
    return counts

def hybrid_feature_vector(text: str) -> np.ndarray:
    """
    Compute the hybrid feature vector by fusing stylometry features and 
    weighted feature vector.
    """
    stylometry = stylometry_features(text)
    raw_counts = np.array(list(_raw_counts(text).values()))
    positive_contribution = np.multiply(stylometry, _POSITIVE_WEIGHTS)
    negative_contribution = np.multiply(stylometry, _NEGATIVE_WEIGHTS)
    weighted_feature_vector = np.where(raw_counts > 0, positive_contribution, negative_contribution)
    return weighted_feature_vector

def analyze_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze the input text and return the hybrid feature vector and 
    the stylometry feature vector.
    """
    hybrid_vector = hybrid_feature_vector(text)
    stylometry_vector = stylometry_features(text)
    return hybrid_vector, stylometry_vector

if __name__ == "__main__":
    text = "This is a test text for analyzing."
    hybrid_vector, stylometry_vector = analyze_text(text)
    print("Hybrid Feature Vector:", hybrid_vector)
    print("Stylometry Feature Vector:", stylometry_vector)