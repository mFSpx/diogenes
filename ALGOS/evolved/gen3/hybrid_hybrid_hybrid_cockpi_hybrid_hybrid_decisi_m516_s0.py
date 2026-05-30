# DARWIN HAMMER — match 516, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s2.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s4.py (gen2)
# born: 2026-05-29T23:29:26Z

"""
Hybrid module fusing the mathematical interfaces of 
hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s2.py and 
hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s4.py.

The core mathematical bridge lies in the fusion of linguistic style matching 
from the cockpit metrics and the weighted feature extraction from the decision 
hygiene module. By treating the weighted feature extraction as a modulation 
factor on the linguistic style matching, we establish a hybrid model that 
integrates the strengths of both parent modules.

The mathematical interface is established through the following equations:
1. lsm_vector(text) = {cat: sum(cnt[w] for w in vocab) / total} 
   characterizes the linguistic style of a given text.
2. feature_weights = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64) 
   provides the weighted feature extraction.

By treating the cockpit metrics as a weighting factor on the feature weights, 
we obtain a trust-weighted feature extraction that fuses the two topologies.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

# Constants and regular expressions for feature extraction
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

EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known-good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def _raw_counts(text: str) -> dict[str, int]:
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

def _weighted_feature_extraction(text: str, weight: float) -> np.ndarray:
    counts = _raw_counts(text)
    feature_vector = np.array([counts["evidence_count"], counts["planning_count"], counts["delay_count"], counts["support_count"], counts["boundary_count"], counts["outcome_count"], counts["impulsive_count"], counts["scarcity_count"], counts["risk_count"]])
    weighted_feature_vector = weight * (_POSITIVE_WEIGHTS * feature_vector + _NEGATIVE_WEIGHTS * (1 - feature_vector))
    return weighted_feature_vector

def hybrid_lsm_score(text_a: str, text_b: str, weight: float) -> float:
    """Hybrid LSM score between two vectors."""
    feature_vector_a = _weighted_feature_extraction(text_a, weight)
    feature_vector_b = _weighted_feature_extraction(text_b, weight)
    return np.dot(feature_vector_a, feature_vector_b) / (np.linalg.norm(feature_vector_a) * np.linalg.norm(feature_vector_b))

def hybrid_cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int, text: str) -> float:
    """Hybrid cockpit honesty score."""
    weight = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    feature_vector = _weighted_feature_extraction(text, weight)
    return np.mean(feature_vector)

def hybrid_anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int, text: str) -> float:
    """Hybrid anti-slop ratio."""
    weight = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    feature_vector = _weighted_feature_extraction(text, weight)
    return np.mean(feature_vector)

if __name__ == "__main__":
    text_a = "This is a test sentence with evidence and planning."
    text_b = "This is another test sentence with delay and support."
    displayed_ok = 10
    unknown_displayed_as_ok = 2
    claims_with_evidence = 5
    total_claims_emitted = 10
    print(hybrid_lsm_score(text_a, text_b, 0.5))
    print(hybrid_cockpit_honesty(displayed_ok, unknown_displayed_as_ok, text_a))
    print(hybrid_anti_slop_ratio(claims_with_evidence, total_claims_emitted, text_b))