# DARWIN HAMMER — match 153, survivor 1
# gen: 3
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# born: 2026-05-29T23:27:18Z

"""
Hybrid Algorithm: fusing hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py and hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py

This hybrid algorithm integrates the linguistic feature extraction and similarity scoring from 
hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py with the regex-based evidence extraction and 
weighted scoring from hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py.

The mathematical bridge between the two algorithms lies in the use of vectorized operations and 
weighted scoring. The linguistic features extracted from the text data are used to compute a 
similarity score, which is then combined with the weighted evidence scores to produce a final 
output.
"""

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

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

def words(text: str) -> List[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())

def lsm_vector(text: str) -> Dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail

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

def hybrid_score(text1: str, text2: str) -> Tuple[float, Dict[str, float]]:
    vector1 = lsm_vector(text1)
    vector2 = lsm_vector(text2)
    lsm_similarity, _ = lsm_score(vector1, vector2)

    counts1 = _raw_counts(text1)
    counts2 = _raw_counts(text2)

    evidence_score = (counts1["evidence"] + counts2["evidence"]) / (1 + abs(counts1["evidence"] - counts2["evidence"]))
    planning_score = (counts1["planning"] + counts2["planning"]) / (1 + abs(counts1["planning"] - counts2["planning"]))

    weighted_score = (
        (_POSITIVE_WEIGHTS * np.array([counts1[f] for f in _FEATURE_ORDER])) +
        (_NEGATIVE_WEIGHTS * np.array([counts2[f] for f in _FEATURE_ORDER]))
    ).sum() / (_POSITIVE_WEIGHTS.sum() + _NEGATIVE_WEIGHTS.sum())

    hybrid_similarity = 0.5 * lsm_similarity + 0.5 * weighted_score
    return hybrid_similarity, {}

if __name__ == "__main__":
    text1 = "This is a test sentence with evidence and planning."
    text2 = "This sentence also has evidence but less planning."
    score, _ = hybrid_score(text1, text2)
    print(f"Hybrid similarity score: {score:.4f}")