# DARWIN HAMMER — match 153, survivor 0
# gen: 3
# parent_a: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py (gen2)
# born: 2026-05-29T23:27:18Z

"""
Module for the hybrid algorithm that fuses the core topologies of 
'hybrid_hard_truth_math_hybrid_minimum_cost__m12_s6.py' and 
'hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s5.py'. 

The mathematical bridge between the two structures is found in the 
integration of linguistic feature extraction and risk assessment. The 
former parent provides a framework for extracting linguistic features 
from text, while the latter parent provides a framework for assessing risk 
based on linguistic cues. This hybrid algorithm combines these two 
frameworks by using the linguistic features extracted from the text to 
inform the risk assessment.
"""

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
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


def risk_assessment(text: str) -> Dict[str, int]:
    risk_dict = {
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
    return risk_dict


def hybrid_score(text: str) -> Tuple[float, Dict[str, float]]:
    lsm_vec = lsm_vector(text)
    risk_dict = risk_assessment(text)
    risk_vec = np.array([risk_dict[feature] for feature in _FEATURE_ORDER])
    overall_score = np.dot(risk_vec, _POSITIVE_WEIGHTS) - np.dot(risk_vec, _NEGATIVE_WEIGHTS)
    detail = {}
    for cat in FUNCTION_CATS:
        av = lsm_vec.get(cat, 0.0)
        detail[cat] = av * (1 + (overall_score / 10000))
    return overall_score, detail


def hybrid_vector(text: str) -> Dict[str, float]:
    lsm_vec = lsm_vector(text)
    risk_dict = risk_assessment(text)
    risk_vec = np.array([risk_dict[feature] for feature in _FEATURE_ORDER])
    overall_score = np.dot(risk_vec, _POSITIVE_WEIGHTS) - np.dot(risk_vec, _NEGATIVE_WEIGHTS)
    detail = {}
    for cat in FUNCTION_CATS:
        av = lsm_vec.get(cat, 0.0)
        detail[cat] = av * (1 + (overall_score / 10000))
    return detail


if __name__ == "__main__":
    text = "This is a sample text for testing the hybrid algorithm."
    overall_score, detail = hybrid_score(text)
    print(f"Overall Score: {overall_score}")
    print(f"Detail: {detail}")
    hybrid_vec = hybrid_vector(text)
    print(f"Hybrid Vector: {hybrid_vec}")