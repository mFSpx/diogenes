# DARWIN HAMMER — match 1631, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s7.py (gen3)
# parent_b: hybrid_hybrid_label_foundry_path_signature_m231_s2.py (gen3)
# born: 2026-05-29T23:37:51Z

"""
Hybrid Algorithm: Fusion of DARWIN HAMMER (hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s7.py) 
and HybridSignatureLabeler (hybrid_hybrid_label_foundry_path_signature_m231_s2.py) through 
confidence modulation and regex-based feature extraction.

The mathematical bridge between the two parents lies in the confidence 
modulation scheme of HybridSignatureLabeler and the feature extraction 
mechanism of DARWIN HAMMER. Specifically, we use the regex-based features 
from DARWIN HAMMER to compute a feature confidence factor that scales 
the confidence and threshold of HybridSignatureLabeler.

Governing equations:
- Feature confidence factor (ρ) is computed using the regex-based features
- Confidence modulation: c_hybrid = c · ρ
- Threshold modulation: τ_hybrid = τ_base / (1 + ρ)
"""

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from random import random
from typing import Any, Callable, Dict, List
import re
import sys

@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out = []
    for doc_id, labels in votes.items():
        label = Counter(labels).most_common(1)[0][0]
        confidence = labels.count(label) / len(labels)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out

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
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+ to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
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
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 2000, 2500, 3000], dtype=np.int64)

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

def compute_feature_confidence_factor(text: str) -> float:
    features = []
    for feature, regex in _REGEX_MAP.items():
        if regex.search(text):
            features.append(feature)
    feature_confidence_factor = sum(_POSITIVE_WEIGHTS[_FEATURE_ORDER.index(feature)] for feature in features if feature in _FEATURE_ORDER[:6]) / (sum(_POSITIVE_WEIGHTS) + sum(_NEGATIVE_WEIGHTS))
    feature_confidence_factor += sum(_NEGATIVE_WEIGHTS[_FEATURE_ORDER.index(feature)] for feature in features if feature in _FEATURE_ORDER[6:]) / (sum(_POSITIVE_WEIGHTS) + sum(_NEGATIVE_WEIGHTS))
    return feature_confidence_factor

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T):
        out[2*t, :d] = path[t]
        out[2*t, d:] = path[t-1] if t > 0 else 0
        if t < T - 1:
            out[2*t+1, :d] = path[t+1] - path[t]
            out[2*t+1, d:] = path[t] - path[t-1] if t > 0 else 0
    return out

def hybrid_labeling_function(text: str, path: List[float]) -> ProbabilisticLabel:
    feature_confidence_factor = compute_feature_confidence_factor(text)
    lead_lag_path = lead_lag_transform(np.array(path).reshape(-1, 1))
    path_confidence_factor = np.mean(np.linalg.norm(lead_lag_path, axis=1))
    confidence = 0.5 * feature_confidence_factor * path_confidence_factor
    label = 1 if confidence > 0.5 else 0
    return ProbabilisticLabel("doc_id", label, confidence)

if __name__ == "__main__":
    text = "The evidence suggests that the plan is working."
    path = [1, 2, 3, 4, 5]
    label = hybrid_labeling_function(text, path)
    print(label)