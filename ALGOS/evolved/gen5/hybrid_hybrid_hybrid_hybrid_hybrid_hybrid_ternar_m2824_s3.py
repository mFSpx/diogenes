# DARWIN HAMMER — match 2824, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m1855_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# born: 2026-05-29T23:46:13Z

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter
import re

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

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

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
    r"\b(?:boundary|boundaries|limit|limits|restriction|restrictions|rule|rules|constraint|constraints)\b",
    re.I,
)

OUTCOME_RE = re.compile(
    r"\b(?:outcome|result|consequence|effect|impact)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:impulsive|impulsivity|urge|urges)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:scarcity|limited|shortage|constraint)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:risk|risky|riskier|danger|dangerous)\b",
    re.I,
)

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    pred = W @ x
    t = x if target is None else target
    return 2 * np.outer(pred - t, x)

def calculate_feature_vector(text):
    feature_vector = np.zeros(len(_FEATURE_ORDER))
    for i, feature in enumerate(_FEATURE_ORDER):
        if feature == "evidence":
            feature_vector[i] = len(EVIDENCE_RE.findall(text))
        elif feature == "planning":
            feature_vector[i] = len(PLANNING_RE.findall(text))
        elif feature == "delay":
            feature_vector[i] = len(DELAY_RE.findall(text))
        elif feature == "support":
            feature_vector[i] = len(SUPPORT_RE.findall(text))
        elif feature == "boundary":
            feature_vector[i] = len(BOUNDARY_RE.findall(text))
        elif feature == "outcome":
            feature_vector[i] = len(OUTCOME_RE.findall(text))
        elif feature == "impulsive":
            feature_vector[i] = len(IMPULSIVE_RE.findall(text))
        elif feature == "scarcity":
            feature_vector[i] = len(SCARCITY_RE.findall(text))
        elif feature == "risk":
            feature_vector[i] = len(RISK_RE.findall(text))
    return feature_vector

def calculate_minimum_cost_tree(feature_vector):
    tree = np.zeros((len(_FEATURE_ORDER), len(_FEATURE_ORDER)))
    for i in range(len(_FEATURE_ORDER)):
        for j in range(len(_FEATURE_ORDER)):
            if i == j:
                tree[i, j] = 0
            else:
                tree[i, j] = _POSITIVE_WEIGHTS[i] + _NEGATIVE_WEIGHTS[j]
    return tree

def calculate_epistemic_certainty(feature_vector):
    certainty = np.zeros(len(EPISTEMIC_FLAGS))
    for i, flag in enumerate(EPISTEMIC_FLAGS):
        if flag == "FACT":
            certainty[i] = feature_vector[0]  # evidence
        elif flag == "PROBABLE":
            certainty[i] = feature_vector[1]  # planning
        elif flag == "POSSIBLE":
            certainty[i] = feature_vector[2]  # delay
        elif flag == "BULLSHIT":
            certainty[i] = feature_vector[6]  # impulsive
        elif flag == "SURE_MAYBE":
            certainty[i] = feature_vector[7]  # scarcity
    return certainty

def route_command(text, W):
    feature_vector = calculate_feature_vector(text)
    epistemic_certainty = calculate_epistemic_certainty(feature_vector)
    tree = calculate_minimum_cost_tree(feature_vector)
    for i in range(len(_FEATURE_ORDER)):
        for j in range(len(_FEATURE_ORDER)):
            tree[i, j] *= (1 + epistemic_certainty[i]) / (1 + epistemic_certainty[j])
    pred = W @ feature_vector
    residual = pred - feature_vector
    loss = float(residual @ residual)
    if loss < 0.5:
        return "The input text is well-structured."
    else:
        return "The input text is not well-structured."

if __name__ == "__main__":
    W = init_ttt(len(_FEATURE_ORDER))
    text = "This is a sample text with evidence and planning."
    response = route_command(text, W)
    print(response)