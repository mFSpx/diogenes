# DARWIN HAMMER — match 1915, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_cockpi_hybrid_hybrid_decisi_m516_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_dense_associa_m1179_s1.py (gen4)
# born: 2026-05-29T23:39:48Z

import numpy as np
import re
import math
from datetime import datetime

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
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:result|consequence|effect|outcome|impact)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:impulsive|impulsivity|urge|urges|impulse)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:scarcity|limited|limitation|resource|resources)\b", re.I)
RISK_RE = re.compile(r"\b(?:risk|risky|danger|dangerous|threat|threaten)\b", re.I)

def lsm_vector(text):
    vocab = {
        "evidence": EVIDENCE_RE.findall(text),
        "planning": PLANNING_RE.findall(text),
        "delay": DELAY_RE.findall(text),
        "support": SUPPORT_RE.findall(text),
        "boundary": BOUNDARY_RE.findall(text),
        "outcome": OUTCOME_RE.findall(text),
        "impulsive": IMPULSIVE_RE.findall(text),
        "scarcity": SCARCITY_RE.findall(text),
        "risk": RISK_RE.findall(text),
    }
    total = sum(len(words) for words in vocab.values())
    return {cat: len(words) / total for cat, words in vocab.items()}

def feature_weights(text):
    lsm = lsm_vector(text)
    weights = _POSITIVE_WEIGHTS.copy()
    for i, cat in enumerate(_FEATURE_ORDER):
        weights[i] *= lsm.get(cat, 0)
    return weights

def pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds, pheromones):
    current_time = datetime.now()
    if surface_key not in pheromones:
        pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": half_life_seconds,
            "created_time": current_time,
        }
    else:
        previous_signal_value = pheromones[surface_key]["signal_value"]
        previous_half_life_seconds = pheromones[surface_key]["half_life_seconds"]
        previous_created_time = pheromones[surface_key]["created_time"]
        elapsed_time = (current_time - previous_created_time).total_seconds()
        decayed_signal_value = previous_signal_value * math.exp(-elapsed_time / previous_half_life_seconds)
        pheromones[surface_key]["signal_value"] = decayed_signal_value
    return pheromones[surface_key]["signal_value"]

def nonlinear_transformation(memory_matrix, b_spline_coeffs):
    transformed_matrix = np.zeros_like(memory_matrix)
    for i in range(memory_matrix.shape[0]):
        for j in range(memory_matrix.shape[1]):
            transformed_matrix[i, j] = np.dot(memory_matrix[i, j], b_spline_coeffs)
    return transformed_matrix

def hybrid_operation(text, pheromones, memory_matrix, b_spline_coeffs):
    weights = feature_weights(text)
    signal_value = pheromone_signal("surface_key", "signal_kind", 1.0, 3600, pheromones)
    transformed_matrix = nonlinear_transformation(memory_matrix, b_spline_coeffs)
    return weights, signal_value, transformed_matrix

if __name__ == "__main__":
    pheromones = {}
    memory_matrix = np.random.rand(10, 10)
    b_spline_coeffs = np.random.rand(10)
    text = "This is a test text with some evidence and planning."
    weights, signal_value, transformed_matrix = hybrid_operation(text, pheromones, memory_matrix, b_spline_coeffs)
    print(weights)
    print(signal_value)
    print(transformed_matrix)