# DARWIN HAMMER — match 2356, survivor 2
# gen: 5
# parent_a: hybrid_indy_learning_vector_hybrid_hybrid_hybrid_m823_s0.py (gen4)
# parent_b: hybrid_pheromone_infotaxis_m3_s1.py (gen1)
# born: 2026-05-29T23:41:55Z

"""
Hybrid Algorithm: Fusing DARWIN HAMMER — match 823, survivor 0 (hybrid_indy_learning_vector_hybrid_hybrid_hybrid_m823_s0.py) 
and DARWIN HAMMER — match 3, survivor 1 (hybrid_pheromone_infotaxis_m3_s1.py)

This module mathematically fuses the core topologies of hybrid_indy_learning_vector_hybrid_hybrid_hybrid_m823_s0.py 
and hybrid_pheromone_infotaxis_m3_s1.py. The bridge between the two parents lies in the application of 
regret-weighted strategy (Parent A) and pheromone signals with their decay rates (Parent B). 
The hybrid algorithm:

1. Builds a vector of regex-based feature counts from text (Parent A).
2. Applies separate positive and negative weight vectors to yield a raw utility vector.
3. Generates a regret-weighted probability vector (Parent A).
4. Computes the Gini coefficient of the resulting probability distribution.
5. Integrates the pheromone signal system with the entropy search algorithms (Parent B).

The mathematical interface between the two parents is established through the use of utility vectors 
and pheromone signals.

"""

import numpy as np
import re
import json
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Any, List, Dict

# Constants
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)

# Regular expressions
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

def load_go_terms(root: Path) -> List[str]:
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)

def tokenize(text: str) -> List[Dict[str, Any]]:
    return [{"token": m.group(0), "start": m.start(), "end": m.end()} for m in EVIDENCE_RE.finditer(text)] + \
           [{"token": m.group(0), "start": m.start(), "end": m.end()} for m in PLANNING_RE.finditer(text)]

def calculate_feature_counts(text: str) -> Dict[str, int]:
    tokens = tokenize(text)
    return Counter(token["token"] for token in tokens)

def apply_weights(feature_counts: Dict[str, int], positive_weights: Dict[str, float], negative_weights: Dict[str, float]) -> np.ndarray:
    utility_vector = np.zeros(len(feature_counts))
    for i, (feature, count) in enumerate(feature_counts.items()):
        utility_vector[i] = count * (positive_weights.get(feature, 0) - negative_weights.get(feature, 0))
    return utility_vector

def regret_weighted_probability(utility_vector: np.ndarray) -> np.ndarray:
    exp_utility_vector = np.exp(utility_vector)
    return exp_utility_vector / np.sum(exp_utility_vector)

def gini_coefficient(probability_vector: np.ndarray) -> float:
    return 1 - np.sum(np.square(probability_vector))

def calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds):
    """
    Calculates the pheromone signal strength based on the surface key, signal kind, signal value, and half-life seconds.
    """
    return signal_value * math.pow(0.5, 1 / half_life_seconds)

def calculate_entropy(probabilities):
    """
    Calculates the entropy of a given probability distribution.
    """
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    probabilities = [p/total for p in probabilities if p > 0]
    return -sum(p * math.log(p) for p in probabilities)

def expected_entropy(p_hit, hit_state, miss_state):
    """
    Calculates the expected entropy of a given probability distribution and hit/miss states.
    """
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)

def hybrid_operation(text: str, positive_weights: Dict[str, float], negative_weights: Dict[str, float], 
                      surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: float) -> float:
    feature_counts = calculate_feature_counts(text)
    utility_vector = apply_weights(feature_counts, positive_weights, negative_weights)
    probability_vector = regret_weighted_probability(utility_vector)
    gini = gini_coefficient(probability_vector)
    pheromone_signal = calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
    hit_state = [probability_vector[i] * pheromone_signal for i in range(len(probability_vector))]
    miss_state = [1 - p for p in hit_state]
    expected_ent = expected_entropy(pheromone_signal, hit_state, miss_state)
    return gini, expected_ent

if __name__ == "__main__":
    text = "The evidence suggests that the plan is to verify the source."
    positive_weights = {"evidence": 1.0, "plan": 1.0}
    negative_weights = {"verify": -1.0, "source": -1.0}
    surface_key = "test_surface"
    signal_kind = "test_signal"
    signal_value = 1.0
    half_life_seconds = 10.0
    gini, expected_ent = hybrid_operation(text, positive_weights, negative_weights, surface_key, signal_kind, signal_value, half_life_seconds)
    print(f"Gini Coefficient: {gini}")
    print(f"Expected Entropy: {expected_ent}")