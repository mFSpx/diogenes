# DARWIN HAMMER — match 172, survivor 0
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s1.py (gen2)
# parent_b: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s2.py (gen2)
# born: 2026-05-29T23:25:54Z

"""
This module fuses the hybrid_label_foundry_hybrid_endpoint_circ_m5_s1.py and 
hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s2.py algorithms.

The mathematical bridge between the two structures is the concept of "information 
richness," which is used to determine the likelihood of an endpoint recovering 
from a failure. This richness is calculated based on the Shannon entropy of the 
feature count vector, and this value is then used to adjust the circuit breaker's 
threshold for determining when to open or close the circuit. In this fusion, we 
use the labeling functions from hybrid_label_foundry_hybrid_endpoint_circ_m5_s1.py 
to determine the labels of the endpoints, and then use the information richness to 
adjust the circuit breaker's behavior.

The hybrid score combines the original hygiene score with the entropy-adjusted 
pruning probability. When the observed text is information-rich (high entropy), 
the algorithm prunes less aggressively and preserves more of the hygiene contribution; 
conversely, low-entropy (repetitive) inputs are pruned more heavily.

The module implements:
1. Feature extraction and vectorisation.
2. Shannon entropy calculation.
3. Exponential pruning probability.
4. Hybrid functions that intertwine the two mathematical structures.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, log
from random import random
from sys import exit
from pathlib import Path
from typing import Callable, Dict, List

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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def labeling_function(name: str|None=None):
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__; 
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    votes = {}
    for batch in batches:
        for r in batch:
            if r.label in (0,1): 
                if r.doc_id not in votes: 
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5)); 
            continue
        c = {0: 0, 1: 0}
        for v in vs:
            c[v] += 1
        label = 1 if c[1]>=c[0] else 0
        out.append(ProbabilisticLabel(d,label,c[label]/len(vs)))
    return out

def find_label_errors(docs: List[dict], given: List[int], probs: List[float], threshold: float=0.65) -> List[LabelError]:
    errors = []
    for i, (doc, given_label, prob) in enumerate(zip(docs, given, probs)):
        if prob < threshold:
            errors.append(LabelError(doc['id'], given_label, 1 - given_label, 1 - prob))
    return errors

def calculate_shannon_entropy(vector: np.ndarray) -> float:
    """Calculates the Shannon entropy of a given vector."""
    vector = vector / np.sum(vector)
    entropy = 0.0
    for p in vector:
        if p > 0:
            entropy -= p * log(p, 2)
    return entropy

def calculate_pruning_probability(t: int, vector: np.ndarray, alpha: float=0.1, lambda_: float=1.0) -> float:
    """Calculates the pruning probability at a given time step."""
    entropy = calculate_shannon_entropy(vector)
    gamma = 1 + entropy / np.log2(len(vector))
    return min(1, lambda_ * exp(-alpha * t)) / gamma

def hybrid_score(s: float, t: int, vector: np.ndarray, alpha: float=0.1, lambda_: float=1.0) -> float:
    """Calculates the hybrid score at a given time step."""
    pruning_prob = calculate_pruning_probability(t, vector, alpha, lambda_)
    return s * (1 - pruning_prob)

if __name__ == "__main__":
    # Test the functions
    docs = [{'id': 'doc1'}, {'id': 'doc2'}]
    given = [1, 0]
    probs = [0.8, 0.4]
    label_errors = find_label_errors(docs, given, probs)
    print(label_errors)

    vector = np.array([0.2, 0.3, 0.5])
    entropy = calculate_shannon_entropy(vector)
    print(entropy)

    pruning_prob = calculate_pruning_probability(1, vector)
    print(pruning_prob)

    hybrid_s = hybrid_score(0.5, 1, vector)
    print(hybrid_s)