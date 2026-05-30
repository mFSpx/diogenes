# DARWIN HAMMER — match 1631, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s7.py (gen3)
# parent_b: hybrid_hybrid_label_foundry_path_signature_m231_s2.py (gen3)
# born: 2026-05-29T23:37:51Z

# hybrid_hybrid_darwin_hammer_label_foundry_m2357_s4.py

"""
HybridSignatureLabeler: Fusion of Path Signature (parent B) and 
Hybrid Label Foundry (parent A) through confidence modulation.

Mathematical bridge
-------------------
Parent A produces a confidence *c*∈[0,1] for each document via vote 
majority and defines an error-detection threshold *τ*.  
Parent B generates a path signature *S* that captures the geometry of 
a given path. The hybrid uses the lead-lag transform of the path 
to modulate the confidence *c* and threshold *τ*.

Specifically, we use the level-1 signature (total increment vector) 
of the lead-lag transformed path to compute a *path confidence factor* 
*ρ*∈[0,1], which scales the confidence and threshold:

    c_hybrid = c · ρ
    τ_hybrid = τ_base / (1 + ρ)

Parent A: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s7.py
Parent B: hybrid_hybrid_label_foundry_path_signature_m231_s2.py
"""

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp
from pathlib import Path
from random import random
from typing import Any, Callable, Dict, List

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

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T):
        out[t] = path[t]
        out[T + t] = path[t]
    out[T] = np.sum(path[:T], axis=0)
    return out

def hybrid_path_signature(path):
    signature = lead_lag_transform(path)
    increment_vector = np.sum(signature, axis=0)
    path_confidence_factor = np.exp(-np.linalg.norm(increment_vector)) / (1 + np.exp(-np.linalg.norm(increment_vector)))
    return path_confidence_factor

def hybridize_confidence(confidence, path_confidence_factor):
    return confidence * path_confidence_factor

def hybridize_threshold(threshold, path_confidence_factor):
    return threshold / (1 + path_confidence_factor)

def hybrid_label_foundry(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    probabilities = aggregate_labels(batches)
    thresholds = np.array([0.5] * len(probabilities))  # default threshold
    out = []
    for i, p in enumerate(probabilities):
        path_confidence_factor = hybrid_path_signature(p.doc_id)  # assume doc_id contains path data
        hybrid_confidence = hybridize_confidence(p.confidence, path_confidence_factor)
        hybrid_threshold = hybridize_threshold(thresholds[i], path_confidence_factor)
        label = 1 if hybrid_confidence > hybrid_threshold else 0
        out.append(ProbabilisticLabel(p.doc_id, label, hybrid_confidence))
    return out

def smoke_test():
    import sys
    batches = [
        [LabelingFunctionResult('lf1', 'doc1', 1), LabelingFunctionResult('lf2', 'doc1', 0)],
        [LabelingFunctionResult('lf3', 'doc2', 1), LabelingFunctionResult('lf4', 'doc2', 0)],
    ]
    results = hybrid_label_foundry(batches)
    for result in results:
        print(result)

if __name__ == "__main__":
    smoke_test()