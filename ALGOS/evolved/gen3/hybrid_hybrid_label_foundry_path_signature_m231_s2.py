# DARWIN HAMMER — match 231, survivor 2
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:27:44Z

"""HybridSignatureLabeler: Fusion of Path Signature (parent B) and 
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

The module implements three core functions that embody this unified 
system and a tiny smoke-test.
"""

from __future__ import annotations
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
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def compute_path_confidence_factor(path):
    lead_lag_path = lead_lag_transform(path)
    level1_signature = signature_level1(lead_lag_path)
    path_confidence_factor = np.linalg.norm(level1_signature) / (1 + np.linalg.norm(level1_signature))
    return path_confidence_factor

def hybrid_labeling(batches: List[List[LabelingFunctionResult]], path):
    probabilistic_labels = aggregate_labels(batches)
    path_confidence_factor = compute_path_confidence_factor(path)
    hybrid_labels = []
    for label in probabilistic_labels:
        hybrid_confidence = label.confidence * path_confidence_factor
        hybrid_labels.append(ProbabilisticLabel(label.doc_id, label.label, hybrid_confidence))
    return hybrid_labels

def hybrid_error_detection(hybrid_labels, tau_base):
    hybrid_tau = tau_base / (1 + compute_path_confidence_factor(np.random.rand(10, 2)))
    label_errors = []
    for label in hybrid_labels:
        if label.confidence < hybrid_tau:
            suggested_label = 1 - label.label
            error_probability = 1 - label.confidence
            label_errors.append(LabelError(label.doc_id, label.label, suggested_label, error_probability))
    return label_errors

if __name__ == "__main__":
    batches = [[LabelingFunctionResult("lf1", "doc1", 0), LabelingFunctionResult("lf2", "doc1", 0)], 
               [LabelingFunctionResult("lf1", "doc2", 1), LabelingFunctionResult("lf2", "doc2", 1)]]
    path = np.random.rand(10, 2)
    hybrid_labels = hybrid_labeling(batches, path)
    for label in hybrid_labels:
        print(label)