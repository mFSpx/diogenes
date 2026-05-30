# DARWIN HAMMER — match 1030, survivor 1
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py (gen3)
# parent_b: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# born: 2026-05-29T23:32:24Z

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

# Hybrid Perceptual-RBF Deduplication and Hybrid Label Circuit Module
# ==================================================================
# This module fuses the perceptual hashing utilities and RBF surrogate modeling
# from *hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py* (Parent A)
# with the weak-supervision labeling and endpoint circuit-breaker recovery
# priority from *hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py* (Parent B).
#
# Mathematical bridge
# -------------------
# The bridge is established by using the perceptual hash as a clustering key
# for the RBF surrogate models and as a feature for the labeling confidence
# aggregation. The recovery priority from Parent B is used to scale the
# labeling confidence and adapt the error-detection threshold.

Vector = Sequence[float]

# ---------- Parent A: perceptual hashing utilities ----------
def compute_dhash(values: List[float]) -> int:
    """Difference hash: 1 bit per adjacent pair, 1 if decreasing."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Average hash limited to first 64 values."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count('1')

# ---------- Parent B: labeling primitives ----------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1

@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]

@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float

def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A‑logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                if r.doc_id not in votes:
                    votes[r.doc_id] = []
                votes[r.doc_id].append(r.label)
    out = []
    for doc_id, labels in votes.items():
        label = max(set(labels), key=labels.count)
        confidence = labels.count(label) / len(labels)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out

# ---------- Hybrid Module ----------
@dataclass
class RBFSurrogate:
    hash: int
    model: np.ndarray

def compute_combined_hash(dhash: int, phash: int) -> int:
    return dhash ^ phash

def fit_surrogates_by_hash(data: List[Vector], labels: List[int]) -> Dict[int, RBFSurrogate]:
    surrogates = {}
    for hash, points in groupby(data, key=compute_phash):
        points = list(points)
        labels = [labels[i] for i in range(len(data)) if data[i] in points]
        model = np.array([compute_rbf_kernel(point) for point in points])
        surrogates[hash] = RBFSurrogate(hash, model)
    return surrogates

def compute_rbf_kernel(point: Vector) -> float:
    return math.exp(-np.linalg.norm(point) ** 2)

def hybrid_predict(query: Vector, surrogates: Dict[int, RBFSurrogate]) -> ProbabilisticLabel:
    query_hash = compute_phash(query)
    closest_hash = min(surrogates.keys(), key=lambda hash: hamming_distance(query_hash, hash))
    surrogate = surrogates[closest_hash]
    confidence = compute_labeling_confidence(query, surrogate.model)
    label = 1 if confidence > 0.5 else 0
    return ProbabilisticLabel("query", label, confidence)

def compute_labeling_confidence(query: Vector, model: np.ndarray) -> float:
    recovery_priority = compute_recovery_priority(query)
    confidence = np.dot(query, model) * recovery_priority
    return confidence

def compute_recovery_priority(query: Vector) -> float:
    # Simple righting time index for demonstration purposes
    return 1 / (1 + np.linalg.norm(query))

def groupby(data: List[Vector], key: Callable[[Vector], int]) -> Iterable[Tuple[int, Iterable[Vector]]]:
    groups = {}
    for point in data:
        hash = key(point)
        if hash not in groups:
            groups[hash] = []
        groups[hash].append(point)
    for hash, points in groups.items():
        yield hash, points

if __name__ == "__main__":
    # Simple smoke test
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    labels = [0, 1, 0]
    surrogates = fit_surrogates_by_hash(data, labels)
    query = [2, 3, 4]
    result = hybrid_predict(query, surrogates)
    print(result)