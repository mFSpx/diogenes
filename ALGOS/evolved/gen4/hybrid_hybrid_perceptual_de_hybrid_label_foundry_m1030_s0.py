# DARWIN HAMMER — match 1030, survivor 0
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s7.py (gen3)
# parent_b: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# born: 2026-05-29T23:32:24Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Tuple, Dict

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

def hamming_distance(a: int, b: int) -> float:
    """Hamming distance between two integers."""
    x = a ^ b
    m = 0
    while x:
        m += x & 1
        x >>= 1
    return m

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
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out: List[ProbabilisticLabel] = []
    for doc_id, votes in votes.items():
        labels = Counter(votes)
        max_votes = max(labels.values())
        confidence = max_votes / len(votes)
        out.append(ProbabilisticLabel(doc_id, 1 if max_votes > len(votes) / 2 else 0, confidence))
    return out

def labeling_confidence(doc_id: str, votes: List[int]) -> float:
    """Compute confidence = proportion of votes."""
    labels = Counter(votes)
    max_votes = max(labels.values())
    confidence = max_votes / len(votes)
    return confidence

def compute_recovery_priority(m: float) -> float:
    """Compute recovery priority = clamp(righting_time_index(m) / max_index)."""
    return min(max(m, 0), 1)

# ---------- Hybrid Perceptual-Labeling Circuit ----------
def hybrid_labeling_confidence(doc_id: str, votes: List[int], morphology: float) -> float:
    """Compute hybrid confidence = confidence · ρ."""
    confidence = labeling_confidence(doc_id, votes)
    recovery_priority = compute_recovery_priority(morphology)
    return confidence * recovery_priority

def hybrid_error_probability(doc_id: str, given_label: int, suggested_label: int, votes: List[int], morphology: float) -> float:
    """Compute hybrid error probability = error_probability / (1 + ρ)."""
    error_probability = labeling_confidence(doc_id, votes)
    recovery_priority = compute_recovery_priority(morphology)
    return error_probability / (1 + recovery_priority)

def hybrid_perceptual_labeling(values: List[float], morphology: float) -> int:
    """Compute hybrid label = phash."""
    phash = compute_phash(values)
    return phash

def hybrid_predict(query: int, cluster_hashes: List[int], cluster_surrogates: List[float]) -> float:
    """Compute hybrid prediction = surrogate of closest hash."""
    closest_hash = min(cluster_hashes, key=lambda x: hamming_distance(x, query))
    closest_surrogate = cluster_surrogates[cluster_hashes.index(closest_hash)]
    return closest_surrogate

# ---------- Smoke Test ----------
if __name__ == "__main__":
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    morphology = 0.5
    phash = hybrid_perceptual_labeling(values, morphology)
    print(f"Perceptual hash: {phash}")
    votes = [1, 1, 0, 0, 1]
    confidence = hybrid_labeling_confidence("doc_id", votes, morphology)
    print(f"Hybrid confidence: {confidence}")
    error_probability = hybrid_error_probability("doc_id", 1, 0, votes, morphology)
    print(f"Hybrid error probability: {error_probability}")