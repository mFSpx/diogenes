# DARWIN HAMMER — match 2586, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m842_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_sparse_hybrid_fisher_locali_m406_s0.py (gen4)
# born: 2026-05-29T23:42:55Z

"""
Hybrid fusion of the hybrid_hybrid_hybrid_bandit_label_foundry_m21_s0.py and hybrid_hybrid_hybrid_sparse_hybrid_fisher_locali_m406_s0.py algorithms.
Mathematical bridge: The integration of Count-Min sketch from the former with the hash-based sparse projection from the latter.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Callable, Any
import numpy as np
import hashlib

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
class SparseVector:
    values: List[float]
    sparsity: int

def labeling_function(name: str|None=None):
    def deco(fn: Callable[[dict],int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]:
    votes = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0,1): 
                votes[r.doc_id].append(r.label)
    out = []
    for d, vs in votes.items():
        if not vs: 
            out.append(ProbabilisticLabel(d, 0, 0.5))
            continue
        c = defaultdict(int)
        for v in vs:
            c[v] += 1
        label = 1 if c[1] >= c[0] else 0
        out.append(ProbabilisticLabel(d, label, c[label]/len(vs)))
    return out

def find_label_errors(docs: list[dict], given: list[int], probs: list[float], threshold: float=0.65) -> list[LabelError]:
    if not (len(docs) == len(given) == len(probs)):
        raise ValueError("Input lists must have equal length")
    errors = []
    for d, g, p in zip(docs, given, probs):
        if p < threshold:
            errors.append(LabelError(d['doc_id'], g, 1-g, p))
    return errors

def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    """Hash-based sparse expansion of `values` into a vector of length `m`."""
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(values: List[float], k: int) -> List[int]:
    """Return a binary mask with 1 at the indices of the top-k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]

def hamming(a: List[int], b: List[int]) -> int:
    """Hamming distance between two binary strings."""
    return sum(x != y for x, y in zip(a, b))

def count_min_sketch(values: List[float], m: int, w: int) -> List[float]:
    """Count-Min sketch estimation of the empirical mean reward and its variance."""
    sketch = [0.0] * m
    for v in values:
        sketch[hashlib.sha256(str(v).encode()).digest()%m] += v
    return [s / w for s in sketch]

def sparse_projection(values: List[float], m: int, salt: str = "") -> SparseVector:
    """Hash-based sparse projection of `values` into a vector of length `m`."""
    return SparseVector(expand(values, m, salt), top_k_mask(values, m))

def fisher_localization(vector: List[float], k: int) -> List[float]:
    """Fisher localization of the input vector using the top-k sparse projection."""
    return top_k_mask(vector, k)

def hybrid_operation(values: List[float], m: int, k: int) -> List[float]:
    """Hybrid operation combining the Count-Min sketch and sparse projection."""
    cm = count_min_sketch(values, m, len(values))
    sp = sparse_projection(values, m)
    return [c + s for c, s in zip(cm, sp.values[:k])]

def main():
    # Smoke test
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = 10
    k = 3
    print(hybrid_operation(values, m, k))

if __name__ == "__main__":
    main()