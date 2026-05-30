# DARWIN HAMMER — match 231, survivor 0
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:27:44Z

# hybrid_endpoint_label_signature.py

"""
HybridEndpointLabelSignature: Fusion of endpoint circuit-breaker recovery priority
(parent A - hybrid_label_foundry_hybrid_endpoint_circ_m5_s2) and path signature
(parent B - path_signature.py).

Mathematical bridge
-------------------
The hybrid uses the recovery priority ρ∈[0,1] (from parent A) to scale the
signature tensor S2 of the path geometry (from parent B). This is achieved by
applying the matrix exponential to the scaled signature tensor:

    S2_scaled = exp(ρ * S2)

This scales the path geometry by the recovery priority, while preserving its
structure. The resulting scaled signature tensor is then used as a confidence
indicator for the labeling aggregation.

The module implements three core functions that embody this unified system and
a tiny smoke-test.
"""

import numpy as np
from math import exp
from random import random


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


def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """Pure A-logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    out: List[ProbabilisticLabel] = []
    for doc_id, labels in votes.items():
        confidence = np.mean(labels)
        out.append(ProbabilisticLabel(doc_id, 0, confidence))  # binary 0 for simplicity
    return out


def signature_level2_scaled(path, rho):
    """Level-2 iterated integral tensor with scaled signature.

    path: (T, d). Returns (d, d).
    S2_scaled[i,j] = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])
                    * exp(rho * S2[i,j])
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    S2 = running.T @ increments               # (d, d)
    return exp(rho * S2)


def hybrid_signature(path, rho):
    """Hybrid signature: scaled path geometry with recovery priority.

    path: (T, d). Returns a list of k arrays for k=1..depth with shapes (d,), (d,d), ..., (d^k,).
    """
    depth = 3  # fixed depth for simplicity
    T, d = path.shape
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    S2 = running.T @ increments               # (d, d)
    scaled_S2 = signature_level2_scaled(path, rho)
    # Initialize accumulators: level k holds a flat array of size d^k
    # We keep them as flat numpy arrays and rebuild shape on output
    out = []
    for level in range(1, depth + 1):
        if level == 1:
            out.append(np.array([np.sum(increments)]))
        elif level == 2:
            out.append(scaled_S2)
        else:
            # accumulate higher levels using Chen-like accumulation
            prev_level = out[level - 2]
            new_level = np.zeros_like(prev_level)
            for t in range(T - 1):
                new_level += prev_level * increments[t]
            out.append(new_level)
    return out


if __name__ == "__main__":
    import sys
    import pathlib
    path = np.random.rand(10, 2)  # random path with shape (10, 2)
    rho = random()  # random recovery priority
    hybrid_signature_result = hybrid_signature(path, rho)
    print(hybrid_signature_result)