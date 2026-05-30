# DARWIN HAMMER — match 231, survivor 3
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:27:44Z

"""HybridPathLabelFoundry: Fusion of Path Signature (parent B) and 
Hybrid Label Foundry (parent A) via a confidence-boosting recovery priority.

Mathematical bridge
-------------------
Parent A produces a confidence c∈[0,1] for each document via vote majority.  
Parent B defines a recovery priority ρ∈[0,1] based on path signature morphology.

The hybrid uses ρ as a multiplicative scaling factor for the confidence 
produced by the labeling aggregation:

    c_hybrid = c · ρ

Thus a well-shaped (high ρ) path boosts confidence, while a fragile 
one attenuates it. Conversely, the error-detection threshold is relaxed 
by the same factor:

    τ_hybrid = τ_base / (1 + ρ)

so that high-priority morphologies tolerate fewer apparent errors.

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

# ----------------------------------------------------------------------
# Parent A – labeling primitives
# ----------------------------------------------------------------------
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
    out = []
    for doc_id, labels in votes.items():
        label = Counter(labels).most_common(1)[0][0]
        confidence = labels.count(label) / len(labels)
        out.append(ProbabilisticLabel(doc_id, label, confidence))
    return out


# ----------------------------------------------------------------------
# Parent B – path signature
# ----------------------------------------------------------------------
def lead_lag_transform(path):
    """Lead-lag transform: interleave (lead, lag) channels for causality encoding.

    path: (T, d). Returns (2T-1, 2d) interleaved lead-lag path.

    At even indices 2t   : (X_t,   X_t)    (lead and lag both at t)
    At odd  indices 2t+1 : (X_{t+1}, X_t)  (lead advances, lag holds)
    This matches the standard Chevyrev-Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path):
    """Level-1 signature: total increment vector.

    path: (T, d). Returns (d,). Equal to path[-1] - path[0].
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path):
    """Level-2 iterated integral tensor.

    path: (T, d). Returns (d, d).
    S2[i,j] = sum_{s<t} (X_s[i] - X_0[i]) * dX_t[j]
             = sum_{t=1..T-1} (X_{t-1}[i] - X_0[i]) * (X_t[j] - X_{t-1}[j])

    Uses the standard left-point Riemann sum on the increment path.
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T-1, d)
    running    = path[:-1] - path[0]            # (T-1, d)  X_{t-1} - X_0
    # S2[i,j] = sum_t running[t,i] * increments[t,j]
    return running.T @ increments               # (d, d)


def compute_recovery_priority(path):
    """Compute recovery priority ρ∈[0,1] based on path signature morphology."""
    level1 = signature_level1(path)
    level2 = signature_level2(path)
    # Compute a simple recovery priority based on the norms of level 1 and 2 signatures
    rho = np.linalg.norm(level1) * np.linalg.norm(level2)
    rho = np.clip(rho, 0, 1)  # clamp to [0,1]
    return rho


# ----------------------------------------------------------------------
# Hybrid Logic
# ----------------------------------------------------------------------
def hybrid_labeling(batches: List[List[LabelingFunctionResult]], paths: List[np.ndarray], 
                    tau_base: float = 0.5) -> List[ProbabilisticLabel]:
    """Hybrid labeling with confidence boosted by recovery priority."""
    labels = aggregate_labels(batches)
    for i, label in enumerate(labels):
        path = paths[i]
        rho = compute_recovery_priority(path)
        confidence = label.confidence * rho
        # Relax error-detection threshold by the same factor
        tau_hybrid = tau_base / (1 + rho)
        # Update label with hybrid confidence
        labels[i] = ProbabilisticLabel(label.doc_id, label.label, confidence)
    return labels


def evaluate_hybrid_performance(hybrid_labels: List[ProbabilisticLabel], 
                                true_labels: List[int]) -> float:
    """Evaluate hybrid labeling performance."""
    correct = sum(1 for label, true_label in zip(hybrid_labels, true_labels) 
                   if label.label == true_label)
    accuracy = correct / len(true_labels)
    return accuracy


if __name__ == "__main__":
    # Smoke test
    batches = [[LabelingFunctionResult("lf1", "doc1", 1), 
                LabelingFunctionResult("lf2", "doc1", 1)],
               [LabelingFunctionResult("lf1", "doc2", 0), 
                LabelingFunctionResult("lf2", "doc2", 0)]]
    paths = [np.random.rand(10, 2), np.random.rand(10, 2)]
    true_labels = [1, 0]
    hybrid_labels = hybrid_labeling(batches, paths)
    accuracy = evaluate_hybrid_performance(hybrid_labels, true_labels)
    print(f"Hybrid labeling accuracy: {accuracy:.4f}")