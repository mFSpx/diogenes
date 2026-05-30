# DARWIN HAMMER — match 231, survivor 4
# gen: 3
# parent_a: hybrid_label_foundry_hybrid_endpoint_circ_m5_s2.py (gen2)
# parent_b: path_signature.py (gen0)
# born: 2026-05-29T23:27:44Z

"""HybridLabelSignature: unified system merging weak‑supervision label aggregation (Parent A)
with path‑signature based recovery priority (Parent B).

Mathematical bridge
-------------------
*Parent A* yields a confidence *c*∈[0,1] for each document via majority vote.  
*Parent B* supplies a morphology‑dependent priority ρ∈[0,1] extracted from the
iterated‑integral (level‑2) signature of a document‑specific time‑series
(e.g. token embeddings).  The norm of the level‑2 signature captures the
geometric “shape” of the path; we map it to [0,1] with a smooth sigmoid.

Hybrid quantities:

    c_hybrid = c · ρ                         # confidence scaled by priority
    τ_hybrid = τ_base / (1 + ρ)               # relaxed error threshold

Thus a well‑shaped (high‑ρ) document boosts its label confidence while also
tightening the tolerance for inconsistencies.

The module implements:
* labeling primitives (Parent A),
* signature utilities (Parent B),
* a priority extractor from signatures,
* hybrid confidence/threshold calculators,
* a demo that runs end‑to‑end without external data.
"""

from __future__ import annotations

import sys
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

import numpy as np
from math import exp, tanh


# ----------------------------------------------------------------------
# Parent A – labeling primitives
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


def labeling_function(name: str | None = None) -> Callable[[Callable[[Dict[str, Any]], int]], Callable[[Dict[str, Any]], int]]:
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[Dict[str, Any]], int]) -> Callable[[Dict[str, Any]], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """
    Pure A‑logic: majority vote with confidence = proportion of votes.
    Returns a list of ProbabilisticLabel, one per document.
    """
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label not in (0, 1):
                continue
            votes.setdefault(r.doc_id, []).append(r.label)

    out: List[ProbabilisticLabel] = []
    for doc_id, lbls in votes.items():
        count = sum(lbls)
        total = len(lbls)
        majority = 1 if count > total / 2 else 0
        confidence = count / total if total > 0 else 0.0
        out.append(ProbabilisticLabel(doc_id=doc_id, label=majority, confidence=confidence))
    return out


# ----------------------------------------------------------------------
# Parent B – path signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag transform: interleave (lead, lag) channels for causality encoding.

    Args:
        path: (T, d) array.

    Returns:
        (2T‑1, 2d) array following the Chevyrev‑Kormilitzin convention.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 signature: total increment vector."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """Level‑2 iterated‑integral tensor using left‑point Riemann sums."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running = path[:-1] - path[0]               # (T‑1, d)
    return running.T @ increments               # (d, d)


def signature(path: np.ndarray, depth: int = 3) -> List[np.ndarray]:
    """
    Compute signatures up to a given depth (inclusive).
    Returns a list where element k‑1 corresponds to level k (1‑based).
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    increments = np.diff(path, axis=0)          # (T‑1, d)

    # accumulator for level‑k flattened tensors
    acc: List[np.ndarray] = [np.zeros(d)]               # level‑1 placeholder (will be overwritten)
    acc[0] = signature_level1(path)                     # level‑1 vector

    for level in range(2, depth + 1):
        size = d ** level
        acc.append(np.zeros(size))

    # iterative Chen accumulation
    for inc in increments:
        # update higher levels first to avoid overwriting needed lower‑level values
        for level in range(depth, 1, -1):
            prev = acc[level - 2]                       # level‑(k‑1) flattened
            # reshape previous to (d^(k‑1),) and compute outer product with inc
            outer = np.kron(prev, inc)                  # kron gives correct flattening order
            acc[level - 1] += outer
        # level‑1 update (simple additive)
        acc[0] += inc

    return acc


# ----------------------------------------------------------------------
# Fusion layer – priority from signature & hybrid operations
# ----------------------------------------------------------------------
def signature_priority(path: np.ndarray) -> float:
    """
    Derive a recovery priority ρ∈[0,1] from the geometry of a path.
    We use the Frobenius norm of the level‑2 signature, passed through a
    smooth sigmoid (tanh) to squash into (0,1).

    Args:
        path: (T, d) array representing the document’s trajectory.

    Returns:
        ρ – a scalar priority.
    """
    sig2 = signature_level2(path)
    norm = np.linalg.norm(sig2, ord='fro')
    # tanh maps ℝ→(-1,1); shift/scale to (0,1)
    rho = (tanh(norm) + 1.0) / 2.0
    return rho


def hybrid_confidence(confidence: float, priority: float) -> float:
    """
    Multiply base confidence by priority (c_hybrid = c·ρ) and clip to [0,1].
    """
    return max(0.0, min(1.0, confidence * priority))


def hybrid_threshold(base_threshold: float, priority: float) -> float:
    """
    Relax the error‑detection threshold using the same priority:
    τ_hybrid = τ_base / (1 + ρ).
    """
    if base_threshold < 0:
        raise ValueError("base_threshold must be non‑negative")
    return base_threshold / (1.0 + priority)


def hybrid_aggregate(
    label_batches: List[List[LabelingFunctionResult]],
    doc_paths: Dict[str, np.ndarray],
    base_threshold: float = 0.2,
) -> List[ProbabilisticLabel]:
    """
    End‑to‑end hybrid pipeline:

    1. Aggregate raw labeling functions → (doc_id, label, c).
    2. For each document, compute priority ρ from its path signature.
    3. Produce hybrid confidence c_hybrid and optionally filter by τ_hybrid.

    Args:
        label_batches: list of batches, each batch is a list of LF results.
        doc_paths: mapping doc_id → (T, d) numpy array representing the path.
        base_threshold: τ_base used before hybrid scaling.

    Returns:
        List of ProbabilisticLabel with hybrid confidence.
    """
    # Step 1 – pure A aggregation
    base_labels = {pl.doc_id: pl for pl in aggregate_labels(label_batches)}

    hybrid_labels: List[ProbabilisticLabel] = []
    for doc_id, base_pl in base_labels.items():
        path = doc_paths.get(doc_id)
        if path is None:
            # No geometric information → fallback to base confidence
            hybrid_labels.append(base_pl)
            continue

        # Step 2 – compute priority from signature
        rho = signature_priority(path)

        # Step 3 – combine
        c_h = hybrid_confidence(base_pl.confidence, rho)
        tau_h = hybrid_threshold(base_threshold, rho)

        # Optionally we could discard low‑confidence labels; here we keep them
        hybrid_labels.append(
            ProbabilisticLabel(doc_id=doc_id, label=base_pl.label, confidence=c_h)
        )
    return hybrid_labels


# ----------------------------------------------------------------------
# Demo / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- synthetic labeling functions -----
    @labeling_function()
    def lf_random_positive(doc: dict) -> int:
        """Randomly returns 1 with probability 0.7."""
        return 1 if random.random() < 0.7 else 0

    @labeling_function()
    def lf_random_negative(doc: dict) -> int:
        """Randomly returns 0 with probability 0.6."""
        return 0 if random.random() < 0.6 else 1

    # Create a tiny corpus of three documents
    docs = {
        "doc1": {"text": "alpha"},
        "doc2": {"text": "beta"},
        "doc3": {"text": "gamma"},
    }

    # Apply labeling functions to each doc, collect results in batches
    batch1 = [
        LabelingFunctionResult(lf_name=lf_random_positive.lf_name, doc_id=doc_id, label=lf_random_positive(doc))
        for doc_id, doc in docs.items()
    ]
    batch2 = [
        LabelingFunctionResult(lf_name=lf_random_negative.lf_name, doc_id=doc_id, label=lf_random_negative(doc))
        for doc_id, doc in docs.items()
    ]
    label_batches = [batch1, batch2]

    # ----- synthetic paths (e.g., 2‑D random walks) -----
    np.random.seed(0)
    doc_paths: Dict[str, np.ndarray] = {}
    for doc_id in docs:
        T = 5  # length of path
        d = 2  # dimensionality
        # simple cumulative sum of Gaussian steps to mimic a trajectory
        steps = np.random.randn(T, d)
        path = np.cumsum(steps, axis=0)
        doc_paths[doc_id] = path

    # Run hybrid aggregation
    hybrid_results = hybrid_aggregate(label_batches, doc_paths, base_threshold=0.25)

    # Print results
    for pl in hybrid_results:
        print(f"Document {pl.doc_id}: label={pl.label}, hybrid_confidence={pl.confidence:.3f}")

    sys.exit(0)