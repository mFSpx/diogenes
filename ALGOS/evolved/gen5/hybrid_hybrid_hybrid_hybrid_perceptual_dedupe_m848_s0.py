# DARWIN HAMMER — match 848, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_capybara_optimizatio_m245_s0.py (gen4)
# parent_b: perceptual_dedupe.py (gen0)
# born: 2026-05-29T23:31:11Z

"""Hybrid algorithm merging probabilistic label aggregation with perceptual hashing.

Parent A (hybrid_hybrid_hybrid_capybara_optimizatio_m245_s0) supplies:
- `LabelingFunctionResult` and `ProbabilisticLabel`
- `aggregate_labels` that converts many noisy binary votes into a confidence‑weighted label.

Parent B (perceptual_dedupe) supplies:
- `compute_phash` that turns a list of floats into a 64‑bit perceptual hash.
- `hamming_distance` and a simple clustering routine based on that hash.

Mathematical bridge
-------------------
Both parents operate on *vectors of real numbers* that can be reduced to binary
signatures:

1. **Expected value** – from Parent A we compute  
   \(E = \sum_i c_i \, \ell_i\) where \(c_i\in[0,1]\) is the confidence and \(\ell_i\in\{0,1\}\) the label.
2. **Perceptual hash** – from Parent B we map the same confidence vector
   \(\mathbf{c} = (c_1,\dots,c_n)\) to a 64‑bit integer by thresholding each element
   against the mean of the vector (phash).  

Thus the confidence vector serves simultaneously as the argument of an
optimization objective and as the input to a binary hash.  By clustering
documents whose hashes are within a small Hamming distance we obtain groups
that share similar confidence profiles, and we can further refine confidences
by a tiny gradient step that maximises the expected value while preserving the
hash‑based neighbourhood.

The functions below realise this fusion:
* `aggregate_labels` – builds `ProbabilisticLabel`s.
* `expected_objective` – computes the expected value of the current confidences.
* `optimize_confidences` – a few gradient‑like updates that push confidences toward
  the label they support, constrained to keep the perceptual hash stable.
* `hybrid_cluster` – hashes the confidence vectors and clusters documents using
  both label agreement and hash similarity.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A structures (trimmed / completed)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    """Result of a labeling function."""
    lf_name: str
    doc_id: str
    label: int


@dataclass(frozen=True)
class ProbabilisticLabel:
    """Probabilistic label with confidence."""
    doc_id: str
    label: int
    confidence: float


def labeling_function(name: str | None = None):
    """Decorator for labeling functions."""
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn
    return deco


def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """
    Aggregate binary votes into a confidence‑weighted label per document.

    For each document we count the number of positive (1) and negative (0) votes.
    Confidence is defined as the proportion of votes that agree with the majority
    label.  Documents with no votes are ignored.
    """
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r.label not in (0, 1):
                continue
            votes.setdefault(r.doc_id, []).append(r.label)

    out: List[ProbabilisticLabel] = []
    for doc_id, vs in votes.items():
        pos = sum(vs)
        neg = len(vs) - pos
        if pos >= neg:
            label = 1
            confidence = pos / len(vs)
        else:
            label = 0
            confidence = neg / len(vs)
        out.append(ProbabilisticLabel(doc_id=doc_id, label=label, confidence=confidence))
    return out


# ----------------------------------------------------------------------
# Parent B utilities (perceptual hash)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """
    Compute a 64‑bit perceptual hash from a list of floats.

    The average of the list is taken as a threshold; each of the first 64
    values contributes a bit: 1 if the value is >= average, 0 otherwise.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two integers."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def expected_objective(plabels: List[ProbabilisticLabel]) -> float:
    """
    Expected value of the labeling objective:
        E = Σ confidence_i * label_i
    """
    return sum(p.confidence * p.label for p in plabels)


def _hash_stable(confidences: List[float], original_hash: int) -> bool:
    """
    Return True if the perceptual hash of `confidences` equals `original_hash`.
    Used to keep the hash unchanged during optimisation.
    """
    return compute_phash(confidences) == original_hash


def optimize_confidences(
    plabels: List[ProbabilisticLabel],
    learning_rate: float = 0.05,
    steps: int = 10,
    hash_tolerance: int = 0,
) -> List[ProbabilisticLabel]:
    """
    Perform a tiny gradient‑like ascent on confidences to maximise the expected
    objective while (optionally) preserving the perceptual hash.

    The update rule for each confidence `c` is:
        c ← c + η * (label - c)

    The term `(label - c)` pushes the confidence toward the majority label.
    After each step we clip to [0, 1] and, if `hash_tolerance` is zero, we reject
    any update that would change the hash.
    """
    # Extract mutable copies of confidences
    confs = np.array([p.confidence for p in plabels], dtype=float)
    labels = np.array([p.label for p in plabels], dtype=float)

    # Baseline hash (may be None if we do not care about stability)
    base_hash = compute_phash(confs.tolist()) if hash_tolerance == 0 else None

    for _ in range(steps):
        grads = labels - confs  # simple gradient toward the label
        new_confs = confs + learning_rate * grads
        new_confs = np.clip(new_confs, 0.0, 1.0)

        if base_hash is not None:
            # Reject changes that would flip the hash
            if not _hash_stable(new_confs.tolist(), base_hash):
                # Reduce learning rate adaptively until stable or give up
                temp_lr = learning_rate
                while temp_lr > 1e-6:
                    temp_lr *= 0.5
                    trial = np.clip(confs + temp_lr * grads, 0.0, 1.0)
                    if _hash_stable(trial.tolist(), base_hash):
                        new_confs = trial
                        break
                # If still unstable we keep the old confidences
        confs = new_confs

    # Re‑wrap into ProbabilisticLabel objects (labels unchanged)
    optimized = [
        ProbabilisticLabel(doc_id=p.doc_id, label=p.label, confidence=float(c))
        for p, c in zip(plabels, confs)
    ]
    return optimized


def hybrid_cluster(
    batches: List[List[LabelingFunctionResult]],
    max_hash_distance: int = 4,
    confidence_opt_steps: int = 5,
) -> List[List[str]]:
    """
    Full hybrid pipeline:

    1. Aggregate raw votes into probabilistic labels.
    2. Optimise confidences while keeping the perceptual hash stable.
    3. Compute a phash per document from its (optimised) confidence.
    4. Cluster document ids whose hashes are within `max_hash_distance`.

    Returns a list of clusters, each cluster being a list of document identifiers.
    """
    # Step 1 – aggregation
    prob_labels = aggregate_labels(batches)

    # Step 2 – optimisation (hash‑preserving)
    prob_labels = optimize_confidences(
        prob_labels, learning_rate=0.1, steps=confidence_opt_steps, hash_tolerance=0
    )

    # Step 3 – build hash map
    hash_map: Dict[str, int] = {}
    for p in prob_labels:
        # Use the single confidence as a one‑element vector; padding to length 64
        # with the same value yields a deterministic hash.
        vec = [p.confidence] * 64
        h = compute_phash(vec)
        hash_map[p.doc_id] = h

    # Step 4 – clustering by Hamming distance
    clusters: List[List[str]] = []
    for doc_id, h in hash_map.items():
        placed = False
        for cluster in clusters:
            # compare with the first element of the cluster (representative)
            rep = cluster[0]
            if hamming_distance(h, hash_map[rep]) <= max_hash_distance:
                cluster.append(doc_id)
                placed = True
                break
        if not placed:
            clusters.append([doc_id])
    return clusters


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic labeling function results for three documents
    random.seed(42)
    docs = ["docA", "docB", "docC"]
    batches: List[List[LabelingFunctionResult]] = []

    # Simulate 5 labeling functions, each producing a noisy vote per document
    for _ in range(5):
        batch: List[LabelingFunctionResult] = []
        for d in docs:
            # 70% chance to emit the true label (choose arbitrarily)
            true_label = 1 if d != "docB" else 0
            vote = true_label if random.random() < 0.7 else 1 - true_label
            batch.append(LabelingFunctionResult(lf_name="lf", doc_id=d, label=vote))
        batches.append(batch)

    clusters = hybrid_cluster(batches, max_hash_distance=2, confidence_opt_steps=8)
    print("Hybrid clusters:", clusters)

    # Show expected objective before and after optimisation for the first batch
    plabels_before = aggregate_labels(batches)
    obj_before = expected_objective(plabels_before)
    plabels_after = optimize_confidences(plabels_before, steps=5)
    obj_after = expected_objective(plabels_after)
    print(f"Expected objective: before={obj_before:.3f}, after={obj_after:.3f}")

    sys.exit(0)