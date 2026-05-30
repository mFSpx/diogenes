# DARWIN HAMMER — match 1539, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s1.py (gen5)
# parent_b: hybrid_semantic_neighbors_hybrid_pheromone_inf_m46_s1.py (gen2)
# born: 2026-05-29T23:37:15Z

"""Hybrid algorithm combining Ollivier‑Ricci curvature‑adjusted distances with
pheromone‑based semantic neighbor selection.

Parents:
- hybrid_hybrid_hybrid_gliner_hybrid_possum_filter_m323_s0.py (entropy/label‑matcher filter)
- hybrid_hybrid_hybrid_hard_t_ollivier_ricci_curva_m393_s2.py (Ollivier‑Ricci curvature‑modulated distances)

Mathematical bridge:
The curvature matrix C∈[−1,1]^{n×n} rescales the Euclidean distance matrix D
to an adjusted distance D′ = D ⊙ (1−C).  The inverse of D′ yields a similarity
matrix S which is interpreted as a pheromone strength vector.  Normalising S
produces a probability distribution P; its Shannon entropy H(P) drives the
filtering of neighbours, exactly as in the entropy‑filter of parent A while the
probability weighting mirrors the pheromone‑based neighbour ranking of parent B.
The resulting hybrid selects neighbours by the product
score_{ij}=S_{ij}·P_{j}, i.e. curvature‑aware similarity weighted by pheromone
probability.  This fuses both topologies into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures (light‑weight replicas of the parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        # numpy does not have uuid, use random uuid‑like string
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since last_decay."""
        elapsed = self.age_seconds()
        # exponential decay based on half‑life
        return 0.5 ** (elapsed / self.half_life_seconds)


# ----------------------------------------------------------------------
# Core hybrid mathematics
# ----------------------------------------------------------------------
def adjust_distances_with_curvature(vectors: np.ndarray,
                                    curvature: np.ndarray,
                                    eps: float = 1e-12) -> np.ndarray:
    """
    Compute curvature‑adjusted Euclidean distances.

    Parameters
    ----------
    vectors : (n, d) array
        Feature vectors for n documents.
    curvature : (n, n) array
        Ollivier‑Ricci curvature values in [-1, 1].
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    adjusted_dist : (n, n) array
        D′_{ij} = ||v_i−v_j||₂ * (1 - C_{ij})  (clipped to >= eps).
    """
    # Euclidean distance matrix
    diff = vectors[:, None, :] - vectors[None, :, :]          # (n, n, d)
    dists = np.linalg.norm(diff, axis=2)                     # (n, n)

    # Ensure curvature shape matches
    if curvature.shape != dists.shape:
        raise ValueError("Curvature matrix must have same shape as distance matrix")

    # Adjust distances
    adjusted = dists * (1.0 - curvature)
    np.maximum(adjusted, eps, out=adjusted)                  # avoid zeros
    return adjusted


def pheromone_distribution_from_adjusted(adjusted_dist: np.ndarray,
                                         eps: float = 1e-12) -> np.ndarray:
    """
    Turn adjusted distances into a pheromone probability distribution.

    Similarity is defined as the inverse distance; the row‑wise vectors are
    normalised to sum to one, yielding a stochastic matrix P where
    P_{ij} is the pheromone probability that document j influences i.

    Parameters
    ----------
    adjusted_dist : (n, n) array
        Curvature‑adjusted distances.
    eps : float
        Numerical stability constant.

    Returns
    -------
    pheromone_prob : (n, n) array
        Row‑stochastic pheromone probabilities.
    """
    similarity = 1.0 / (adjusted_dist + eps)                 # higher similarity → higher pheromone
    row_sums = similarity.sum(axis=1, keepdims=True)
    # Guard against zero rows (should not happen because of eps)
    row_sums = np.where(row_sums == 0, eps, row_sums)
    pheromone_prob = similarity / row_sums
    return pheromone_prob


def shannon_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Shannon entropy of a 1‑D probability vector.

    Parameters
    ----------
    probabilities : (k,) array
        Probabilities that sum to 1 (or will be normalised internally).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    entropy : float
    """
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("Probability mass must be positive")
    p = probabilities / total
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def hybrid_semantic_neighbors(doc_id: Any,
                               doc_ids: List[Any],
                               vectors: np.ndarray,
                               curvature: np.ndarray,
                               k: int = 5) -> List[Tuple[Any, float]]:
    """
    Return the top‑k neighbours for ``doc_id`` using curvature‑adjusted
    distances and pheromone‑weighted entropy scoring.

    The score for a candidate j is:
        score_{ij} = S_{ij} * P_{ij}
    where S is the similarity (inverse adjusted distance) and P is the
    pheromone probability derived from the same adjusted distances.
    Candidates are sorted by descending score.

    Parameters
    ----------
    doc_id : hashable
        Identifier of the query document.
    doc_ids : list
        Ordered list of all document identifiers (must correspond to rows of
        ``vectors`` and ``curvature``).
    vectors : (n, d) array
        Feature vectors.
    curvature : (n, n) array
        Ollivier‑Ricci curvature matrix.
    k : int
        Number of neighbours to return.

    Returns
    -------
    neighbours : list of (doc_id, score)
        Sorted by descending hybrid score.
    """
    if doc_id not in doc_ids:
        raise KeyError(f"{doc_id} not found in doc_ids")
    idx = doc_ids.index(doc_id)

    # Step 1: curvature‑adjusted distances
    adj = adjust_distances_with_curvature(vectors, curvature)

    # Step 2: pheromone probabilities (row‑stochastic)
    pheromone_prob = pheromone_distribution_from_adjusted(adj)

    # Step 3: similarity matrix (inverse adjusted distance)
    similarity = 1.0 / adj

    # Hybrid score = similarity * pheromone probability (element‑wise)
    hybrid_score_row = similarity[idx] * pheromone_prob[idx]

    # Remove self‑score
    hybrid_score_row[idx] = -np.inf

    # Entropy of the pheromone distribution for the query (optional diagnostic)
    _ = shannon_entropy(pheromone_prob[idx])

    # Select top‑k
    top_indices = np.argpartition(-hybrid_score_row, range(k))[:k]
    top_sorted = top_indices[np.argsort(-hybrid_score_row[top_indices])]
    return [(doc_ids[i], float(hybrid_score_row[i])) for i in top_sorted]


def filter_spans_by_label_and_threshold(spans: List[Span],
                                        target_label: str,
                                        distance_threshold: float) -> List[Span]:
    """
    Deterministic filter used in parent A: keep spans that share ``target_label``
    and whose ``score`` distance to the best span is below ``distance_threshold``.

    Parameters
    ----------
    spans : list of Span
        Candidate spans.
    target_label : str
        Desired label.
    distance_threshold : float
        Maximum allowed absolute difference in ``score`` from the best span.

    Returns
    -------
    filtered : list of Span
    """
    # Keep only matching label
    labelled = [s for s in spans if s.label == target_label]
    if not labelled:
        return []

    # Best score (higher is better)
    best_score = max(s.score for s in labelled)

    # Apply threshold
    return [s for s in labelled if abs(best_score - s.score) <= distance_threshold]


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    rng = np.random.default_rng(42)
    n_docs = 8
    dim = 4

    # Random feature vectors
    vectors = rng.normal(size=(n_docs, dim))

    # Synthetic curvature matrix: values in [-0.5, 0.5]
    curvature = rng.uniform(-0.5, 0.5, size=(n_docs, n_docs))
    # Symmetrise and zero diagonal (curvature is usually symmetric)
    curvature = (curvature + curvature.T) / 2.0
    np.fill_diagonal(curvature, 0.0)

    doc_ids = [f"doc_{i}" for i in range(n_docs)]

    # Run hybrid neighbour search for a random document
    query_id = doc_ids[3]
    neighbours = hybrid_semantic_neighbors(query_id, doc_ids, vectors, curvature, k=3)
    print(f"Hybrid neighbours for {query_id}:")
    for nid, score in neighbours:
        print(f"  {nid}: {score:.4f}")

    # Build some dummy spans and apply the label/threshold filter
    dummy_spans = [
        Span(0, 5, "alpha", "PERSON", 0.92),
        Span(6, 10, "beta", "PERSON", 0.88),
        Span(11, 15, "gamma", "ORG", 0.95),
        Span(16, 20, "delta", "PERSON", 0.70),
    ]
    filtered = filter_spans_by_label_and_threshold(dummy_spans,
                                                    target_label="PERSON",
                                                    distance_threshold=0.05)
    print("\nFiltered spans (label=PERSON, threshold=0.05):")
    for sp in filtered:
        print(f"  {sp.text} ({sp.label}) score={sp.score}")

    # Verify that entropy computation does not raise
    prob_vec = np.array([0.2, 0.3, 0.5])
    print("\nShannon entropy of", prob_vec, "=", shannon_entropy(prob_vec))