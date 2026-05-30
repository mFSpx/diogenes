# DARWIN HAMMER — match 4022, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2308_s2.py (gen6)
# parent_b: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py (gen2)
# born: 2026-05-29T23:53:08Z

"""Hybrid Algorithm Fusion of Perceptual RBF Surrogate & Morphology‑Driven Semantic Scoring

Parents
-------
* hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m2308_s2.py
  – Provides a radial‑basis function (RBF) surrogate model, Euclidean distance,
    Gaussian RBF kernel and a linear‑system solver.

* hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s6.py
  – Supplies morphology‑derived recovery priority and cosine‑similarity based
    semantic neighbor scoring.

Mathematical Bridge
-------------------
The two lineages intersect on the notion of a *pairwise similarity* between
objects represented by feature vectors **v**.  The RBF parent contributes a
distance‑based kernel `g(d)=exp(-(ε·d)²)` while the semantic parent contributes
a cosine similarity `c(v_i,v_j)`.  We fuse them multiplicatively to obtain a
*semantic‑RBF similarity*:

    s(i,j) = c(v_i,v_j) · g(‖v_i−v_j‖).

Morphology enters via a normalized recovery priority `p(m_j)∈[0,1]`.  The final
hybrid score is a convex combination controlled by `α∈[0,1]`:

    h(i,j) = α·s(i,j) + (1−α)·p(m_j).

The surrogate model predicts a scalar `ŷ(i)` from the feature vector of the
source item.  We let the surrogate modulate the hybrid score, yielding the
ultimate decision function:

    f(i,j) = ŷ(i) · h(i,j).

The code below implements all components, provides three public functions that
exercise the fused mathematics, and ends with a smoke test."""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, List, Tuple

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Radial‑basis surrogate utilities (Parent A)
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function  g(r)=exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve a dense linear system Ax = b via Gauss‑Jordan elimination."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    """Radial‑basis function surrogate model.

    Predicts ŷ(x) = Σ_i w_i·g(‖x−c_i‖) where c_i are centres and w_i are weights.
    """
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        if len(self.centers) != len(self.weights):
            raise ValueError("centers and weights length mismatch")
        total = 0.0
        for c, w in zip(self.centers, self.weights):
            total += w * gaussian(euclidean(x, c), self.epsilon)
        return total


def train_rbf_surrogate(
    centers: List[Tuple[float, ...]],
    targets: List[float],
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """Fit an RBF surrogate to the supplied (center, target) pairs.

    The linear system A·w = t is built with A_ij = g(‖c_i−c_j‖).
    """
    n = len(centers)
    if n != len(targets):
        raise ValueError("centers and targets must have same length")
    A: List[List[float]] = [
        [gaussian(euclidean(ci, cj), epsilon) for cj in centers] for ci in centers
    ]
    w = solve_linear(A, targets)
    return RBFSurrogate(centers=centers, weights=w, epsilon=epsilon)


# ----------------------------------------------------------------------
# Morphology & Recovery Priority (Parent B)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1] derived from morphology."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Semantic similarity utilities (derived from Parent B)
# ----------------------------------------------------------------------


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ----------------------------------------------------------------------
# Hybrid operations (the fused core)
# ----------------------------------------------------------------------


def semantic_rbf_similarity(
    vi: Vector,
    vj: Vector,
    epsilon: float = 1.0,
) -> float:
    """Product of cosine similarity and Gaussian RBF kernel."""
    return cosine_similarity(vi, vj) * gaussian(euclidean(vi, vj), epsilon)


def hybrid_score(
    vi: Vector,
    vj: Vector,
    morph_j: Morphology,
    alpha: float = 0.7,
    epsilon: float = 1.0,
) -> float:
    """Convex combination of semantic‑RBF similarity and recovery priority."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be within [0,1]")
    s = semantic_rbf_similarity(vi, vj, epsilon)
    p = recovery_priority(morph_j)
    return alpha * s + (1.0 - alpha) * p


def surrogate_hybrid_predict(
    surrogate: RBFSurrogate,
    vi: Vector,
    vj: Vector,
    morph_j: Morphology,
    alpha: float = 0.7,
    epsilon: float = 1.0,
) -> float:
    """
    Final decision function f(i,j) = ŷ(i) · h(i,j).

    - ŷ(i) is the surrogate prediction for the source vector vi.
    - h(i,j) is the hybrid score defined above.
    """
    y_hat = surrogate.predict(vi)
    h = hybrid_score(vi, vj, morph_j, alpha, epsilon)
    return y_hat * h


def rank_neighbors(
    query_vec: Vector,
    candidate_vecs: List[Vector],
    candidate_morphs: List[Morphology],
    surrogate: RBFSurrogate,
    alpha: float = 0.7,
    epsilon: float = 1.0,
) -> List[Tuple[int, float]]:
    """
    Rank candidate indices by descending surrogate‑hybrid score.

    Returns a list of (index, score) tuples sorted high→low.
    """
    if len(candidate_vecs) != len(candidate_morphs):
        raise ValueError("candidates and morphologies length mismatch")
    scores = [
        (i, surrogate_hybrid_predict(surrogate, query_vec, vj, mj, alpha, epsilon))
        for i, (vj, mj) in enumerate(zip(candidate_vecs, candidate_morphs))
    ]
    scores.sort(key=lambda pair: pair[1], reverse=True)
    return scores


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create synthetic 5‑dimensional feature vectors
    dim = 5
    centers = [tuple(np.random.rand(dim)) for _ in range(8)]
    targets = [random.uniform(0, 1) for _ in range(8)]

    # Train an RBF surrogate
    surrogate = train_rbf_surrogate(centers, targets, epsilon=1.5)

    # Query vector
    query = tuple(np.random.rand(dim))

    # Candidate pool
    candidates = [tuple(np.random.rand(dim)) for _ in range(6)]
    morphologies = [
        Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(0.1, 5.0),
        )
        for _ in range(6)
    ]

    # Compute and print rankings
    ranking = rank_neighbors(
        query_vec=query,
        candidate_vecs=candidates,
        candidate_morphs=morphologies,
        surrogate=surrogate,
        alpha=0.6,
        epsilon=1.2,
    )
    print("Ranked candidate indices and scores (high → low):")
    for idx, score in ranking:
        print(f"  idx={idx}, score={score:.4f}")

    # Demonstrate direct hybrid score between first two items
    hs = hybrid_score(
        vi=query,
        vj=candidates[0],
        morph_j=morphologies[0],
        alpha=0.6,
        epsilon=1.2,
    )
    print(f"\nHybrid score between query and candidate 0: {hs:.4f}")

    # Show surrogate prediction for the query vector
    pred = surrogate.predict(query)
    print(f"Surrogate prediction for query vector: {pred:.4f}")