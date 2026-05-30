# DARWIN HAMMER — match 3350, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_hybrid_m1710_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1303_s0.py (gen5)
# born: 2026-05-29T23:49:25Z

"""Hybrid algorithm merging:
- Parent A: Hoeffding bound, Gini impurity, pheromone decay.
- Parent B: Radial‑basis‑function (Gaussian) similarity matrix, Shannon entropy weighting.

Mathematical bridge:
The similarity matrix S computed from feature vectors is used to weight the
information‑theoretic quantities of Parent A.  Specifically, the Hoeffding
epsilon is scaled by the average similarity of the current sample,
the Gini gain is multiplied by an entropy‑based confidence factor, and a
pheromone level is decayed proportionally to the same weighted confidence.
Thus the statistical split test of a Hoeffding tree becomes aware of the
geometric relationships among samples encoded by the RBF kernel."""

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (r² * ln(1/δ)) / (2n) )."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_impurity(labels: Iterable[int]) -> float:
    """Gini impurity of a categorical label distribution."""
    total = 0
    counts: Counter = Counter()
    for lbl in labels:
        counts[lbl] += 1
        total += 1
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return 1.0 - np.sum(probs ** 2)


def pheromone_decay(level: float, decay_rate: float = 0.1) -> float:
    """Exponential decay of a pheromone level."""
    if level < 0:
        raise ValueError("pheromone level must be non‑negative")
    return level * math.exp(-decay_rate)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
FeatureVec = Tuple[float, ...]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian (RBF) kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def calculate_similarity_matrix(points: List[FeatureVec]) -> np.ndarray:
    """Full pairwise similarity matrix using the Gaussian kernel."""
    n = len(points)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(points[i], points[j])
            sim = gaussian(d)
            S[i, j] = sim
            S[j, i] = sim
    np.fill_diagonal(S, 1.0)
    return S


def shannon_entropy(prob_dist: Iterable[float]) -> float:
    """Shannon entropy H = -∑ p log p."""
    probs = np.array(list(prob_dist))
    if np.any(probs < 0):
        raise ValueError("probabilities must be non‑negative")
    probs = probs / probs.sum()
    nonzero = probs[probs > 0]
    return -np.sum(nonzero * np.log2(nonzero))


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gini_gain: float
    weighted_confidence: float
    reason: str


def compute_weighted_confidence(
    labels: List[int],
    similarity_matrix: np.ndarray,
    sample_idx: int,
) -> float:
    """
    Combine Gini impurity and Shannon entropy, weighted by the average
    similarity of the sample to the rest of the batch.
    """
    # Gini impurity of the whole batch
    gini = gini_impurity(labels)

    # Class distribution for entropy
    total = len(labels)
    counts = Counter(labels)
    probs = np.array([counts[c] / total for c in counts])
    entropy = shannon_entropy(probs)

    # Average similarity of the target sample
    avg_sim = similarity_matrix[sample_idx].mean()

    # Confidence factor: higher similarity -> higher trust in the statistical
    # measures, lower entropy -> higher confidence.
    confidence = (1.0 - entropy) * avg_sim * (1.0 - gini)
    return confidence


def hybrid_split_decision(
    labels: List[int],
    features: List[FeatureVec],
    delta: float = 0.05,
    r_range: float = 1.0,
    pheromone: float = 1.0,
) -> SplitDecision:
    """
    Perform a Hoeffding‑tree style split test where the bound ε and the
    required Gini gain are modulated by similarity‑aware confidence.
    """
    n = len(labels)
    if n == 0:
        return SplitDecision(False, 0.0, 0.0, 0.0, "empty batch")

    # 1. similarity matrix (Parent B)
    S = calculate_similarity_matrix(features)

    # 2. Base Hoeffding bound (Parent A)
    eps = hoeffding_bound(r_range, delta, n)

    # 3. Gini impurity before split
    pre_impurity = gini_impurity(labels)

    # 4. Simulate a binary split by a random threshold on first feature
    threshold = np.median([f[0] for f in features])
    left_labels = [lbl for lbl, f in zip(labels, features) if f[0] <= threshold]
    right_labels = [lbl for lbl, f in zip(labels, features) if f[0] > threshold]

    # Gini impurity after split
    left_imp = gini_impurity(left_labels)
    right_imp = gini_impurity(right_labels)
    post_impurity = (len(left_labels) * left_imp + len(right_labels) * right_imp) / n
    gini_gain = pre_impurity - post_impurity

    # 5. Weighted confidence (bridge)
    # use the sample that is closest to the threshold as representative
    distances = [abs(f[0] - threshold) for f in features]
    rep_idx = int(np.argmin(distances))
    confidence = compute_weighted_confidence(labels, S, rep_idx)

    # 6. Adjust epsilon and required gain by confidence
    eps_adj = eps / (1.0 + confidence)          # higher confidence → tighter bound
    gain_req = eps_adj * confidence             # required gain scales with confidence

    # 7. Decision
    should = gini_gain > gain_req
    reason = (
        "split" if should else "no split"
    )
    # 8. Pheromone update (decay proportional to confidence)
    pheromone_new = pheromone_decay(pheromone, decay_rate=0.1 * (1.0 - confidence))

    return SplitDecision(
        should_split=should,
        epsilon=eps_adj,
        gini_gain=gini_gain,
        weighted_confidence=confidence,
        reason=reason,
    )


def synthetic_path_from_features(features: List[FeatureVec]) -> List[int]:
    """
    Construct a simple synthetic path over the feature set by repeatedly
    moving to the most similar unvisited node.  This mirrors the “path”
    concept of Parent B while providing an ordering useful for batch
    processing in the hybrid split test.
    """
    if not features:
        return []

    n = len(features)
    visited = [False] * n
    path = []

    # start from a random node
    current = random.randrange(n)
    path.append(current)
    visited[current] = True

    # pre‑compute similarity matrix for speed
    S = calculate_similarity_matrix(features)

    for _ in range(1, n):
        sims = S[current]
        # mask visited nodes
        sims = np.where(np.array(visited), -1.0, sims)
        next_idx = int(np.argmax(sims))
        visited[next_idx] = True
        path.append(next_idx)
        current = next_idx

    return path


if __name__ == "__main__":
    # Smoke test: generate a small random batch
    random.seed(42)
    batch_size = 20
    # binary class labels
    labels = [random.choice([0, 1]) for _ in range(batch_size)]
    # two‑dimensional feature vectors in [0, 1]
    features = [(random.random(), random.random()) for _ in range(batch_size)]

    decision = hybrid_split_decision(labels, features, delta=0.05, r_range=1.0, pheromone=1.0)
    print("Hybrid split decision:", decision)

    path = synthetic_path_from_features(features)
    print("Synthetic path (node order):", path)

    # Verify that similarity matrix is positive semi‑definite (basic check)
    S = calculate_similarity_matrix(features)
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-8), "Similarity matrix not PSD"
    print("Similarity matrix PSD check passed.")