# DARWIN HAMMER — match 2062, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_label_foundry_m1042_s0.py (gen4)
# born: 2026-05-29T23:40:46Z

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, List, Tuple

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass
class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    failure_threshold: int = 3
    failures: int = 0
    open: bool = False
    last_event_at: str = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


def calculate_health_score(morphology: Morphology) -> float:
    """Health score based solely on shape (parent A)."""
    return sphericity_index(morphology.length, morphology.width, morphology.height)


def calculate_entropy(feature_vector: List[float]) -> float:
    """Shannon entropy of a normalized feature vector."""
    total = sum(feature_vector)
    if total == 0:
        return 0.0
    probs = [f / total for f in feature_vector if f > 0]
    return -sum(p * math.log(p, 2) for p in probs)


# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # 0.0 .. 1.0


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


# ----------------------------------------------------------------------
# Core hybrid functions (mathematical fusion)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], weights: List[float],
                     width: int = 64, depth: int = 4) -> List[List[float]]:
    """Weighted Count‑Min sketch. Each cell accumulates the sum of weights."""
    table = [[0.0] * width for _ in range(depth)]
    for item, weight in zip(items, weights):
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += weight
    return table


def tropical_max_plus(matrix: List[List[float]]) -> List[float]:
    """
    Apply tropical (max‑plus) algebra to collapse a 2‑D sketch into a 1‑D score vector.
    For each column we compute max over rows (tropical addition) and then add the column index
    (tropical multiplication) to obtain a convex piece‑wise linear score.
    """
    depth = len(matrix)
    if depth == 0:
        return []
    width = len(matrix[0])
    scores = []
    for col in range(width):
        col_max = max(matrix[row][col] for row in range(depth))
        # tropical multiplication: add column index (acts like a linear term)
        scores.append(col_max + col)
    return scores


def compute_hybrid_sketch(morphologies: List[Morphology],
                          labels: List[ProbabilisticLabel],
                          width: int = 64,
                          depth: int = 4) -> Tuple[List[List[float]], List[float]]:
    """
    Fuse Parent A's health scores with Parent B's weighted sketch.

    Returns:
        sketch: the raw Count‑Min table
        scores: tropical max‑plus scores derived from the sketch
    """
    if len(morphologies) != len(labels):
        raise ValueError("Morphology list and label list must be of equal length")
    # health‑aware weight for each item
    weights = [
        calculate_health_score(morph) * max(0.0, min(1.0, lbl.confidence))
        for morph, lbl in zip(morphologies, labels)
    ]
    items = [lbl.doc_id for lbl in labels]
    sketch = count_min_sketch(items, weights, width=width, depth=depth)
    scores = tropical_max_plus(sketch)
    return sketch, scores


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound used for split confidence (Parent B)."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid parameters for Hoeffding bound")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def assess_split(scores: List[float],
                r: float,
                delta: float,
                n: int,
                tie_threshold: float = 0.05) -> SplitDecision:
    """
    Decide whether to split based on the tropical scores.
    The best_gain is the maximum score, second_best_gain the runner‑up.
    """
    if not scores:
        return SplitDecision(False, 0.0, 0.0, "empty score vector")
    sorted_scores = sorted(scores, reverse=True)
    best_gain = sorted_scores[0]
    second_best_gain = sorted_scores[1] if len(sorted_scores) > 1 else best_gain
    epsilon = hoeffding_bound(r, delta, n)

    gain_gap = best_gain - second_best_gain
    should_split = gain_gap > epsilon and gain_gap > tie_threshold * best_gain
    reason = ("gap exceeds Hoeffding bound" if should_split else "gap too small")
    return SplitDecision(should_split, epsilon, gain_gap, reason)


def update_circuit_breaker(cb: EndpointCircuitBreaker,
                           success: bool,
                           health_score: float,
                           threshold: float = 0.6) -> None:
    """
    Integrate health information into the circuit breaker.
    If health falls below *threshold* we treat it as a failure.
    """
    if not success or health_score < threshold:
        cb.record_failure()
    else:
        cb.record_success()


def robust_compute_hybrid_sketch(morphologies: List[Morphology],
                                 labels: List[ProbabilisticLabel],
                                 width: int = 64,
                                 depth: int = 4,
                                 epsilon: float = 1e-6) -> Tuple[List[List[float]], List[float]]:
    """
    Improved version of compute_hybrid_sketch with more robust handling of edge cases.

    Returns:
        sketch: the raw Count‑Min table
        scores: tropical max‑plus scores derived from the sketch
    """
    if len(morphologies) != len(labels):
        raise ValueError("Morphology list and label list must be of equal length")
    # health‑aware weight for each item
    weights = [
        max(epsilon, calculate_health_score(morph) * max(epsilon, min(1.0 - epsilon, lbl.confidence)))
        for morph, lbl in zip(morphologies, labels)
    ]
    items = [lbl.doc_id for lbl in labels]
    sketch = count_min_sketch(items, weights, width=width, depth=depth)
    scores = tropical_max_plus(sketch)
    return sketch, scores


def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    Compute the KL divergence between two probability distributions.

    Args:
    p (List[float]): The first probability distribution.
    q (List[float]): The second probability distribution.

    Returns:
    float: The KL divergence between p and q.
    """
    kl_div = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            kl_div += p[i] * math.log(p[i] / q[i])
    return kl_div


def kullback_leibler_test(p: List[float], q: List[float], threshold: float = 1e-3) -> bool:
    """
    Perform a Kullback-Leibler divergence test to compare two probability distributions.

    Args:
    p (List[float]): The first probability distribution.
    q (List[float]): The second probability distribution.

    Returns:
    bool: True if the KL divergence is below the threshold, False otherwise.
    """
    kl_div = kl_divergence(p, q)
    return kl_div < threshold


if __name__ == "__main__":
    # Create synthetic morphologies
    morphs = [
        Morphology(1.0, 2.0, 3.0, 10.0),
        Morphology(4.0, 5.0, 6.0, 20.0),
        Morphology(7.0, 8.0, 9.0, 30.0),
    ]

    labels = [
        ProbabilisticLabel("doc1", 1, 0.8),
        ProbabilisticLabel("doc2", 2, 0.4),
        ProbabilisticLabel("doc3", 1, 0.9),
    ]

    sketch, scores = robust_compute_hybrid_sketch(morphs, labels)
    print("Hybrid Sketch:")
    for row in sketch:
        print(row)
    print("Tropical Max-Plus Scores:")
    print(scores)

    # Test Kullback-Leibler divergence
    p = [0.2, 0.3, 0.5]
    q = [0.1, 0.4, 0.5]
    kl_div = kl_divergence(p, q)
    print("KL Divergence:", kl_div)

    # Perform Kullback-Leibler test
    test_result = kullback_leibler_test(p, q)
    print("Kullback-Leibler Test:", test_result)