# DARWIN HAMMER — match 13, survivor 6
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

"""Hybrid Hoeffding‑Gini module.

Both parent algorithms address statistical decisions on streaming data.
* `hoeffding_tree.py` supplies a concentration bound  ε = √(r²·ln(1/δ)/(2·n)) that
  quantifies how many samples n are required to distinguish two statistics
  (e.g. split gains) with confidence 1‑δ.
* `gini_coefficient.py` computes the Gini impurity of a class‑frequency vector,
  a standard split quality measure in decision‑tree learning.

The mathematical bridge is that a split decision can be based on **Gini gain**
(the reduction of impurity) while the Hoeffding bound can be used to decide
whether the observed gain difference between the best and the second‑best
candidate is statistically significant.  The range *r* required by the bound
is the maximum possible Gini gain for a k‑class problem, which is
`1 - 1/k`.  By feeding the Gini‑based gains into the Hoeffding test we obtain
a single unified criterion for online tree growth.

The module therefore provides:
* Gini impurity / gain utilities.
* Hoeffding‑bound helpers (unchanged).
* A hybrid split‑decision routine that evaluates candidate attributes with
  Gini gain and applies the Hoeffding bound to the gain gap.

"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Hoeffding helpers
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range *r*, confidence *δ* and sample count *n*."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑based split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(
    best_gain: float,
    second_best_gain: float,
    r: float,
    delta: float,
    n: int,
    tie_threshold: float = 0.05,
) -> SplitDecision:
    """Standard Hoeffding split test."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = (
        "gap_exceeds_bound"
        if gap > eps
        else ("tie_threshold" if eps < tie_threshold else "wait")
    )
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Parent B – Gini helpers
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini inequality coefficient of a non‑negative value list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gini_impurity(class_counts: Iterable[int]) -> float:
    """Gini impurity for a discrete class distribution."""
    counts = np.array(list(class_counts), dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return 1.0 - np.sum(probs ** 2)


def gini_gain(parent_counts: Iterable[int],
              left_counts: Iterable[int],
              right_counts: Iterable[int]) -> float:
    """Reduction in Gini impurity obtained by splitting parent into left/right."""
    parent_imp = gini_impurity(parent_counts)
    n_parent = sum(parent_counts)
    n_left = sum(left_counts)
    n_right = sum(right_counts)
    if n_parent == 0:
        return 0.0
    left_imp = gini_impurity(left_counts)
    right_imp = gini_impurity(right_counts)
    weighted_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
    return parent_imp - weighted_imp


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_attribute_gini_gains(
    parent_counts: List[int],
    splits: Dict[str, Tuple[List[int], List[int]]],
) -> Dict[str, float]:
    """
    For each attribute (key) compute the Gini gain of the proposed split.

    Parameters
    ----------
    parent_counts : list[int]
        Class frequencies before splitting.
    splits : dict
        Mapping ``attribute_name -> (left_counts, right_counts)`` where each
        count list corresponds to the class distribution in the child node.

    Returns
    -------
    dict
        ``attribute_name -> gini_gain``.
    """
    gains = {}
    for attr, (left, right) in splits.items():
        gains[attr] = gini_gain(parent_counts, left, right)
    return gains


def hybrid_split_decision(
    parent_counts: List[int],
    splits: Dict[str, Tuple[List[int], List[int]]],
    delta: float = 1e-7,
    n_samples: int = 1,
    tie_threshold: float = 0.05,
) -> Tuple[SplitDecision, str]:
    """
    Evaluate candidate attribute splits with Gini gain and decide via Hoeffding bound.

    Returns a ``SplitDecision`` and the name of the chosen attribute (or empty
    string if no split is performed).
    """
    if not splits:
        return SplitDecision(False, 0.0, 0.0, "no_candidates"), ""

    gains = compute_attribute_gini_gains(parent_counts, splits)
    # Sort attributes by gain descending
    sorted_attrs = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)

    best_attr, best_gain = sorted_attrs[0]
    second_gain = sorted_attrs[1][1] if len(sorted_attrs) > 1 else 0.0

    # Maximum possible Gini gain for k classes:
    k = len(parent_counts)
    max_gain = 1.0 - 1.0 / k if k > 0 else 1.0

    decision = should_split(
        best_gain,
        second_gain,
        r=max_gain,
        delta=delta,
        n=n_samples,
        tie_threshold=tie_threshold,
    )
    chosen_attr = best_attr if decision.should_split else ""
    return decision, chosen_attr


def gini_hoeffding_score(
    parent_counts: List[int],
    left_counts: List[int],
    right_counts: List[int],
    delta: float = 1e-7,
    n_samples: int = 1,
) -> Tuple[float, float]:
    """
    Return a tuple ``(gain, epsilon)`` for a single split using the hybrid metric.

    This helper is convenient when the caller wants the raw gain and the
    corresponding Hoeffding bound without performing the full decision logic.
    """
    gain = gini_gain(parent_counts, left_counts, right_counts)
    k = len(parent_counts)
    max_gain = 1.0 - 1.0 / k if k > 0 else 1.0
    epsilon = hoeffding_bound(max_gain, delta, n_samples)
    return gain, epsilon


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a binary classification problem (2 classes)
    random.seed(42)
    np.random.seed(42)

    # Parent node class counts (e.g., 60% class 0, 40% class 1)
    parent = [60, 40]

    # Generate three candidate attributes with random split distributions
    candidate_splits = {}
    for attr_id in range(3):
        # Randomly allocate each class count to left/right child
        left = [
            random.randint(0, parent[0]),
            random.randint(0, parent[1]),
        ]
        # Ensure totals match the parent
        right = [parent[0] - left[0], parent[1] - left[1]]
        candidate_splits[f"attr_{attr_id}"] = (left, right)

    # Show generated splits
    print("Candidate splits (left_counts, right_counts):")
    for name, (l, r) in candidate_splits.items():
        print(f"  {name}: left={l}, right={r}")

    # Perform hybrid decision
    decision, chosen = hybrid_split_decision(
        parent_counts=parent,
        splits=candidate_splits,
        delta=1e-5,
        n_samples=sum(parent),  # total observations seen so far
        tie_threshold=0.01,
    )

    print("\nHybrid split decision:")
    print(f"  Should split?  {decision.should_split}")
    print(f"  Reason:       {decision.reason}")
    print(f"  Gain gap:     {decision.gain_gap:.6f}")
    print(f"  Epsilon:      {decision.epsilon:.6f}")
    if decision.should_split:
        print(f"  Chosen attribute: {chosen}")

    # Demonstrate the low‑level helper
    sample_attr = list(candidate_splits.values())[0]
    gain, eps = gini_hoeffding_score(parent, *sample_attr)
    print(f"\nSample attribute gain = {gain:.6f}, Hoeffding ε = {eps:.6f}")

    sys.exit(0)