# DARWIN HAMMER — match 13, survivor 7
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hoeffding bound utilities (parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Return the Hoeffding bound ε for range ``r``, confidence ``1‑δ`` and
    sample size ``n``.

    Raises:
        ValueError: if arguments are out of the admissible domain.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini split test."""
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


# ----------------------------------------------------------------------
# Gini utilities (parent B)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Compute the Gini inequality coefficient for a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def gini_impurity(labels: Iterable[int]) -> float:
    """Gini impurity of a categorical label distribution.

    ``labels`` can be any iterable of hashable class identifiers.
    """
    total = 0
    counts: Counter = Counter()
    for lbl in labels:
        counts[lbl] += 1
        total += 1
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return 1.0 - np.sum(probs ** 2)


def gini_gain(parent_labels: Iterable[int],
              left_labels: Iterable[int],
              right_labels: Iterable[int]) -> float:
    """Reduction in Gini impurity obtained by splitting ``parent`` into
    ``left`` and ``right``.
    """
    parent_imp = gini_impurity(parent_labels)
    n_parent = len(list(parent_labels))
    n_left = len(list(left_labels))
    n_right = len(list(right_labels))
    if n_parent == 0:
        return 0.0
    left_imp = gini_impurity(left_labels)
    right_imp = gini_impurity(right_labels)
    weighted_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
    return parent_imp - weighted_imp


# ----------------------------------------------------------------------
# Hybrid decision logic with improved tie handling
# ----------------------------------------------------------------------
def should_split_hybrid(gains: List[float],
                        delta: float,
                        n: int,
                        tie_threshold: float = 0.05,
                        min_gain: float = 0.0) -> SplitDecision:
    """Apply Hoeffding’s bound to a list of Gini gains.

    The maximal possible Gini gain for a binary classification problem is
    ``r = 0.5`` (impurity can drop from 0.5 to 0).  For multi‑class problems the
    bound is still safe because the impurity range never exceeds 0.5.

    Args:
        gains: List of candidate split gains (higher is better).
        delta: Desired probability of choosing the wrong split.
        n: Number of examples observed at the node.
        tie_threshold: If the bound ε falls below this value we accept the
            best split even if the gap is small.
        min_gain: Minimum gain required to consider a split.

    Returns:
        SplitDecision describing the outcome.
    """
    if not gains:
        return SplitDecision(False, 0.0, 0.0, "no_candidates")
    sorted_gains = sorted(gains, reverse=True)
    best_gain = sorted_gains[0]
    second_best_gain = sorted_gains[1] if len(sorted_gains) > 1 else 0.0
    if best_gain < min_gain:
        return SplitDecision(False, 0.0, 0.0, "gain_below_min")
    # Range of Gini gain (max possible reduction)
    r = 0.5
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Simple streaming node that uses the hybrid rule with buffering
# ----------------------------------------------------------------------
class StreamingNode:
    """A minimal streaming decision node that tracks class counts and
    evaluates binary feature splits using the hybrid Hoeffding‑Gini rule.
    """

    def __init__(self, delta: float = 1e-7, tie_threshold: float = 0.05, buffer_size: int = 100):
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.buffer_size = buffer_size
        self.n_examples = 0
        self.class_counts: Counter[int] = Counter()
        self.buffer: List[Tuple[Dict[int, int], int]] = []
        # For each binary feature we keep separate class count tables:
        # feature_id -> (left_counts, right_counts)
        self.feature_stats: Dict[int, Tuple[Counter[int], Counter[int]]] = {}

    def update(self, features: Dict[int, int], label: int) -> None:
        """Ingest a single instance.

        ``features`` maps feature ids to binary values (0/1).
        """
        self.n_examples += 1
        self.class_counts[label] += 1
        self.buffer.append((features, label))
        if len(self.buffer) > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        for features, label in self.buffer:
            for fid, val in features.items():
                left, right = self.feature_stats.setdefault(fid, (Counter(), Counter()))
                if val == 0:
                    left[label] += 1
                else:
                    right[label] += 1
        self.buffer.clear()

    def evaluate_splits(self) -> SplitDecision:
        """Compute Gini gains for all observed binary features and decide
        whether to split the node.
        """
        if self.n_examples < 2:
            return SplitDecision(False, 0.0, 0.0, "insufficient_examples")
        self._flush_buffer()
        parent_labels = list(self.class_counts.elements())
        gains = []
        for left_counts, right_counts in self.feature_stats.values():
            left_labels = list(left_counts.elements())
            right_labels = list(right_counts.elements())
            gain = gini_gain(parent_labels, left_labels, right_labels)
            gains.append(gain)
        return should_split_hybrid(gains, self.delta, self.n_examples, self.tie_threshold)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _synthetic_stream(num_instances: int = 1000,
                      num_features: int = 5,
                      num_classes: int = 2) -> Tuple[Dict[int, int], int]:
    """Yield random binary feature dictionaries and class labels."""
    for _ in range(num_instances):
        features = {i: random.randint(0, 1) for i in range(num_features)}
        label = random.randint(0, num_classes - 1)
        yield features, label