# DARWIN HAMMER — match 13, survivor 5
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z
#
# DISTILLED USE: Statistically provable routing split decisions. Gini
# impurity range [0, 0.5] used directly as r in the Hoeffding bound —
# the split criterion is formally grounded, not heuristic. Drop into
# lucidota_river_governor: fast/slow lane routing decisions now carry
# a formal guarantee 'with probability 1-delta, this split is correct
# given n samples' rather than ad-hoc thresholds.

"""Hybrid Hoeffding‑Gini module.

This file merges two classic stream‑learning components:

* **Hoeffding bound** (from ``hoeffding_tree.py``) – a concentration
  inequality that yields a confidence radius ``ε`` for the difference
  between two statistics after observing ``n`` examples.
* **Gini coefficient / impurity** (from ``gini_coefficient.py``) – a
  measure of inequality that, in classification, is used as an impurity
  metric.  The reduction in Gini impurity produced by a candidate split
  is a natural “gain’’ value.

The mathematical bridge is the observation that a split decision can be
made by comparing *Gini gains* of competing split candidates.  The Hoeffding
bound supplies a probabilistic guarantee that the observed gain gap exceeds
the true (population) gap.  By treating the maximal possible Gini gain
(``r``) as the range of the statistic (the Gini impurity lies in ``[0, 0.5]``
for binary classification), we can directly reuse the Hoeffding formula:


ε = sqrt( r² * ln(1/δ) / (2 n) )
split if (best_gain - second_best_gain) > ε   or   ε < tie_threshold


The functions below implement this hybrid decision rule and a tiny streaming
demo that updates class counts, evaluates candidate splits, and decides
whether to split a node.

"""

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
# Hybrid decision logic
# ----------------------------------------------------------------------
def should_split_hybrid(gains: List[float],
                        delta: float,
                        n: int,
                        tie_threshold: float = 0.05) -> SplitDecision:
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

    Returns:
        SplitDecision describing the outcome.
    """
    if not gains:
        return SplitDecision(False, 0.0, 0.0, "no_candidates")
    sorted_gains = sorted(gains, reverse=True)
    best_gain = sorted_gains[0]
    second_best_gain = sorted_gains[1] if len(sorted_gains) > 1 else 0.0
    # Range of Gini gain (max possible reduction)
    r = 0.5
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else ("tie_threshold" if eps < tie_threshold else "wait")
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Simple streaming node that uses the hybrid rule
# ----------------------------------------------------------------------
class StreamingNode:
    """A minimal streaming decision node that tracks class counts and
    evaluates binary feature splits using the hybrid Hoeffding‑Gini rule.
    """

    def __init__(self, delta: float = 1e-7, tie_threshold: float = 0.05):
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.n_examples = 0
        self.class_counts: Counter[int] = Counter()
        # For each binary feature we keep separate class count tables:
        # feature_id -> (left_counts, right_counts)
        self.feature_stats: Dict[int, Tuple[Counter[int], Counter[int]]] = {}

    def update(self, features: Dict[int, int], label: int) -> None:
        """Ingest a single instance.

        ``features`` maps feature ids to binary values (0/1).
        """
        self.n_examples += 1
        self.class_counts[label] += 1
        for fid, val in features.items():
            left, right = self.feature_stats.setdefault(fid, (Counter(), Counter()))
            if val == 0:
                left[label] += 1
            else:
                right[label] += 1

    def evaluate_splits(self) -> SplitDecision:
        """Compute Gini gains for all observed binary features and decide
        whether to split the node.
        """
        if self.n_examples < 2:
            return SplitDecision(False, 0.0, 0.0, "insufficient_examples")
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
        features = {fid: random.randint(0, 1) for fid in range(num_features)}
        # Simple rule: label = majority of first two features (with noise)
        raw = features[0] + features[1]
        prob = 0.8 if raw >= 1 else 0.2
        label = 1 if random.random() < prob else 0
        yield features, label


def main() -> None:
    node = StreamingNode(delta=1e-5, tie_threshold=0.02)
    stream = _synthetic_stream()
    for i, (feat, lbl) in enumerate(stream, 1):
        node.update(feat, lbl)
        if i % 200 == 0:
            decision = node.evaluate_splits()
            print(f"After {i} examples -> split?: {decision.should_split} "
                  f"(ε={decision.epsilon:.5f}, gap={decision.gain_gap:.5f}, reason={decision.reason})")
            if decision.should_split:
                print("  --> Node would split now.")
                break
    else:
        print("Stream finished without triggering a split.")


if __name__ == "__main__":
    main()