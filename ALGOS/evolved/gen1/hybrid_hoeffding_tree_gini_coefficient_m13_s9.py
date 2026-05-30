# DARWIN HAMMER — match 13, survivor 9
# gen: 1
# parent_a: hoeffding_tree.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:25:17Z

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Hoeffding utilities (parent A) – now adaptive to the statistic range
# ----------------------------------------------------------------------
def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) ).

    Args:
        range_: The known range R of the bounded random variable (max - min).
        delta: Desired error probability (0 < δ < 1).
        n: Number of independent observations.

    Returns:
        The confidence radius ε.
    """
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))


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
def gini_impurity_from_counts(counts: Counter) -> float:
    """Gini impurity given a Counter of class frequencies."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)


def gini_gain(parent_counts: Counter,
              left_counts: Counter,
              right_counts: Counter) -> float:
    """Reduction in Gini impurity obtained by splitting ``parent`` into left/right.

    This version works directly with Counters to avoid materialising label lists.
    """
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0

    parent_imp = gini_impurity_from_counts(parent_counts)
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)

    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())

    weighted_imp = (n_left / n_parent) * left_imp + (n_right / n_parent) * right_imp
    return parent_imp - weighted_imp


# ----------------------------------------------------------------------
# Hybrid decision logic – deeper integration
# ----------------------------------------------------------------------
def should_split_hybrid(gains: List[float],
                        parent_impurity: float,
                        delta: float,
                        n: int,
                        tie_threshold: float = 0.05) -> SplitDecision:
    """Apply Hoeffding’s bound to a list of Gini gains with an adaptive range.

    The statistic we bound is the *gain difference* between the best and the
    second‑best split. Its range is at most the maximal possible gain,
    which equals the parent impurity (the impurity can drop from
    ``parent_impurity`` to 0).

    Additionally, we prune hopeless candidates early: any split whose
    upper confidence bound (gain + ε) is lower than the current best gain
    is discarded from future consideration.

    Args:
        gains: List of candidate split gains (higher is better).
        parent_impurity: Impurity of the node before splitting.
        delta: Desired error probability.
        n: Number of examples observed at the node.
        tie_threshold: Accept the best split if ε falls below this value.

    Returns:
        SplitDecision describing the outcome.
    """
    if not gains:
        return SplitDecision(False, 0.0, 0.0, "no_candidates")

    # Sort descending while keeping original order for possible tie‑breaking
    sorted_gains = sorted(gains, reverse=True)
    best_gain = sorted_gains[0]
    second_best_gain = sorted_gains[1] if len(sorted_gains) > 1 else 0.0

    # Adaptive range: maximal possible gain = parent_impurity
    range_gain = parent_impurity
    eps = hoeffding_bound(range_gain, delta, n)

    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = "gap_exceeds_bound" if gap > eps else (
        "tie_threshold" if eps < tie_threshold else "wait"
    )
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Streaming node with adaptive Hoeffding‑Gini integration
# ----------------------------------------------------------------------
class StreamingNode:
    """A minimal streaming decision node that tracks class counts and
    evaluates binary feature splits using an adaptive Hoeffding‑Gini rule.
    """

    def __init__(self, delta: float = 1e-7, tie_threshold: float = 0.05):
        self.delta = delta
        self.tie_threshold = tie_threshold
        self.n_examples = 0
        self.class_counts: Counter[int] = Counter()
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
        whether to split the node using the adaptive bound.
        """
        if self.n_examples < 2:
            return SplitDecision(False, 0.0, 0.0, "insufficient_examples")

        parent_impurity = gini_impurity_from_counts(self.class_counts)
        gains = []
        for left_counts, right_counts in self.feature_stats.values():
            gain = gini_gain(self.class_counts, left_counts, right_counts)
            gains.append(gain)

        return should_split_hybrid(
            gains,
            parent_impurity,
            self.delta,
            self.n_examples,
            self.tie_threshold,
        )


# ----------------------------------------------------------------------
# Simple synthetic stream for demonstration / testing
# ----------------------------------------------------------------------
def _synthetic_stream(num_instances: int = 1000,
                      num_features: int = 5,
                      num_classes: int = 2) -> Iterable[Tuple[Dict[int, int], int]]:
    """Yield random binary feature dictionaries and class labels."""
    for _ in range(num_instances):
        features = {fid: random.randint(0, 1) for fid in range(num_features)}
        label = random.randint(0, num_classes - 1)
        yield features, label


def demo() -> None:
    node = StreamingNode()
    for feats, lbl in _synthetic_stream():
        node.update(feats, lbl)
        decision = node.evaluate_splits()
        if decision.should_split:
            print(
                f"Split after {node.n_examples} examples: "
                f"gain_gap={decision.gain_gap:.4f}, ε={decision.epsilon:.4f}, reason={decision.reason}"
            )
            break


if __name__ == "__main__":
    demo()