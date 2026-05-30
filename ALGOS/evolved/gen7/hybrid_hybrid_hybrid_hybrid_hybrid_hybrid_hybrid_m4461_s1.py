# DARWIN HAMMER — match 4461, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2526_s0.py (gen6)
# born: 2026-05-29T23:55:55Z

"""Hybrid RBF‑Bandit Hoeffding Tree
=================================

This module fuses **Parent Algorithm A** (RBF surrogate + Hoeffding‑tree split logic)
with **Parent Algorithm B** (contextual bandit actions providing *propensity*,
*expected reward* and *confidence bound*).

Mathematical bridge
------------------
* The RBF surrogate predicts a real‑valued response by a weighted sum of Gaussian
  kernels.  In the hybrid we **modulate each kernel weight by the bandit
  propensity**, i.e. `w_i ← w_i * propensity`.  This makes the surrogate
  *probabilistically* biased towards actions that the bandit deems likely.
* The Hoeffding‑tree split criterion (Gini gain vs. Hoeffding bound) uses a
  confidence parameter `δ`.  We replace `δ` with a *bandit‑adjusted* value
  `δ' = δ * (1‑confidence_bound)`, shrinking the bound when the bandit is
  confident and allowing earlier splits.

The resulting hybrid can be trained on a data stream while simultaneously
receiving a bandit action for each example, producing a tree whose leaves host
bandit‑weighted RBF surrogates.

Only the Python standard library and NumPy are used.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b via Gauss‑Jordan elimination (no external libs)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

# ----------------------------------------------------------------------
# Bandit structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # ∈ (0,1]
    expected_reward: float     # arbitrary scale
    confidence_bound: float    # ∈ (0,1)
    algorithm: str

# ----------------------------------------------------------------------
# Hoeffding‑tree utilities (adapted from Parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound used for split decisions."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

def gini_impurity_from_counts(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)

def gini_gain(parent_counts: Dict[int, int],
              left_counts:  Dict[int, int],
              right_counts: Dict[int, int]) -> float:
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0
    parent_imp = gini_impurity_from_counts(parent_counts)
    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)
    weighted_imp = (n_left * left_imp + n_right * right_imp) / n_parent
    return parent_imp - weighted_imp

# ----------------------------------------------------------------------
# Hybrid node definition
# ----------------------------------------------------------------------
@dataclass
class HoeffdingRBFNode:
    depth: int
    max_depth: int
    min_samples_split: int
    epsilon: float = 1.0                     # RBF shape parameter
    counts: Dict[int, int] = None            # class counts {0:...,1:...}
    data: List[Tuple[Vector, int]] = None    # stored (x, y) pairs
    left: 'HoeffdingRBFNode' = None
    right: 'HoeffdingRBFNode' = None
    split_feature: int = None
    split_threshold: float = None
    surrogate_weights: List[float] = None    # fitted RBF weights
    surrogate_centers: List[Tuple[float, ...]] = None

    def __post_init__(self):
        self.counts = {0: 0, 1: 0}
        self.data = []

    # ------------------------------------------------------------------
    # Leaf‑side operations
    # ------------------------------------------------------------------
    def add_example(self, x: Vector, y: int) -> None:
        """Record a training example and update class counts."""
        self.counts[y] = self.counts.get(y, 0) + 1
        self.data.append((tuple(x), y))

    def maybe_fit_surrogate(self) -> None:
        """Fit an RBF surrogate if enough data is present."""
        if len(self.data) < self.min_samples_split:
            return
        centers = [c for c, _ in self.data]
        targets = [float(y) for _, y in self.data]
        n = len(centers)
        # Build kernel matrix K_ij = gaussian(||c_i-c_j||, epsilon)
        K = [[gaussian(euclidean(ci, cj), self.epsilon) for cj in centers] for ci in centers]
        try:
            w = solve_linear(K, targets)
        except ValueError:
            # fallback to simple averaging if singular
            w = [np.mean(targets)] * n
        self.surrogate_weights = w
        self.surrogate_centers = centers

    def bandit_weighted_rbf_predict(self, x: Vector,
                                    bandit: BanditAction) -> float:
        """Predict using the leaf’s surrogate, scaling each weight by bandit propensity."""
        if self.surrogate_weights is None or self.surrogate_centers is None:
            # no surrogate yet – fall back to class majority probability
            total = sum(self.counts.values())
            if total == 0:
                return 0.5
            prob_one = self.counts[1] / total
            return prob_one * bandit.propensity
        # weighted kernel sum
        weighted_sum = 0.0
        for w, c in zip(self.surrogate_weights, self.surrogate_centers):
            weighted_sum += (w * bandit.propensity *
                             gaussian(euclidean(x, c), self.epsilon))
        return weighted_sum

    # ------------------------------------------------------------------
    # Split logic that incorporates bandit confidence
    # ------------------------------------------------------------------
    def evaluate_split(self, bandit: BanditAction,
                       delta: float = 0.05,
                       range_: float = 1.0) -> Tuple[bool, str]:
        """
        Decide whether to split this leaf.
        The bandit confidence bound tightens the Hoeffding bound:
            δ' = δ * (1 - confidence_bound)
        """
        if self.depth >= self.max_depth or len(self.data) < self.min_samples_split:
            return False, "insufficient depth or samples"

        # Simple split on the first feature using median of observed values
        feature_vals = [x[self.split_feature] if self.split_feature is not None else x[0]
                        for x, _ in self.data]
        threshold = float(np.median(feature_vals))
        left_counts = {0: 0, 1: 0}
        right_counts = {0: 0, 1: 0}
        for (x, y) in self.data:
            if x[self.split_feature if self.split_feature is not None else 0] <= threshold:
                left_counts[y] += 1
            else:
                right_counts[y] += 1

        gain = gini_gain(self.counts, left_counts, right_counts)
        # Adjust delta with bandit confidence
        adj_delta = delta * (1.0 - bandit.confidence_bound)
        bound = hoeffding_bound(range_, adj_delta, len(self.data))

        should = gain > bound
        reason = f"gain={gain:.4f}, bound={bound:.4f}, adj_delta={adj_delta:.4f}"
        return should, reason

    def split(self, bandit: BanditAction) -> None:
        """Perform a binary split on the first feature."""
        # Choose split feature (first for simplicity) and median threshold
        self.split_feature = 0
        feature_vals = [x[self.split_feature] for x, _ in self.data]
        self.split_threshold = float(np.median(feature_vals))

        left_node = HoeffdingRBFNode(depth=self.depth + 1,
                                    max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    epsilon=self.epsilon)
        right_node = HoeffdingRBFNode(depth=self.depth + 1,
                                     max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     epsilon=self.epsilon)

        for (x, y) in self.data:
            if x[self.split_feature] <= self.split_threshold:
                left_node.add_example(x, y)
            else:
                right_node.add_example(x, y)

        # Fit surrogates for children (optional)
        left_node.maybe_fit_surrogate()
        right_node.maybe_fit_surrogate()

        self.left = left_node
        self.right = right_node
        # Clear leaf‑specific data to free memory
        self.data = []
        self.surrogate_weights = None
        self.surrogate_centers = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route(self, x: Vector) -> 'HoeffdingRBFNode':
        """Descend the tree until a leaf is reached."""
        node = self
        while node.left is not None and node.right is not None:
            if x[node.split_feature] <= node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node

# ----------------------------------------------------------------------
# Public hybrid API
# ----------------------------------------------------------------------
def bandit_weighted_rbf_predict(root: HoeffdingRBFNode,
                               x: Vector,
                               bandit: BanditAction) -> float:
    """Top‑level prediction: route to leaf and use bandit‑weighted surrogate."""
    leaf = root.route(x)
    return leaf.bandit_weighted_rbf_predict(x, bandit)

def train_hybrid_stream(stream: List[Tuple[Vector, int]],
                        bandit_actions: List[BanditAction],
                        max_depth: int = 5,
                        min_samples_split: int = 10,
                        epsilon: float = 1.0) -> HoeffdingRBFNode:
    """
    Incrementally train the hybrid tree on a finite stream.
    `stream` and `bandit_actions` must be of equal length.
    """
    if len(stream) != len(bandit_actions):
        raise ValueError("stream and bandit_actions must match in length")

    root = HoeffdingRBFNode(depth=0,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            epsilon=epsilon)

    for (x, y), bandit in zip(stream, bandit_actions):
        leaf = root.route(x)
        leaf.add_example(x, y)
        leaf.maybe_fit_surrogate()
        should, reason = leaf.evaluate_split(bandit)
        if should:
            leaf.split(bandit)
    return root

def simulate_bandit(x: Vector) -> BanditAction:
    """
    Dummy bandit that returns a propensity proportional to the norm of x
    and a confidence that grows with the norm.
    """
    norm = euclidean(x, [0.0] * len(x))
    propensity = min(1.0, 0.1 + 0.9 * (norm / (norm + 1.0)))
    expected_reward = propensity * 2.0
    confidence = min(0.99, norm / (norm + 5.0))
    return BanditAction(action_id="dummy",
                        propensity=propensity,
                        expected_reward=expected_reward,
                        confidence_bound=confidence,
                        algorithm="dummy_bandit")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Generate a simple binary classification stream in 2‑D
    def gen_point():
        label = random.choice([0, 1])
        if label == 0:
            x = np.random.normal(loc=-2.0, scale=1.0, size=2)
        else:
            x = np.random.normal(loc=2.0, scale=1.0, size=2)
        return (x.tolist(), label)

    stream = [gen_point() for _ in range(200)]
    bandits = [simulate_bandit(x) for x, _ in stream]

    root = train_hybrid_stream(stream, bandits,
                               max_depth=4,
                               min_samples_split=15,
                               epsilon=0.8)

    # Test predictions on a few fresh points
    test_points = [
        ([-3, -3], simulate_bandit([-3, -3])),
        ([3, 3], simulate_bandit([3, 3])),
        ([0, 0], simulate_bandit([0, 0])),
    ]

    for pt, ba in test_points:
        pred = bandit_weighted_rbf_predict(root, pt, ba)
        print(f"Input {pt} → prediction {pred:.4f} (propensity {ba.propensity:.2f})")