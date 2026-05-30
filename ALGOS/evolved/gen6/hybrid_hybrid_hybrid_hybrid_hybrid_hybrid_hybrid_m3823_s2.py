# DARWIN HAMMER — match 3823, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m958_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_hybrid_hybrid_m2033_s0.py (gen5)
# born: 2026-05-29T23:51:49Z

"""
Hybrid Algorithm Fusion: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m958_s0.py &
hybrid_hybrid_hybrid_liquid_hybrid_hybrid_hybrid_m2033_s0.py

Mathematical Bridge
-------------------
Parent A provides *Count‑Min* and *HyperLogLog* sketches that give, respectively,
an approximate log‑likelihood contribution of a reward sequence and an estimate of the
number of distinct contexts observed.  Parent B introduces a *Diffusion Forcing* noise
schedule that can be shaped by a scalar *Gini coefficient* derived from data
inequality.

The fusion exploits the fact that the Gini coefficient can be computed directly
from the frequency distribution stored in a Count‑Min sketch.  By feeding this
Gini value into the diffusion‑forcing schedule we obtain a noise schedule that
adapts to the observed inequality of context frequencies.  The estimated distinct
contexts from HyperLogLog are used as a regulariser for Hoeffding‑tree split
decisions, biasing splits toward features that increase the diversity of observed
contexts.

The resulting hybrid system therefore consists of:
1. Sketch‑based statistics (Count‑Min & HyperLogLog).
2. Gini‑driven diffusion forcing noise schedule.
3. Hoeffding‑tree split evaluation that incorporates distinct‑context estimates.

All three components are combined in a lightweight online learner that can be
used for contextual bandit or streaming classification tasks.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Sketch utilities (Count‑Min & HyperLogLog)
# ----------------------------------------------------------------------
def _hash(item: str, seed: int) -> int:
    """Deterministic 64‑bit hash for sketching."""
    data = seed.to_bytes(4, "big") + item.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def init_count_min(width: int = 2000, depth: int = 5) -> List[List[int]]:
    """Create an empty Count‑Min sketch matrix."""
    return [[0] * width for _ in range(depth)]


def update_count_min(sketch: List[List[int]], item: str, increment: int = 1) -> None:
    """Update Count‑Min sketch with *item*."""
    for row, seed in enumerate(range(len(sketch))):
        col = _hash(item, seed) % len(sketch[0])
        sketch[row][col] += increment


def query_count_min(sketch: List[List[int]], item: str) -> int:
    """Estimate frequency of *item* (minimum over rows)."""
    estimates = []
    for row, seed in enumerate(range(len(sketch))):
        col = _hash(item, seed) % len(sketch[0])
        estimates.append(sketch[row][col])
    return min(estimates)


def init_hyperloglog(p: int = 12) -> Tuple[int, List[int]]:
    """
    Very small HyperLogLog estimator.
    Returns (register_bits, registers) where register_bits = p.
    """
    m = 1 << p
    return (p, [0] * m)


def _rho(w: int, max_bits: int) -> int:
    """Position of first 1-bit (1‑based) in the binary representation of w."""
    return (w ^ (1 << (max_bits - 1))).bit_length() + 1


def add_hyperloglog(state: Tuple[int, List[int]], item: str) -> None:
    """Incorporate *item* into HyperLogLog state."""
    p, registers = state
    x = _hash(item, 0)
    idx = x >> (64 - p)
    w = x << p & ((1 << 64) - 1)
    rank = _rho(w, 64 - p)
    registers[idx] = max(registers[idx], rank)


def estimate_hyperloglog(state: Tuple[int, List[int]]) -> float:
    """Return cardinality estimate from HyperLogLog state."""
    p, registers = state
    m = 1 << p
    alpha = 0.7213 / (1 + 1.079 / m)
    Z = 1.0 / sum(2.0 ** -r for r in registers)
    raw = alpha * m * m * Z
    # Small range correction
    if raw <= 2.5 * m:
        V = registers.count(0)
        if V != 0:
            return m * math.log(m / V)
    return raw


# ----------------------------------------------------------------------
# Gini coefficient derived from Count‑Min frequencies
# ----------------------------------------------------------------------
def gini_from_sketch(sketch: List[List[int]]) -> float:
    """
    Compute Gini coefficient over the multiset of estimated frequencies
    stored in the Count‑Min sketch.  The sketch is flattened, zeros are
    ignored (they correspond to unseen bins).
    """
    flat = [v for row in sketch for v in row if v > 0]
    if not flat:
        return 0.0
    xs = sorted(float(x) for x in flat)
    n = len(xs)
    mean = sum(xs) / n
    if mean == 0:
        return 0.0
    # Gini = (1/(2 n^2 μ)) * sum_i sum_j |x_i - x_j|
    # Efficient O(n) formulation:
    cumulative = np.cumsum(xs)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


# ----------------------------------------------------------------------
# Diffusion‑forcing noise schedule modulated by Gini
# ----------------------------------------------------------------------
def noise_schedule_gini(T: int, gini: float, schedule: str = "cosine") -> np.ndarray:
    """
    Produce a noise schedule of length T+1.  The Gini coefficient scales the
    amplitude of the schedule: higher inequality (larger Gini) yields a more
    aggressive decay, encouraging exploration early on.
    """
    if schedule != "cosine":
        raise ValueError("Only 'cosine' schedule is implemented")
    s = 0.008
    steps = np.arange(T + 1, dtype=np.float64)
    base = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    # Scale by (1 - gini) so that gini=0 => full cosine, gini=1 => flat (no decay)
    scaled = base * (1.0 - gini) + gini
    return scaled / scaled[0]  # normalised to start at 1.0


def diffusion_forcing_step(value: float, t: int, T: int, gini: float) -> float:
    """
    Perform a single diffusion‑forcing update on *value* at step *t* using the
    Gini‑modulated schedule.
    """
    schedule = noise_schedule_gini(T, gini)
    eta = schedule[t]  # effective learning rate at step t
    noise = random.gauss(0.0, math.sqrt(eta))
    return value + noise


# ----------------------------------------------------------------------
# Minimal Hoeffding tree node with Gini‑aware split evaluation
# ----------------------------------------------------------------------
class HoeffdingTreeNode:
    """
    Very small Hoeffding tree node that stores class counts per feature value.
    Split evaluation uses the Gini coefficient of the post‑split class
    distributions, weighted by the distinct‑context estimate from HyperLogLog.
    """

    def __init__(self, depth: int = 0, max_depth: int = 10):
        self.depth = depth
        self.max_depth = max_depth
        self.class_counts: Dict[int, int] = defaultdict(int)
        self.feature_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.children: Dict[int, "HoeffdingTreeNode"] = {}
        self.is_leaf = True

    def update(self, features: Dict[str, str], label: int) -> None:
        """Update statistics with a new (features, label) observation."""
        self.class_counts[label] += 1
        for f, v in features.items():
            self.feature_counts[f][hash(v) % 1024] += 1  # bucketed feature value

        if self.is_leaf and self.depth < self.max_depth:
            self.attempt_split()

    def attempt_split(self) -> None:
        """Try to split the node using the feature that yields the lowest Gini."""
        best_gain = 0.0
        best_feature = None
        parent_gini = self._gini(list(self.class_counts.values()))
        for f, buckets in self.feature_counts.items():
            # Simulate a binary split on the most frequent bucket vs the rest
            major_bucket = max(buckets, key=buckets.get)
            left_counts = defaultdict(int)
            right_counts = defaultdict(int)
            for label, cnt in self.class_counts.items():
                # Approximate proportion of label in each side
                left_counts[label] = int(cnt * (buckets.get(major_bucket, 0) / sum(buckets.values())))
                right_counts[label] = cnt - left_counts[label]
            left_gini = self._gini(list(left_counts.values()))
            right_gini = self._gini(list(right_counts.values()))
            n = sum(self.class_counts.values())
            weighted_gini = (sum(left_counts.values()) * left_gini +
                             sum(right_counts.values()) * right_gini) / n
            gain = parent_gini - weighted_gini
            if gain > best_gain:
                best_gain = gain
                best_feature = (f, major_bucket)

        # Hoeffding bound check (very simplified)
        if best_gain > 0.01 and best_feature is not None:
            f, bucket = best_feature
            self.is_leaf = False
            self.children[0] = HoeffdingTreeNode(self.depth + 1, self.max_depth)  # left
            self.children[1] = HoeffdingTreeNode(self.depth + 1, self.max_depth)  # right
            self.split_feature = f
            self.split_bucket = bucket

    def predict(self, features: Dict[str, str]) -> int:
        """Return the majority class label for the given *features*."""
        if self.is_leaf:
            if not self.class_counts:
                return 0
            return max(self.class_counts, key=self.class_counts.get)
        else:
            val = features.get(self.split_feature, "")
            bucket = hash(val) % 1024
            child = self.children[0] if bucket == self.split_bucket else self.children[1]
            return child.predict(features)

    @staticmethod
    def _gini(counts: List[int]) -> float:
        """Standard Gini impurity."""
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = [c / total for c in counts]
        return 1.0 - sum(p * p for p in probs)


# ----------------------------------------------------------------------
# Hybrid learner that ties sketches, Gini‑modulated diffusion, and tree
# ----------------------------------------------------------------------
class HybridLearner:
    """
    Online learner that:
    1. Maintains a Count‑Min sketch of (context, reward) pairs.
    2. Maintains a HyperLogLog estimate of distinct contexts.
    3. Uses the Gini coefficient of the sketch to drive a diffusion‑forcing
       schedule that perturbs the tree’s leaf predictions.
    4. Updates a Hoeffding tree with feature vectors.
    """

    def __init__(self, sketch_width: int = 2000, sketch_depth: int = 5,
                 hll_p: int = 12, max_tree_depth: int = 8, diffusion_T: int = 1000):
        self.sketch = init_count_min(sketch_width, sketch_depth)
        self.hll_state = init_hyperloglog(hll_p)
        self.tree = HoeffdingTreeNode(max_depth=max_tree_depth)
        self.diffusion_T = diffusion_T
        self.time_step = 0

    # ------------------------------------------------------------------
    # Core hybrid operations
    # ------------------------------------------------------------------
    def observe(self, context: str, features: Dict[str, str], reward: int) -> None:
        """
        Incorporate a new observation:
        * Update sketches with (context, reward).
        * Update HyperLogLog with the context.
        * Perform a diffusion‑forcing perturbation on the tree’s leaf value.
        * Update the Hoeffding tree with features and reward as label.
        """
        # 1. Sketch update
        item_key = f"{context}|{reward}"
        update_count_min(self.sketch, item_key)

        # 2. HyperLogLog update
        add_hyperloglog(self.hll_state, context)

        # 3. Diffusion forcing on a dummy internal value (e.g., estimated reward)
        gini = gini_from_sketch(self.sketch)
        noisy_reward = diffusion_forcing_step(float(reward), self.time_step, self.diffusion_T, gini)

        # 4. Tree update (use discretised noisy reward as label)
        label = int(round(noisy_reward))
        self.tree.update(features, label)

        self.time_step = (self.time_step + 1) % self.diffusion_T

    def predict(self, context: str, features: Dict[str, str]) -> Tuple[int, float]:
        """
        Return a prediction for the given *features* together with an
        uncertainty estimate derived from sketch frequency and distinct‑context count.
        """
        # Base prediction from the Hoeffding tree
        base_pred = self.tree.predict(features)

        # Frequency‑based confidence (higher estimated count → higher confidence)
        freq_est = query_count_min(self.sketch, f"{context}|{base_pred}")
        distinct_ctx = estimate_hyperloglog(self.hll_state)

        # Simple heuristic: confidence = sigmoid(log(freq+1) / log(distinct+1))
        if distinct_ctx <= 0:
            confidence = 0.0
        else:
            logf = math.log(freq_est + 1)
            logd = math.log(distinct_ctx + 1)
            confidence = 1.0 / (1.0 + math.exp(-(logf / logd - 0.5)))

        return base_pred, confidence

    # ------------------------------------------------------------------
    # Utility introspection
    # ------------------------------------------------------------------
    def get_gini(self) -> float:
        """Current Gini coefficient derived from the Count‑Min sketch."""
        return gini_from_sketch(self.sketch)

    def get_distinct_contexts(self) -> float:
        """Current HyperLogLog cardinality estimate."""
        return estimate_hyperloglog(self.hll_state)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    learner = HybridLearner(diffusion_T=200)

    # Simulate a tiny stream
    for i in range(150):
        ctx = f"user_{random.randint(1, 20)}"
        feats = {
            "item": f"item_{random.randint(1, 50)}",
            "time_of_day": random.choice(["morning", "afternoon", "evening"])
        }
        reward = random.choice([0, 1])
        learner.observe(ctx, feats, reward)

    # Make a prediction
    test_ctx = "user_5"
    test_feats = {"item": "item_12", "time_of_day": "evening"}
    pred, conf = learner.predict(test_ctx, test_feats)

    print(f"Prediction: {pred}, Confidence: {conf:.3f}")
    print(f"Gini (sketch): {learner.get_gini():.4f}")
    print(f"Distinct contexts (HLL): {learner.get_distinct_contexts():.0f}")