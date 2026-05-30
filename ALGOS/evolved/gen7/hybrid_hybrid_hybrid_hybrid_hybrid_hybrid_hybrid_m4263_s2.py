# DARWIN HAMMER — match 4263, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1605_s0.py (gen6)
# born: 2026-05-29T23:54:39Z

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Endpoint:
    """Geometric representation of a system state."""
    morphology: List[float]


@dataclass(frozen=True)
class FractionalHealthScore:
    """Result of the health‑score fusion."""
    score: float
    weights: List[float]


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int
    confidence: float = 1.0  # optional confidence supplied by the LF


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float  # probability of being chosen under the logging policy
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid_ucb"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Statistical sketching (Count‑Min Sketch)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very lightweight Count‑Min Sketch for approximate frequency counts."""

    def __init__(self, width: int = 2_048, depth: int = 4, seed: int = 0):
        self.width = width
        self.depth = depth
        rng = np.random.default_rng(seed)
        self.hash_seeds = rng.integers(1, 2**31 - 1, size=depth, dtype=np.int64)
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: str, i: int) -> int:
        return (hash(item) ^ self.hash_seeds[i]) % self.width

    def add(self, item: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.table[i, idx] += increment

    def estimate(self, item: str) -> int:
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))


# ----------------------------------------------------------------------
# Singular‑learning‑theory utilities
# ----------------------------------------------------------------------
def singular_complexity(matrix: np.ndarray) -> float:
    """
    Returns a complexity penalty derived from the singular values of `matrix`.
    Inspired by singular learning theory: larger spread of singular values
    implies higher model complexity.
    """
    if matrix.size == 0:
        return 0.0
    s = np.linalg.svd(matrix, compute_uv=False)
    # Normalise and take a log‑scaled sum to keep the term bounded.
    norm_s = s / (s.sum() + 1e-12)
    return -np.sum(norm_s * np.log1p(norm_s))


# ----------------------------------------------------------------------
# Information‑theoretic primitives
# ----------------------------------------------------------------------
def shannon_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts if c > 0])
    return -np.sum(probs * np.log2(probs))


def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# Fusion core – deeper mathematical integration
# ----------------------------------------------------------------------
def fused_loglikelihood(
    endpoint: Endpoint,
    feature_vector: List[float],
    sketch: CountMinSketch,
    feature_keys: List[str],
) -> float:
    """
    Approximate the log‑likelihood contribution of a reward sequence by
    jointly using:
      * Shannon entropy of the sketch‑based feature frequencies,
      * Euclidean distance between the endpoint morphology and the feature vector,
      * A singular‑complexity penalty on the (endpoint, feature) matrix.
    The three components are combined in a principled log‑space additive fashion.
    """
    # 1️⃣ Entropy term from the sketch
    sketch_counts = [sketch.estimate(k) for k in feature_keys]
    entropy = shannon_entropy(sketch_counts)

    # 2️⃣ Distance term (converted to similarity)
    dist = euclidean(endpoint.morphology, feature_vector)
    similarity = -dist  # larger similarity → higher log‑likelihood

    # 3️⃣ Complexity penalty
    mat = np.vstack([endpoint.morphology, feature_vector])
    complexity = -singular_complexity(mat)

    # Normalise each term to comparable scale
    norm_entropy = entropy / (math.log2(max(1, sum(sketch_counts))) + 1e-12)
    norm_similarity = similarity / (np.linalg.norm(endpoint.morphology) + 1e-12)
    norm_complexity = complexity  # already bounded between -1 and 0

    # Log‑likelihood approximation (additive in log‑space)
    return norm_entropy + norm_similarity + norm_complexity


def compute_health_score(
    endpoint: Endpoint,
    feature_counts: List[int],
    weight_entropy: float = 0.4,
    weight_distance: float = 0.4,
    weight_complexity: float = 0.2,
) -> FractionalHealthScore:
    """
    A richer health score that respects the three fused components.
    """
    # Entropy component
    entropy = shannon_entropy(feature_counts)

    # Distance component (using a dummy feature vector of same length)
    dummy_feature = [float(c) for c in feature_counts]
    distance = euclidean(endpoint.morphology, dummy_feature)

    # Complexity component (matrix built from endpoint + dummy_feature)
    mat = np.vstack([endpoint.morphology, dummy_feature])
    complexity = singular_complexity(mat)

    # Weighted aggregation
    score = (
        weight_entropy * entropy
        - weight_distance * distance  # distance penalises health
        - weight_complexity * complexity
    )
    weights = [weight_entropy, weight_distance, weight_complexity]
    return FractionalHealthScore(score, weights)


# ----------------------------------------------------------------------
# Label aggregation with confidence weighting
# ----------------------------------------------------------------------
def aggregate_labels(
    batches: Iterable[Iterable[LabelingFunctionResult]],
) -> List[ProbabilisticLabel]:
    """
    Weighted majority vote where each LF contributes its confidence.
    """
    result: List[ProbabilisticLabel] = []
    for batch in batches:
        batch = list(batch)
        if not batch:
            continue
        doc_id = batch[0].doc_id
        vote_weights: Dict[int, float] = defaultdict(float)
        for lf_res in batch:
            vote_weights[lf_res.label] += lf_res.confidence
        max_label = max(vote_weights, key=vote_weights.get)
        total_weight = sum(vote_weights.values())
        confidence = vote_weights[max_label] / total_weight if total_weight > 0 else 0.0
        result.append(ProbabilisticLabel(doc_id, max_label, confidence))
    return result


# ----------------------------------------------------------------------
# Contextual bandit with Upper‑Confidence Bound (UCB) that respects propensity
# ----------------------------------------------------------------------
class HybridUCB:
    """
    Maintains per‑action statistics and selects actions using a
    propensity‑aware UCB rule:
        Q + beta * sqrt( log(T) / (N * propensity) )
    where:
        Q – empirical mean reward,
        T – total number of updates,
        N – number of times the action was taken,
        beta – exploration coefficient.
    """

    def __init__(self, beta: float = 1.0):
        self._stats: Dict[str, Tuple[float, int, float]] = defaultdict(
            lambda: (0.0, 0, 1.0)
        )  # action_id -> (cumulative_reward, count, avg_propensity)
        self._total_updates: int = 0
        self.beta = beta

    def register_action(self, action: BanditAction) -> None:
        # Ensure the action exists in the table with its initial propensity.
        cum, cnt, _ = self._stats[action.action_id]
        self._stats[action.action_id] = (cum, cnt, action.propensity)

    def update(self, upd: BanditUpdate) -> None:
        cum, cnt, avg_prop = self._stats[upd.action_id]
        new_cnt = cnt + 1
        new_cum = cum + upd.reward
        # Running average of propensity (use logged propensity)
        new_avg_prop = ((avg_prop * cnt) + upd.propensity) / new_cnt
        self._stats[upd.action_id] = (new_cum, new_cnt, new_avg_prop)
        self._total_updates += 1

    def _ucb_score(self, action_id: str) -> float:
        cum, cnt, avg_prop = self._stats[action_id]
        if cnt == 0:
            return float("inf")
        mean_reward = cum / cnt
        exploration = (
            self.beta
            * math.sqrt(math.log(self._total_updates + 1) / (cnt * max(avg_prop, 1e-12)))
        )
        return mean_reward + exploration

    def select_action(self, candidate_actions: Iterable[BanditAction]) -> BanditAction:
        best_action = None
        best_score = -float("inf")
        for act in candidate_actions:
            self.register_action(act)
            score = self._ucb_score(act.action_id)
            if score > best_score:
                best_score, best_action = score, act
        # Attach the latest UCB estimate to the returned action
        return BanditAction(
            action_id=best_action.action_id,
            propensity=best_action.propensity,
            expected_reward=self._stats[best_action.action_id][0]
            / max(self._stats[best_action.action_id][1], 1),
            confidence_bound=best_score
            - self._stats[best_action.action_id][0]
            / max(self._stats[best_action.action_id][1], 1),
            algorithm=best_action.algorithm,
        )


# ----------------------------------------------------------------------
# Decorator for labeling functions (keeps original semantics)
# ----------------------------------------------------------------------
def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


# ----------------------------------------------------------------------
# Demonstration / simple sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Health score demo -----
    endpoint = Endpoint([0.6, 0.4])
    feature_counts = [3, 7, 2]
    health = compute_health_score(endpoint, feature_counts)
    print("HealthScore:", health)

    # ----- Sketch + fused log‑likelihood demo -----
    sketch = CountMinSketch(width=1024, depth=3, seed=42)
    keys = ["feat_a", "feat_b", "feat_c"]
    for k, c in zip(keys, feature_counts):
        sketch.add(k, c)
    feature_vec = [float(c) for c in feature_counts]
    ll = fused_loglikelihood(endpoint, feature_vec, sketch, keys)
    print("FusedLogLikelihood:", ll)

    # ----- Label aggregation demo -----
    batches = [
        [
            LabelingFunctionResult("lf1", "doc1", 1, 0.9),
            LabelingFunctionResult("lf2", "doc1", 0, 0.6),
            LabelingFunctionResult("lf3", "doc1", 1, 0.8),
        ],
        [
            LabelingFunctionResult("lf1", "doc2", 0, 1.0),
            LabelingFunctionResult("lf2", "doc2", 0, 0.7),
        ],
    ]
    aggregated = aggregate_labels(batches)
    for pl in aggregated:
        print("AggregatedLabel:", pl)

    # ----- Hybrid UCB demo -----
    bandit = HybridUCB(beta=0.7)
    actions = [
        BanditAction("a1", propensity=0.5, algorithm="hybrid_ucb"),
        BanditAction("a2", propensity=0.3, algorithm="hybrid_ucb"),
        BanditAction("a3", propensity=0.2, algorithm="hybrid_ucb"),
    ]

    # Simulate a few rounds
    for t in range(10):
        chosen = bandit.select_action(actions)
        # Synthetic reward: higher for action a1
        reward = 1.0 if chosen.action_id == "a1" else 0.0
        upd = BanditUpdate(context_id=f"c{t}", action_id=chosen.action_id, reward=reward, propensity=chosen.propensity)
        bandit.update(upd)
        print(f"Round {t}: chosen={chosen.action_id}, reward={reward}, ucb={chosen.confidence_bound:.3f}")