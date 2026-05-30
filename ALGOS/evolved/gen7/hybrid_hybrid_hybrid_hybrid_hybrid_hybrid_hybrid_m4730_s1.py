# DARWIN HAMMER — match 4730, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1377_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s3.py (gen6)
# born: 2026-05-29T23:57:52Z

"""Hybrid Decision‑Bandit Associative Memory (HDBAM)

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1377_s0.py (Decision Hygiene + Dense Associative Memory)
- hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s3.py (RBF surrogate + Contextual Bandit + Multivector)

Mathematical bridge:
Both parents rely on matrix‑based similarity measures.  The Decision‑Hygiene model optimises a Gini‑based
weight matrix **W** that governs pattern recall in a Hopfield‑style associative memory.  The Krampus‑Brain
model builds a Gaussian‑RBF kernel matrix **K** (K_{ij}=exp(-||x_i‑x_j||²/(2σ²))) that serves as a similarity
metric for contextual bandits.  We fuse them by **using the kernel matrix K as the weight matrix W** for the
associative memory, while the bandit’s context vector is obtained from the decision‑hygiene optimisation
(Gini‑weighted feature vector).  Multivectors provide a geometric‑algebraic representation of contexts,
allowing scalar (grade‑0) extraction as the effective similarity score used in the Upper‑Confidence‑Bound
(UCB) bandit policy.

The resulting system:
1. Build a kernel matrix K from raw contexts → treat K as Hopfield weight matrix.
2. Optimise a Gini‑derived diagonal scaling D that balances exploration vs exploitation.
3. Represent each context as a Multivector; its scalar part drives the bandit’s propensity.
4. Recall stored patterns via sign(K·p) and use the recalled pattern to update bandit statistics.

This file implements the unified algorithm with three public functions demonstrating the hybrid
operation.
"""

import sys
import math
import random
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Data structures (adapted from parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

class Multivector:
    """Very small subset of geometric algebra needed for the hybrid."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # prune near‑zero components
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    def scalar_part(self) -> float:
        """Grade‑0 (scalar) component."""
        return self.components.get(frozenset(), 0.0)

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        """Encode a real vector as a grade‑1 multivector."""
        comps = {frozenset([i]): float(v) for i, v in enumerate(vec)}
        return Multivector(comps, n=vec.shape[0])

    def dot(self, other: "Multivector") -> float:
        """Geometric inner product reduced to scalar part for grade‑1 vectors."""
        # Only grade‑1 components contribute to the Euclidean dot product
        total = 0.0
        for blade, coef in self.components.items():
            if len(blade) == 1 and blade in other.components:
                total += coef * other.components[blade]
        return total

# ----------------------------------------------------------------------
# Core hybrid utilities
# ----------------------------------------------------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1‑D array."""
    if x.ndim != 1:
        raise ValueError("Gini is defined for 1‑D arrays.")
    sorted_x = np.sort(np.abs(x))
    n = len(x)
    cumulative = np.cumsum(sorted_x)
    return (2.0 / n) * np.sum((np.arange(1, n + 1) * sorted_x)) / cumulative[-1] - (n + 1) / n


def build_kernel_matrix(contexts: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian RBF kernel matrix K_{ij}=exp(-||c_i-c_j||²/(2σ²)).
    The matrix is symmetric positive‑definite and will be used as the Hopfield weight matrix.
    """
    if contexts.ndim != 2:
        raise ValueError("Contexts must be a 2‑D array (samples × features).")
    sq_norms = np.sum(contexts ** 2, axis=1, keepdims=True)
    dists = sq_norms + sq_norms.T - 2.0 * contexts @ contexts.T
    K = np.exp(-dists / (2.0 * sigma ** 2))
    np.fill_diagonal(K, 0.0)  # no self‑connection in Hopfield formulation
    return K


def associative_recall(pattern: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Hopfield‑style recall: sign(W·pattern).  The weight matrix is the kernel K.
    """
    raw = weight_matrix @ pattern
    return np.sign(raw)


def ucb_score(mean: float, confidence: float, exploration: float = 1.0) -> float:
    """Upper Confidence Bound score used by the bandit."""
    return mean + exploration * confidence


def bandit_select(
    context_vec: Multivector,
    actions: List[BanditAction],
    similarity_matrix: np.ndarray,
    action_index_map: Dict[str, int],
) -> BanditAction:
    """
    Select an action using a UCB that incorporates similarity between the
    current context and stored contexts (via the kernel matrix).
    The scalar part of the context multivector modulates the propensity.
    """
    best_score = -np.inf
    best_action = None
    for act in actions:
        idx = action_index_map[act.action_id]
        # similarity of current context to the prototype of this action
        sim = similarity_matrix[idx].dot(context_vec.components.get(frozenset([0]), 0.0))
        # combine similarity with bandit statistics
        mean = act.expected_reward
        conf = act.confidence_bound
        score = ucb_score(mean, conf) + act.propensity * sim
        if score > best_score:
            best_score = score
            best_action = act
    return best_action


# ----------------------------------------------------------------------
# Hybrid functions (public API)
# ----------------------------------------------------------------------
def compute_hybrid_gini_weight(contexts: np.ndarray) -> np.ndarray:
    """
    Compute a diagonal scaling matrix D where D_ii = 1 / (1 + Gini_i)
    for each context vector i.  This balances exploration (high Gini → lower weight)
    and is multiplied element‑wise with the kernel matrix.
    """
    gini_vals = np.apply_along_axis(gini_coefficient, 1, contexts)
    D = np.diag(1.0 / (1.0 + gini_vals))
    return D


def hybrid_predict(
    raw_context: np.ndarray,
    stored_patterns: np.ndarray,
    kernel_sigma: float = 1.0,
) -> Tuple[np.ndarray, BanditAction]:
    """
    End‑to‑end hybrid prediction:
    1. Build kernel matrix K from stored contexts.
    2. Scale K with Gini‑derived diagonal D.
    3. Encode raw_context as a Multivector.
    4. Recall the nearest stored pattern via associative memory.
    5. Use the recalled pattern as the context for a contextual bandit decision.
    Returns the recalled pattern and the selected BanditAction.
    """
    # 1‑2: kernel + Gini scaling
    K = build_kernel_matrix(stored_patterns, sigma=kernel_sigma)
    D = compute_hybrid_gini_weight(stored_patterns)
    W = D @ K @ D.T  # symmetrised, Gini‑scaled weight matrix

    # 3: multivector encoding (grade‑1)
    ctx_mv = Multivector.from_vector(raw_context)

    # 4: associative recall
    recalled = associative_recall(raw_context, W)

    # 5: bandit selection
    # Create a tiny action set for demonstration
    actions = [
        BanditAction(
            action_id=f"a{i}",
            propensity=random.random(),
            expected_reward=random.uniform(0, 1),
            confidence_bound=random.uniform(0.1, 0.5),
            algorithm="HDBAM",
        )
        for i in range(3)
    ]
    # similarity matrix between actions and stored patterns (use rows of K)
    action_index_map = {act.action_id: i % stored_patterns.shape[0] for i, act in enumerate(actions)}
    selected = bandit_select(ctx_mv, actions, K, action_index_map)

    return recalled, selected


def hybrid_update(
    stored_patterns: np.ndarray,
    new_pattern: np.ndarray,
    updates: List[BanditUpdate],
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Update the stored pattern matrix with a new pattern using a Hebbian‑style rule,
    then recompute the kernel matrix (including the new pattern) and return it.
    Also incorporate bandit reward feedback to adjust expected rewards.
    """
    # Hebbian addition (outer product) – simple associative memory learning
    stored_patterns = np.vstack([stored_patterns, new_pattern])
    # Re‑compute kernel with the enlarged set
    K = build_kernel_matrix(stored_patterns, sigma=sigma)

    # Simple bandit reward aggregation (placeholder for a real learning rule)
    # Here we just print the aggregated reward per action for visibility.
    reward_sum: Dict[str, float] = {}
    count_sum: Dict[str, int] = {}
    for upd in updates:
        reward_sum[upd.action_id] = reward_sum.get(upd.action_id, 0.0) + upd.reward
        count_sum[upd.action_id] = count_sum.get(upd.action_id, 0) + 1

    # Debug output (could be replaced by proper state mutation)
    for aid in reward_sum:
        avg = reward_sum[aid] / count_sum[aid]
        print(f"[HybridUpdate] Action {aid}: avg reward {avg:.3f} over {count_sum[aid]} trials")

    return K


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic contexts (5 samples, 4 features)
    np.random.seed(42)
    stored = np.random.randn(5, 4)

    # a fresh raw context
    raw = np.random.randn(4)

    # Run prediction
    pattern, action = hybrid_predict(raw, stored, kernel_sigma=0.8)
    print("Recalled pattern:", pattern)
    print("Selected action:", asdict(action))

    # Simulate an update with a new pattern and dummy bandit feedback
    new_pat = np.random.randn(4)
    updates = [
        BanditUpdate(context_id="c0", action_id="a0", reward=1.0, propensity=0.7),
        BanditUpdate(context_id="c1", action_id="a1", reward=0.2, propensity=0.4),
        BanditUpdate(context_id="c2", action_id="a2", reward=0.5, propensity=0.6),
    ]
    K_new = hybrid_update(stored, new_pat, updates, sigma=0.8)
    print("Updated kernel shape:", K_new.shape)