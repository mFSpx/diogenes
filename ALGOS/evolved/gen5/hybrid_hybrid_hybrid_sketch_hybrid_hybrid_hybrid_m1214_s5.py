# DARWIN HAMMER — match 1214, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s1.py (gen4)
# born: 2026-05-29T23:34:23Z

"""Hybrid Bayesian‑Bandit‑Curvature Algorithm
Parents:
- hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s1.py (Count‑Min sketch + Ollivier‑Ricci curvature for Bayesian log‑likelihood)
- hybrid_hybrid_hybrid_label__hybrid_hybrid_bandit_m415_s1.py (Bandit action selection & labeling confidence)

Mathematical bridge:
The curvature term 𝜅 = Σ_i f_i·log(f_i) is used to modulate the bandit
propensity (exploration term) while the Count‑Min sketch provides an
approximation 𝐿̂ of the empirical log‑likelihood sum.  The hybrid estimate
combines them as  

    𝔈 = 𝐿̂ · 𝜅 · p̂  

where p̂ is the propensity of the selected bandit action after curvature‑aware
UCB adjustment.  This unifies the Bayesian evidence approximation with the
action‑selection dynamics of the bandit framework.""" 

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Simple Count‑Min sketch."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            index = hash((item, d)) % width
            table[d][index] += 1
    return table


def compute_curvature(features: Dict[str, float]) -> float:
    """Ollivier‑Ricci curvature approximation Σ f·log(f)."""
    curvature = 0.0
    for v in features.values():
        if v > 0:
            curvature += v * math.log(v)
    return curvature


def sketch_log_likelihood(sketch: List[List[int]]) -> float:
    """Approximate empirical log‑likelihood as the total count."""
    return float(sum(sum(row) for row in sketch))


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
class BanditAction:
    """Simple bandit action with UCB‑style fields."""
    def __init__(self, action_id: str):
        self.action_id = action_id
        self.propensity: float = 1.0          # base propensity (exploration weight)
        self.expected_reward: float = 0.0
        self.count: int = 0
        self.confidence_bound: float = float('inf')

    def update(self, reward: float, t: int) -> None:
        """Bayesian‑style incremental update of expected reward and UCB bound."""
        self.count += 1
        alpha = 1.0 / self.count
        self.expected_reward = (1 - alpha) * self.expected_reward + alpha * reward
        # classic UCB confidence term
        self.confidence_bound = math.sqrt((2 * math.log(t + 1)) / self.count)


def select_action(actions: List[BanditAction], curvature: float, t: int) -> BanditAction:
    """
    Curvature‑aware UCB selection.
    The curvature scales the exploration term (propensity).
    """
    ucb_values = []
    for a in actions:
        exploration = curvature * a.propensity * a.confidence_bound
        ucb = a.expected_reward + exploration
        ucb_values.append(ucb)
    idx = int(np.argmax(ucb_values))
    return actions[idx]


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_estimate(
    sketch: List[List[int]],
    features: Dict[str, float],
    actions: List[BanditAction],
    t: int,
) -> Tuple[float, BanditAction]:
    """
    Compute the hybrid estimate 𝔈 = 𝐿̂ · 𝜅 · p̂,
    where 𝐿̂ is the sketch log‑likelihood,
    𝜅 is curvature, and p̂ is the propensity of the selected action.
    Returns the estimate and the action chosen for possible downstream use.
    """
    L_hat = sketch_log_likelihood(sketch)
    kappa = compute_curvature(features)
    action = select_action(actions, kappa, t)
    estimate = L_hat * kappa * action.propensity
    return estimate, action


def hybrid_labeling(
    docs: List[str],
    labeling_functions: Dict[str, Callable[[str], int]],
    actions: List[BanditAction],
    features: Dict[str, float],
) -> List[Tuple[str, int, float]]:
    """
    Apply a bandit‑driven labeling function to each document.
    The selected bandit action determines which labeling function to use.
    After labeling, a synthetic reward (1 if label matches a hidden truth) is
    generated and the action is updated using curvature‑aware UCB.
    Returns a list of (doc_id, label, confidence).
    """
    results = []
    hidden_truth = {doc: random.randint(0, 1) for doc in docs}
    curvature = compute_curvature(features)

    for t, doc in enumerate(docs, start=1):
        # Choose action → choose labeling function
        action = select_action(actions, curvature, t)
        lf_name = action.action_id
        lf = labeling_functions[lf_name]
        label = lf(doc)

        # Synthetic reward
        reward = 1.0 if label == hidden_truth[doc] else 0.0

        # Update bandit statistics
        action.update(reward, t)

        # Confidence is a blend of expected reward and curvature scaling
        confidence = min(1.0, action.expected_reward + curvature * 0.01)

        results.append((doc, label, confidence))
    return results


def hybrid_recovery_priority(sketch: List[List[int]], features: Dict[str, float]) -> float:
    """
    Compute a recovery priority score.
    The priority grows with sketch sparsity (more zero cells) and curvature.
    """
    total_cells = len(sketch) * len(sketch[0])
    zero_cells = sum(1 for row in sketch for v in row if v == 0)
    sparsity = zero_cells / total_cells
    curvature = compute_curvature(features)
    priority = sparsity * curvature
    return priority


# ----------------------------------------------------------------------
# Simple mock labeling functions for demonstration
# ----------------------------------------------------------------------
def lf_random(_: str) -> int:
    return random.randint(0, 1)


def lf_bias_low(_: str) -> int:
    return 0 if random.random() < 0.8 else 1


def lf_bias_high(_: str) -> int:
    return 1 if random.random() < 0.7 else 0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic stream of items for the sketch
    items = [f"token_{i}" for i in range(1000)]
    sketch = count_min_sketch(items, width=128, depth=6)

    # Random feature vector
    feature_names = [f"f{i}" for i in range(15)]
    features = {name: random.random() for name in feature_names}

    # Initialise bandit actions (each maps to a labeling function)
    actions = [BanditAction("rand"), BanditAction("bias_low"), BanditAction("bias_high")]
    labeling_fns = {
        "rand": lf_random,
        "bias_low": lf_bias_low,
        "bias_high": lf_bias_high,
    }

    # Hybrid estimate demonstration
    estimate, chosen_action = hybrid_estimate(sketch, features, actions, t=1)
    print(f"Hybrid estimate: {estimate:.3f} (action: {chosen_action.action_id})")

    # Hybrid labeling demonstration
    docs = [f"doc_{i}" for i in range(20)]
    labeling_results = hybrid_labeling(docs, labeling_fns, actions, features)
    for doc_id, label, conf in labeling_results[:5]:
        print(f"{doc_id}: label={label}, confidence={conf:.2f}")

    # Recovery priority demonstration
    priority = hybrid_recovery_priority(sketch, features)
    print(f"Recovery priority score: {priority:.5f}")