# DARWIN HAMMER — match 4461, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hoeffding_tre_m933_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2526_s0.py (gen6)
# born: 2026-05-29T23:55:55Z

"""
Hybrid Algorithm: Fusion of DARWIN HAMMER — match 933, survivor 1 
and DARWIN HAMMER — match 2526, survivor 0

This module integrates the governing equations of the Hybrid Hoeffding Tree 
with the Hybrid Bandit algorithm. The mathematical bridge between the two 
parents is the use of the bandit algorithm's propensity and expected reward 
as probabilistic weights for each node in the Hoeffding Tree, and the 
use of the Hoeffding Tree's gini impurity and gain as a basis for the 
bandit algorithm's exploration-exploitation trade-off.

By leveraging the properties of both algorithms, we can optimize the model's 
performance while minimizing memory usage and incorporating uncertainty.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Dict, List, Tuple
import pathlib

Vector = Sequence[float]

@dataclass(frozen=True)
class RBFSurrogate:
    centers: list[tuple[float, ...]]
    weights: list[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
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

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("range_ > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((range_ ** 2 * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def gini_impurity_from_counts(counts: Dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.fromiter((c / total for c in counts.values()), dtype=float)
    return 1.0 - np.sum(probs ** 2)

def gini_gain(parent_counts: Dict, left_counts: Dict, right_counts: Dict) -> float:
    n_parent = sum(parent_counts.values())
    if n_parent == 0:
        return 0.0

    parent_imp = gini_impurity_from_counts(parent_counts)
    left_imp = gini_impurity_from_counts(left_counts)
    right_imp = gini_impurity_from_counts(right_counts)

    n_left = sum(left_counts.values())
    n_right = sum(right_counts.values())

    return parent_imp - (n_left / n_parent) * left_imp - (n_right / n_parent) * right_imp

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}  # virtual VRAM store per key

def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco

def aggregate_labels(batches: List[List[Dict]]) -> List[Dict]:
    """Pure A-logic: majority vote with confidence = proportion of votes."""
    votes: Dict[str, List[int]] = {}
    for batch in batches:
        for r in batch:
            if r['label'] in (0, 1):
                votes.setdefault(r['doc_id'], []).append(r['label'])
    out = []
    for doc_id, labels in votes.items():
        label = int(np.mean(labels) > 0.5)
        confidence = np.mean(labels) if label == 1 else 1 - np.mean(labels)
        out.append({'doc_id': doc_id, 'label': label, 'confidence': confidence})
    return out

def hybrid_bandit_hoeffding_tree(data: List[Dict], bandit_actions: List[BanditAction]) -> List[Dict]:
    """Hybrid bandit and Hoeffding tree algorithm."""
    # Calculate gini impurity and gain for each node
    gini_impurities = {}
    gini_gains = {}
    for action in bandit_actions:
        counts = {}
        for item in data:
            if item['label'] == action.action_id:
                counts[action.action_id] = counts.get(action.action_id, 0) + 1
        gini_impurities[action.action_id] = gini_impurity_from_counts(counts)
        gini_gains[action.action_id] = gini_gain(counts, {}, {})

    # Update bandit actions with gini impurity and gain
    updated_bandit_actions = []
    for action in bandit_actions:
        updated_action = BanditAction(
            action_id=action.action_id,
            propensity=action.propensity,
            expected_reward=action.expected_reward,
            confidence_bound=action.confidence_bound,
            algorithm=action.algorithm
        )
        updated_action.gini_impurity = gini_impurities[action.action_id]
        updated_action.gini_gain = gini_gains[action.action_id]
        updated_bandit_actions.append(updated_action)

    # Select action with highest gini gain
    selected_action = max(updated_bandit_actions, key=lambda x: x.gini_gain)

    # Update policy and store
    _POLICY[selected_action.action_id] = [selected_action.gini_impurity, selected_action.gini_gain]
    _STORE[selected_action.action_id] = selected_action.confidence_bound

    return aggregate_labels([[{'doc_id': item['doc_id'], 'label': item['label']} for item in data]])

if __name__ == "__main__":
    data = [
        {'doc_id': '1', 'label': 'A'},
        {'doc_id': '2', 'label': 'B'},
        {'doc_id': '3', 'label': 'A'},
        {'doc_id': '4', 'label': 'B'},
        {'doc_id': '5', 'label': 'A'}
    ]

    bandit_actions = [
        BanditAction('A', 0.5, 0.5, 0.1, 'hybrid'),
        BanditAction('B', 0.5, 0.5, 0.1, 'hybrid')
    ]

    result = hybrid_bandit_hoeffding_tree(data, bandit_actions)
    print(result)