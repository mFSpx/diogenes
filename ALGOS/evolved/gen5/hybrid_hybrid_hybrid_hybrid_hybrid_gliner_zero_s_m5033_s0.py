# DARWIN HAMMER — match 5033, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py (gen4)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s5.py (gen1)
# born: 2026-05-29T23:59:20Z

"""
This module fuses two parent algorithms: 
1. hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py - a contextual multi-armed bandit that tracks cumulative reward and uses a LinUCB-style confidence bound, 
2. hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s5.py - a label matcher that returns deterministic spans.

The mathematical bridge between these two structures is the use of a non-linear mapping to predict the reward in the bandit algorithm. 
In this hybrid algorithm, we use the label matcher from the second parent to generate a set of labels that are used as the context in the bandit algorithm.
The bandit algorithm then uses these labels to select the best action and update its confidence bounds.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Shared Types
Vector = list[float]

# Bandit core
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

_POLICY: dict[str, list[float]] = {}  
_STORE: dict[str, float] = {}                
_SURROGATE = None                             

def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = None

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0])
    return total / n if n > 0 else 0.0

class RBFSurrogate:
    def __init__(self, centers: list[Vector], weights: list[float], epsilon: float):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        return sum(w * math.exp(-self.epsilon ** 2 * sum((a - b) ** 2 for a, b in zip(x, c))) for w, c in zip(self.weights, self.centers))

def update_surrogate(context_id: str, action_id: str, reward: float) -> None:
    global _SURROGATE
    if _SURROGATE is None:
        _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)
    x = [float(i) for i in context_id] + [1.0 if action_id == 'action1' else 0.0]
    _SURROGATE.centers.append(x)
    _SURROGATE.weights.append(reward)

def select_action(context_id: str) -> str:
    ucbs = []
    for action_id in _POLICY:
        expected_reward = _empirical_reward(action_id)
        confidence_bound = math.sqrt(2 * math.log(len(_POLICY)) / _POLICY[action_id][1])
        ucb = expected_reward + confidence_bound
        ucbs.append(ucb)
    best_action = 'action1' if ucbs[0] > ucbs[1] else 'action2'
    return best_action

def literal_fallback(text: str, labels: list[str], *, case_sensitive: bool = False) -> list:
    flags = 0 if case_sensitive else 0
    spans = []
    seen = set()
    for label in labels:
        candidates = {label, label.replace(" / ", " ").replace("-", " ")}
        for candidate in candidates:
            for match in [m for m in re.finditer(candidate, text, flags)]:
                span = (match.start(), match.end(), candidate)
                if span not in seen:
                    spans.append(span)
                    seen.add(span)
    return spans

def generate_labels(text: str, labels: list[str]) -> list:
    spans = literal_fallback(text, labels)
    return [span[2] for span in spans]

def run_bandit(text: str, labels: list[str]) -> None:
    reset_policy()
    context_id = str(id(text))
    action_id = select_action(context_id)
    reward = 1.0 if action_id == 'action1' else 0.0
    _POLICY[context_id] = [reward, 1]
    update_surrogate(context_id, action_id, reward)

if __name__ == "__main__":
    text = "This is a test text"
    labels = ["test", "text"]
    run_bandit(text, labels)