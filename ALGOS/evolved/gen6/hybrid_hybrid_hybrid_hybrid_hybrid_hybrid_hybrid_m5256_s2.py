# DARWIN HAMMER — match 5256, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m2337_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m2698_s1.py (gen5)
# born: 2026-05-30T00:01:01Z

import numpy as np
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable

@dataclass(frozen=True)
class MathAction:
    id: str; expected_value: float; cost: float=0.0; risk: float=0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str; outcome_value: float; probability: float=1.0

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

def compute_regret_weighted_strategy(actions: List[MathAction], counterfactuals: List[MathCounterfactual]) -> Dict[str,float]:
    if not actions: return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions}
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}

def path_signature(sequence: List[List[float]]) -> float:
    if not sequence: return 0.0
    dim = len(sequence[0])
    signature = 1.0
    for i in range(len(sequence) - 1):
        diff = np.array(sequence[i + 1]) - np.array(sequence[i])
        signature += np.linalg.norm(diff)
    return signature

def shannon_entropy(probabilities: Iterable[float]) -> float:
    probs = [p for p in probabilities if p > 0]
    return -sum(p * math.log(p, 2) for p in probs)

def hybrid_select_action(actions: List[BanditAction], regret_weighted_strategy: Dict[str, float]) -> str:
    action_ids = [action.action_id for action in actions]
    probabilities = [regret_weighted_strategy.get(action_id, 0.0) for action_id in action_ids]
    probabilities = [p / sum(probabilities) if sum(probabilities) != 0 else 1 / len(action_ids) for p in probabilities]
    selected_action_id = np.random.choice(action_ids, p=probabilities)
    return selected_action_id

def hybrid_rlct_estimate(actions: List[BanditAction], regret_weighted_strategy: Dict[str, float]) -> float:
    action_ids = [action.action_id for action in actions]
    probabilities = [regret_weighted_strategy.get(action_id, 0.0) for action_id in action_ids]
    probabilities = [p / sum(probabilities) if sum(probabilities) != 0 else 1 / len(action_ids) for p in probabilities]
    expected_reward = sum([action.expected_reward * probability for action, probability in zip(actions, probabilities)])
    return expected_reward

def build_hybrid_sketch(actions: List[MathAction], counterfactuals: List[MathCounterfactual], 
                       resource_vector: List[float], feature_vector: List[float]) -> Tuple[Dict[str, float], List[List[float]]]:
    regret_weighted_strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    sequence = [[action.expected_value, action.cost, action.risk] for action in actions]
    augmented_path = np.concatenate((np.array(resource_vector).reshape(-1, 1), np.array(feature_vector).reshape(-1, 1)), axis=1).tolist()
    return regret_weighted_strategy, sequence, augmented_path

def compute_augmented_path_signature(augmented_path: List[List[float]]) -> float:
    if not augmented_path: return 0.0
    signature = 1.0
    for i in range(len(augmented_path) - 1):
        diff = np.array(augmented_path[i + 1]) - np.array(augmented_path[i])
        signature += np.linalg.norm(diff)
    return signature

if __name__ == "__main__":
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    counterfactuals = [MathCounterfactual("action1", 5.0), MathCounterfactual("action2", 10.0)]
    regret_weighted_strategy, sequence, augmented_path = build_hybrid_sketch(actions, counterfactuals, [1.0, 2.0], [3.0, 4.0, 5.0])
    print(compute_regret_weighted_strategy(actions, counterfactuals))
    print(path_signature(sequence))
    print(compute_augmented_path_signature(augmented_path))
    bandit_actions = [BanditAction("action1", 0.5, 10.0, 1.0, "algorithm1"), BanditAction("action2", 0.5, 20.0, 1.0, "algorithm2")]
    print(hybrid_select_action(bandit_actions, regret_weighted_strategy))
    print(hybrid_rlct_estimate(bandit_actions, regret_weighted_strategy))