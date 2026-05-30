# DARWIN HAMMER — match 1605, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m958_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s0.py (gen5)
# born: 2026-05-29T23:37:46Z

import numpy as np
from datetime import date
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable
from collections import defaultdict, Counter

@dataclass(frozen=True)
class LabelingFunctionResult: 
    lf_name: str 
    doc_id: str 
    label: int 

@dataclass(frozen=True)
class ProbabilisticLabel: 
    doc_id: str 
    label: int 
    confidence: float 

@dataclass(frozen=True)
class LabelError: 
    doc_id: str 
    given_label: int 
    suggested_label: int 
    error_probability: float 

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

_POLICY: dict = defaultdict(lambda: [0.0, 0.0])

def labeling_function(name: str|None=None): 
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__ 
        return fn 
    return deco 

def aggregate_labels(batches: list[list[LabelingFunctionResult]]) -> list[ProbabilisticLabel]: 
    votes=defaultdict(list) 
    for batch in batches: 
        for r in batch: 
            if r.label in (0,1): votes[r.doc_id].append(r.label) 
    out=[] 
    for d,vs in votes.items(): 
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5)); 
            continue 
        counts = Counter(vs)
        label = max(counts, key=counts.get)
        confidence = counts[label] / len(vs)
        out.append(ProbabilisticLabel(d, label, confidence))
    return out

def reset_policy() -> None:
    """Reset the bandit policy."""
    global _POLICY
    _POLICY = defaultdict(lambda: [0.0, 0.0])

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY[action]
    return total / max(n, 1) if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY[action][1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return math.log(max(x, eps) / max(abs(x), eps)) if x != 0 else 0.0

def hybrid_operation(context_id: str, action_id: str, reward: float, propensity: float) -> BanditUpdate:
    """Perform the hybrid operation."""
    # Update the bandit policy
    if action_id not in _POLICY:
        _POLICY[action_id] = [0.0, 0.0]
    _POLICY[action_id][0] += reward
    _POLICY[action_id][1] += 1

    # Compute the hybrid store factor
    count = _count(action_id)
    log_count_ratio = _fold_change_detection(count, 1e-6)
    hybrid_store_factor = _hybrid_store_factor(action_id, count, log_count_ratio)

    # Generate a probabilistic label
    labeling_function_result = LabelingFunctionResult("hybrid", context_id, 1 if reward > 0.5 else 0)
    probabilistic_label = aggregate_labels([[labeling_function_result]])[0]

    # Refine the reward calculation
    refined_reward = (reward + _reward(action_id)) / 2

    return BanditUpdate(context_id, action_id, refined_reward, propensity)

def kl_divergence(p: float, q: float) -> float:
    """Compute the KL divergence between two probabilities."""
    if p == 0:
        return 0
    if q == 0:
        return np.inf
    return p * math.log(p / q)

def kullback_leibler_distance(action_id: str) -> float:
    """Compute the Kullback-Leibler distance."""
    reward = _reward(action_id)
    count = _count(action_id)
    kl_div = 0
    for action, (total, n) in _POLICY.items():
        if action != action_id:
            p = total / max(n, 1)
            q = reward
            kl_div += kl_divergence(p, q)
    return kl_div

def improved_hybrid_operation(context_id: str, action_id: str, reward: float, propensity: float) -> BanditUpdate:
    """Perform the improved hybrid operation."""
    # Update the bandit policy
    if action_id not in _POLICY:
        _POLICY[action_id] = [0.0, 0.0]
    _POLICY[action_id][0] += reward
    _POLICY[action_id][1] += 1

    # Compute the hybrid store factor
    count = _count(action_id)
    log_count_ratio = _fold_change_detection(count, 1e-6)
    hybrid_store_factor = _hybrid_store_factor(action_id, count, log_count_ratio)

    # Generate a probabilistic label
    labeling_function_result = LabelingFunctionResult("hybrid", context_id, 1 if reward > 0.5 else 0)
    probabilistic_label = aggregate_labels([[labeling_function_result]])[0]

    # Refine the reward calculation
    refined_reward = (reward + _reward(action_id)) / 2

    # Incorporate KL divergence
    kl_div = kullback_leibler_distance(action_id)

    return BanditUpdate(context_id, action_id, refined_reward + kl_div / (1 + _count(action_id)), propensity)

def main():
    reset_policy()
    bandit_update = improved_hybrid_operation("doc1", "action1", 0.8, 0.5)
    print(bandit_update)

if __name__ == "__main__":
    main()