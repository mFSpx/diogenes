# DARWIN HAMMER — match 1605, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m958_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fold_c_state_space_duality_m407_s0.py (gen5)
# born: 2026-05-29T23:37:46Z

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Set, Callable
from collections import defaultdict

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

_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

def labeling_function(name: str|None=None): 
    def deco(fn: Callable[[dict],int]): 
        fn.lf_name=name or fn.__name__ 
        return fn 
    return deco 

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]: 
    votes=defaultdict(list) 
    for batch in batches: 
        for r in batch: 
            if r.label in (0,1): votes[r.doc_id].append(r.label) 
    out=[] 
    for d,vs in votes.items(): 
        if not vs: 
            out.append(ProbabilisticLabel(d,0,0.5))
            continue 
        from collections import Counter
        counts = Counter(vs)
        label = max(counts, key=counts.get)
        confidence = counts[label] / len(vs)
        out.append(ProbabilisticLabel(d, label, confidence))
    return out

def reset_policy() -> None:
    """Reset the bandit policy."""
    for action in list(_POLICY.keys()):
        del _POLICY[action]

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY[action]
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY[action][1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return np.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _upper_confidence_bound(action_id: str) -> float:
    """Compute the upper confidence bound."""
    count = _count(action_id)
    if count == 0:
        return np.inf
    return _reward(action_id) + np.sqrt(2 * np.log(_count(action_id)) / count)

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

    # Compute the upper confidence bound
    upper_confidence_bound = _upper_confidence_bound(action_id)

    return BanditUpdate(context_id, action_id, reward, propensity)

def main():
    reset_policy()
    bandit_update = hybrid_operation("doc1", "action1", 0.8, 0.5)
    print(bandit_update)

if __name__ == "__main__":
    main()