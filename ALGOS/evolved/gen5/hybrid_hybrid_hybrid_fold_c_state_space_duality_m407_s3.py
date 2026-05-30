# DARWIN HAMMER — match 407, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:28:49Z

"""
This module fuses the 'hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2' and 'state_space_duality' algorithms.
The mathematical bridge between the two algorithms lies in the usage of state transitions and output projections.
In 'hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2', the state transitions are implicit in the bandit algorithm,
whereas in 'state_space_duality', the state transitions are explicit. We can fuse these two algorithms by using the 
explicit state transitions from 'state_space_duality' to inform the bandit algorithm in 'hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2'.
This is achieved by using the state transition matrix A and output projection matrix C from 'state_space_duality' 
to compute the hybrid store factor and reward in the bandit algorithm.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from dataclasses import dataclass

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

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float, A: np.ndarray, C: np.ndarray) -> float:
    """Compute the hybrid store factor using the state transition matrix A and output projection matrix C."""
    return log_count_ratio * count * np.trace(A) * np.trace(C)

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _phermone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Compute the pheromone infotaxis."""
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

def _decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    """Compute the decision hygiene shannon entropy."""
    phermone_infotaxis_value = _phermone_infotaxis(pheromone, log_count_ratio)
    return -phermone_infotaxis_value * math.log(pheromone) if phermone_infotaxis_value != 0 and pheromone != 0 else 0.0

def hybrid_select_action(actions: list, log_count_ratio: float, A: np.ndarray, C: np.ndarray) -> str:
    """Select an action based on the hybrid bandit router with the influence of the store factor and the log-count ratio."""
    best_action = None
    best_value = float('-inf')
    for action in actions:
        value = _hybrid_store_factor(action.action_id, _count(action.action_id), log_count_ratio, A, C) + _reward(action.action_id)
        if value > best_value:
            best_action = action.action_id
            best_value = value
    return best_action

def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step."""
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y

def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run SSM sequentially over a sequence."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    if h0 is None:
        h0 = np.zeros(state_dim)
    h = h0
    Y = np.zeros((T, C.shape[0]))
    for t, x in enumerate(x_seq):
        h, y = ssm_step(h, x, A, B, C)
        Y[t] = y
    return Y

def hybrid_ssm_bandit(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    actions: list,
    log_count_ratio: float,
) -> np.ndarray:
    """Run the hybrid SSM bandit algorithm."""
    Y = ssm_sequential(x_seq, A, B, C)
    actions_selected = []
    for t, y in enumerate(Y):
        action_id = hybrid_select_action(actions, log_count_ratio, A, C)
        actions_selected.append(action_id)
    return np.array(actions_selected)

if __name__ == "__main__":
    # Smoke test
    A = np.array([[0.5, 0.3], [0.2, 0.7]])
    B = np.array([[0.1, 0.2], [0.3, 0.4]])
    C = np.array([[0.5, 0.5], [0.3, 0.7]])
    x_seq = np.random.rand(10, 2)
    actions = [BanditAction(str(i), 0.5, 0.5, 0.5, "algorithm") for i in range(10)]
    log_count_ratio = 0.5
    hybrid_ssm_bandit(x_seq, A, B, C, actions, log_count_ratio)