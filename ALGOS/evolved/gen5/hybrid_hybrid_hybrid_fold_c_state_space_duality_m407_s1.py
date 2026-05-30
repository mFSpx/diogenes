# DARWIN HAMMER — match 407, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:28:49Z

"""
This module implements a novel hybrid algorithm that fuses the core topologies of 
two parent algorithms: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py and 
state_space_duality.py. The mathematical bridge between the two parents lies in the 
interface between the pheromone infotaxis and the state space duality. Specifically, 
the pheromone infotaxis can be viewed as a mechanism for updating the state transition 
matrix A in the state space duality, while the state space duality provides a framework 
for sequentially updating the state and output.

The hybrid algorithm integrates the governing equations of both parents, using the 
pheromone infotaxis to update the state transition matrix A, and the state space duality 
to sequentially update the state and output. This fusion enables the hybrid algorithm 
to leverage the strengths of both parents, combining the adaptability of the pheromone 
infotaxis with the efficiency of the state space duality.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

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

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

def _phermone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Compute the pheromone infotaxis."""
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

def _update_state_transition_matrix(A: np.ndarray, pheromone: float, log_count_ratio: float) -> np.ndarray:
    """Update the state transition matrix A using the pheromone infotaxis."""
    phermone_infotaxis_value = _phermone_infotaxis(pheromone, log_count_ratio)
    return A + phermone_infotaxis_value * np.eye(A.shape[0])

def _hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    pheromone: float,
    log_count_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single hybrid SSM step, using the pheromone infotaxis to update the state transition matrix A."""
    A_updated = _update_state_transition_matrix(A, pheromone, log_count_ratio)
    h_new = A_updated @ h + B @ x
    y = C @ h_new
    return h_new, y

def _hybrid_ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
    pheromone: float = 0.0,
    log_count_ratio: float = 0.0,
) -> np.ndarray:
    """Run hybrid SSM sequentially over a sequence, using the pheromone infotaxis to update the state transition matrix A."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros(state_dim) if h0 is None else h0
    Y = np.zeros((T, C.shape[0]))
    for t in range(T):
        h, y = _hybrid_ssm_step(h, x_seq[t], A, B, C, pheromone, log_count_ratio)
        Y[t] = y
    return Y

def _hybrid_bandit_select_action(actions: list, log_count_ratio: float, pheromone: float) -> str:
    """Select an action based on the hybrid bandit router with the influence of the store factor and the log-count ratio."""
    best_action = None
    best_value = float('-inf')
    for action in actions:
        value = _hybrid_store_factor(action.action_id, _count(action.action_id), log_count_ratio) + _reward(action.action_id)
        if value > best_value:
            best_action = action.action_id
    return best_action

if __name__ == "__main__":
    # Smoke test
    A = np.array([[0.5, 0.3], [0.2, 0.7]])
    B = np.array([[0.1, 0.4], [0.6, 0.8]])
    C = np.array([[0.3, 0.2], [0.1, 0.6]])
    x_seq = np.array([[0.5, 0.3], [0.2, 0.7], [0.1, 0.4]])
    pheromone = 0.5
    log_count_ratio = 0.2
    Y = _hybrid_ssm_sequential(x_seq, A, B, C, pheromone=pheromone, log_count_ratio=log_count_ratio)
    print(Y)