# DARWIN HAMMER — match 407, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:28:49Z

"""
State-Space Duality and Hybrid Bandit Router Fusion

This module fuses the State-Space Duality (SSD) algorithm (state_space_duality.py) 
with the Hybrid Bandit Router (HBR) algorithm (hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py).
The mathematical bridge between the two algorithms lies in the use of 
the log-count ratio in HBR and the state-transition matrix in SSD.

The hybrid algorithm, called `hybrid_ssm_router`, integrates the 
governing equations of SSD with the decision-making process of HBR.
"""

import numpy as np
import math
from dataclasses import dataclass
from collections import defaultdict

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

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _ssm_step(
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

def hybrid_ssm_router(
    actions: list,
    log_count_ratio: float,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray,
    x_seq: np.ndarray,
) -> np.ndarray:
    """Run the hybrid SSM router over a sequence."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    Y = np.zeros((T, C.shape[0]))
    h = h0
    for t in range(T):
        best_action = None
        best_value = float('-inf')
        for action in actions:
            value = _hybrid_store_factor(action.action_id, _count(action.action_id), log_count_ratio) + _reward(action.action_id)
            if value > best_value:
                best_action = action
        # Update policy
        _POLICY[best_action.action_id][0] += 1.0
        _POLICY[best_action.action_id][1] += 1.0
        # SSM step
        h, y = _ssm_step(h, x_seq[t], A, B, C)
        Y[t] = y
    return Y

def verify_duality_hybrid(
    actions: list,
    log_count_ratio: float,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray,
    x_seq: np.ndarray,
) -> None:
    """Verify the duality of the hybrid SSM router."""
    Y_hybrid = hybrid_ssm_router(actions, log_count_ratio, A, B, C, h0, x_seq)
    # Compute the semiseparable matrix M
    M = np.zeros((x_seq.shape[0], x_seq.shape[0]))
    for i in range(x_seq.shape[0]):
        for j in range(i + 1):
            prod_A = np.eye(A.shape[0])
            for k in range(j + 1, i + 1):
                prod_A = A @ prod_A
            M[i, j] = C @ prod_A @ B
    Y_dual = M @ x_seq.T
    assert np.allclose(Y_hybrid, Y_dual.T)

if __name__ == "__main__":
    np.random.seed(0)
    actions = [BanditAction("action1", 0.5, 1.0, 0.1, "algorithm1")]
    log_count_ratio = 0.1
    A = np.array([[0.9]])
    B = np.array([[0.1]])
    C = np.array([[1.0]])
    h0 = np.array([0.0])
    x_seq = np.random.rand(10, 1)
    Y_hybrid = hybrid_ssm_router(actions, log_count_ratio, A, B, C, h0, x_seq)
    print(Y_hybrid)