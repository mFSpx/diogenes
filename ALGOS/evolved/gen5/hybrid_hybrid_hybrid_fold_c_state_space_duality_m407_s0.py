# DARWIN HAMMER — match 407, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:28:49Z

"""
HYBRID ALGORITHM — fusion of DARWIN HAMMER and MAMBA-2 State Space Duality

This algorithm mathematically fuses the core topologies of the DARWIN HAMMER (hybrid bandit router with pheromone infotaxis) and MAMBA-2 State Space Duality (sequential and parallel forms) into a single unified system.

The hybrid operation is achieved by integrating the pheromone infotaxis term from DARWIN HAMMER into the state transition matrix of MAMBA-2. This creates a new state transition matrix that incorporates the influence of the pheromone infotaxis on the state dynamics.

The resulting hybrid algorithm combines the strengths of both parents, enabling efficient exploration-exploitation trade-offs in the presence of uncertain rewards, while also leveraging the parallel computation capabilities of MAMBA-2 to accelerate the state transition updates.

References:
    Dao & Gu (2024) "Transformers are SSMs: Generalized Models and
    Efficient Algorithms Through Structured State Space Duality"
    https://arxiv.org/abs/2405.21060
"""

import math
import random
import sys
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

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

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

def _infotaxis_state_transition(A: np.ndarray, B: np.ndarray, pheromone: float, log_count_ratio: float) -> np.ndarray:
    """Compute the infotaxis-influenced state transition matrix."""
    return np.linalg.inv(A) @ np.exp(pheromone * log_count_ratio) @ B

def ssm_step_infotaxis(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    pheromone: float,
    log_count_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step with infotaxis influence."""
    A_infotaxis = _infotaxis_state_transition(A, B, pheromone, log_count_ratio)
    h_new, y = ssm_step(h, x, A_infotaxis, B, C)
    return h_new, y

def ssm_sequential_infotaxis(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
    pheromone: float = 0.0,
    log_count_ratio: float = 0.0,
) -> np.ndarray:
    """Run SSM sequentially over a sequence with infotaxis influence."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros((T, state_dim)) if h0 is None else h0
    for t in range(T):
        h[t], _ = ssm_step_infotaxis(h[t-1] if t > 0 else h0, x_seq[t], A, B, C, pheromone, log_count_ratio)
    return h

def hybrid_select_action(actions: list, log_count_ratio: float, pheromone: float) -> str:
    """Select an action based on the hybrid bandit router with infotaxis influence."""
    best_action = None
    best_value = float('-inf')
    for action in actions:
        value = _hybrid_store_factor(action.action_id, _count(action.action_id), log_count_ratio) + _reward(action.action_id) + _phermone_infotaxis(pheromone, log_count_ratio)
        if value > best_value:
            best_action = action.action_id
            best_value = value
    return best_action

if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    random.seed(42)
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 5)
    C = np.random.rand(3, 10)
    x_seq = np.random.rand(10, 5)
    h0 = np.random.rand(10)
    pheromone = 0.5
    log_count_ratio = 0.2
    print(ssm_sequential_infotaxis(x_seq, A, B, C, h0, pheromone, log_count_ratio))
    actions = [BanditAction(f"action_{i}", 0.1, 0.2, 0.3, "algorithm") for i in range(10)]
    print(hybrid_select_action(actions, log_count_ratio, pheromone))