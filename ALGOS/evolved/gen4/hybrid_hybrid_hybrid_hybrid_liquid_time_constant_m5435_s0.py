# DARWIN HAMMER — match 5435, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# parent_b: liquid_time_constant.py (gen0)
# born: 2026-05-30T00:01:51Z

"""
Module for the hybrid algorithm that combines the bandit router from 
hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py and the 
liquid time-constant networks from liquid_time_constant.py. The 
mathematical bridge between these two structures is the use of the 
input-dependent time constant in the liquid time-constant networks, 
which can be applied to the bandit router's action selection process 
using the Voronoi partition. This allows the bandit router to consider 
the geometric relationships between actions and contexts, and adapt its 
temporal dynamics based on the current input.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Bandit core
# ----------------------------------------------------------------------
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
_STORE: dict[str, float] = {}  # virtual VRAM store per key
DEFAULT_BUDGET_MB = 1024 * 4  # Assuming 4GB as default budget


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
Point = tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Liquid Time-Constant Networks
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid σ(x) = 1 / (1 + exp(-x))."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Network function f(x, I, t, θ) = σ(W @ [x; I] + b)."""
    return sigmoid(np.dot(W, np.concatenate((x, I))) + b)


def ltc_step(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    """Liquid time-constant step."""
    f_x = ltc_f(x, I, W, b)
    return - (1 / tau + f_x) * x + f_x


# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
class HybridBanditRouter:
    def __init__(self, num_actions: int, num_contexts: int, hidden_dim: int):
        self.num_actions = num_actions
        self.num_contexts = num_contexts
        self.hidden_dim = hidden_dim
        self.W = np.random.rand(hidden_dim, hidden_dim + num_contexts)
        self.b = np.zeros((hidden_dim,))
        self.x = np.zeros((hidden_dim,))
        self.tau = 1.0

    def select_action(self, context: np.ndarray) -> int:
        """Select an action based on the current context and hidden state."""
        I = np.array([context])
        f_x = ltc_f(self.x, I, self.W, self.b)
        self.x = ltc_step(self.x, I, self.W, self.b, self.tau)
        return np.argmax(f_x)

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update the hidden state and parameters based on the reward."""
        I = np.array([context])
        self.x = ltc_step(self.x, I, self.W, self.b, self.tau)
        self.W += np.outer(self.x, I[0])
        self.b += self.x


def hybrid_bandit_router_example() -> None:
    """Example usage of the hybrid bandit router."""
    num_actions = 5
    num_contexts = 10
    hidden_dim = 20
    router = HybridBanditRouter(num_actions, num_contexts, hidden_dim)
    for _ in range(100):
        context = np.random.rand(num_contexts)
        action = router.select_action(context)
        reward = np.random.rand()
        router.update(context, action, reward)


if __name__ == "__main__":
    hybrid_bandit_router_example()