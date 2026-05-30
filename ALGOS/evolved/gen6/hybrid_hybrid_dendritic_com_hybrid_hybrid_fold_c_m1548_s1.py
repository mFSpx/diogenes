# DARWIN HAMMER — match 1548, survivor 1
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s2.py (gen5)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_hard_t_m88_s0.py (gen4)
# born: 2026-05-29T23:37:14Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import defaultdict

"""
Hybrid Dendritic Fold-Change Regret-Weighted Ternary-Decision Analyzer

Parents
-------
* Parent A: Dendritic Compartment (Hodgkin-Huxley cable model) from `dendritic_compartment.py`
* Parent B: Hybrid Fold-Change Stylometric-Geometric Bandit from `hybrid_hybrid_fold_change_d_hybrid_hybrid_hard_t_m88_s0.py`

Mathematical Bridge
-------------------
Both parents operate on vectors in ℝⁿ: 
* The 3-D vector **h** = (x, y, V) ∈ ℝ³ from the dendritic compartment model is mapped onto a ternary alphabet using the regret-weighted probabilities. This ternary sequence is then used as input for the fold-change detection and bandit policy, effectively integrating the dendritic spikes with the temporal signal.

The module below implements this fusion, exposing three core hybrid functions: `hybrid_state_vector`, `assign_region`, and `hybrid_select_action`.
"""

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current."""
    return g_Na * m**3 * h * (V - E_Na)

def potassium_current(V, n, g_K=36.0, E_K=-77.0):
    """Hodgkin-Huxley potassium current."""
    return g_K * n**4 * (V - E_K)

def leak_current(V, g_L=0.3, E_L=-54.4):
    """Hodgkin-Huxley leak current."""
    return g_L * (V - E_L)

def calculate_regret_weighted_probabilities(actions: list[MathAction]) -> np.ndarray:
    """Calculate regret-weighted probabilities from a list of MathAction objects."""
    probabilities = np.zeros(len(actions))
    for i, action in enumerate(actions):
        probabilities[i] = action.risk / (action.risk + action.cost)
    return probabilities

def calculate_dendritic_spikes(V, m, h):
    """Calculate dendritic spikes from the Hodgkin-Huxley model."""
    sodium_current_value = sodium_current(V, m, h)
    potassium_current_value = potassium_current(V, m, h)
    leak_current_value = leak_current(V)
    return sodium_current_value + potassium_current_value - leak_current_value

def hybrid_state_vector(actions: list[MathAction], V, m, h) -> np.ndarray:
    """Combine the regret-weighted probabilities with the dendritic spikes."""
    probabilities = calculate_regret_weighted_probabilities(actions)
    dendritic_spikes = calculate_dendritic_spikes(V, m, h)
    ternary_sequence = np.where(probabilities > 0.5, 1, -1)
    return np.concatenate((probabilities, dendritic_spikes, ternary_sequence))

def assign_region(state_vector: np.ndarray) -> int:
    """Assign the state to a region using the Voronoi partition."""
    # Assuming a list of Voronoi seeds is available
    voronoi_seeds = np.array([[1, 2, 3], [4, 5, 6]])
    distances = np.linalg.norm(state_vector[:, np.newaxis] - voronoi_seeds, axis=2)
    return np.argmin(distances, axis=1)[0]

def hybrid_select_action(state_vector: np.ndarray, policy: dict[str, list[float]]) -> str:
    """Select an action based on the assigned region and the bandit policy."""
    region = assign_region(state_vector)
    actions = list(policy.keys())
    rewards = [policy[action][0] for action in actions]
    return actions[np.argmax(rewards)]

_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear the internal bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Average reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list[tuple[str, float]]) -> None:
    """Incorporate a batch of (action, reward) observations."""
    for action, reward in updates:
        total, n = _POLICY.get(action, [0.0, 0.0])
        _POLICY[action] = [total + reward, n + 1]

if __name__ == "__main__":
    # Smoke test
    actions = [MathAction('action1', 1.0, 0.5, 0.2), MathAction('action2', 2.0, 0.3, 0.1)]
    V = 10.0
    m = 0.5
    h = 0.7
    state_vector = hybrid_state_vector(actions, V, m, h)
    print(state_vector)
    region = assign_region(state_vector)
    print(region)
    policy = {'action1': [1.0, 1.0], 'action2': [2.0, 1.0]}
    selected_action = hybrid_select_action(state_vector, policy)
    print(selected_action)
    update_policy([('action1', 1.0), ('action2', 2.0)])
    print(_reward('action1'))
    print(_count('action1'))