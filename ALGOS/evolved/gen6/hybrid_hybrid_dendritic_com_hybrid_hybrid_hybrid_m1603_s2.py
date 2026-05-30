# DARWIN HAMMER — match 1603, survivor 2
# gen: 6
# parent_a: hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py (gen5)
# born: 2026-05-29T23:37:44Z

"""
Hybrid Dendritic-Bandit Scheduler

This module fuses two distinct parents:

* **Parent A – hybrid_dendritic_compartmen_hybrid_hybrid_hybrid_m991_s4.py**  
  Provides Hodgkin-Huxley multi-compartment ODEs for a dendritic tree and a regret-weighted probabilities mechanism.

* **Parent B – hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m589_s0.py**  
  Supplies a Fisher-information based angle estimator and a contextual multi-armed bandit.

The mathematical bridge is the **regret-weighted bandit propensity**, which combines the regret-weighted probabilities from the dendritic tree with the bandit propensities from the Fisher-bandit scheduler.

The regret-weighted probabilities are used to scale the bandit propensities, effectively creating a hybrid scheduler that balances exploration and exploitation in the dendritic tree.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float

# ----------------------------------------------------------------------
# Hodgkin-Huxley utilities (from Parent A)
# ----------------------------------------------------------------------
def alpha_m(V: float) -> float:
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))

def beta_m(V: float) -> float:
    return 4.0 * math.exp(-(V + 65.0) / 18.0)

def alpha_h(V: float) -> float:
    return 0.07 * math.exp(-(V + 65.0) / 20.0)

def beta_h(V: float) -> float:
    return 1.0 / (1.0 + math.exp(-(V + 34.0) / 5.0))

# ----------------------------------------------------------------------
# Fisher information utilities (from Parent B)
# ----------------------------------------------------------------------
def fisher_information(features: np.ndarray, angles: np.ndarray, importance: np.ndarray) -> np.ndarray:
    fisher_info = np.zeros(features.shape[1])
    for i in range(features.shape[0]):
        fisher_info += importance[i] * np.dot(features[i], angles[i])
    return fisher_info

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_select_action(math_actions: List[MathAction], bandit_actions: List[BanditAction], fisher_info: np.ndarray) -> BanditAction:
    regret_weights = [action.expected_value for action in math_actions]
    scaled_propensities = [bandit_action.propensity * regret_weights[i] for i, bandit_action in enumerate(bandit_actions)]
    scaled_propensities = np.array(scaled_propensities) / sum(scaled_propensities)
    fisher_scaled_propensities = scaled_propensities * fisher_info
    return bandit_actions[np.argmax(fisher_scaled_propensities)]

def hybrid_update_math_actions(math_actions: List[MathAction], regret_weights: List[float]) -> List[MathAction]:
    updated_math_actions = []
    for i, math_action in enumerate(math_actions):
        updated_math_action = MathAction(math_action.id, math_action.expected_value * regret_weights[i])
        updated_math_actions.append(updated_math_action)
    return updated_math_actions

def hybrid_update_bandit_actions(bandit_actions: List[BanditAction], fisher_info: np.ndarray) -> List[BanditAction]:
    updated_bandit_actions = []
    for bandit_action in bandit_actions:
        updated_bandit_action = BanditAction(bandit_action.action_id, bandit_action.propensity * fisher_info[0])
        updated_bandit_actions.append(updated_bandit_action)
    return updated_bandit_actions

if __name__ == "__main__":
    math_actions = [MathAction("action1", 0.5), MathAction("action2", 0.3)]
    bandit_actions = [BanditAction("action1", 0.4), BanditAction("action2", 0.6)]
    features = np.array([[1, 2], [3, 4]])
    angles = np.array([[0.1, 0.2], [0.3, 0.4]])
    importance = np.array([0.5, 0.5])
    fisher_info = fisher_information(features, angles, importance)
    selected_action = hybrid_select_action(math_actions, bandit_actions, fisher_info)
    print(selected_action.action_id)
    updated_math_actions = hybrid_update_math_actions(math_actions, [0.7, 0.3])
    print(updated_math_actions[0].expected_value)
    updated_bandit_actions = hybrid_update_bandit_actions(bandit_actions, fisher_info)
    print(updated_bandit_actions[0].propensity)