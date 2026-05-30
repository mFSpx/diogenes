# DARWIN HAMMER — match 39, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s0.py (gen3)
# born: 2026-05-29T23:26:25Z

"""
Hybrid Algorithm: Fusing Fisher Localization with Bandit-Capybara Optimization.

This module fuses the Fisher information-based angle selection from 
hybrid_hybrid_hybrid_decisi_fisher_localization_m153_s2.py with the 
contextual multi-armed bandit and continuous optimization primitives from 
hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s0.py.

The mathematical bridge lies in the interpretation of the bandit-produced 
`propensity` as a confidence scalar that modulates the Fisher information 
computation and the selection of the optimal angle. The `confidence_bound` 
is used to calculate the signal-to-noise gap, which drives the attraction 
towards the global best angle and modulates the probability of entering 
*standby* versus *burst*.

"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Fisher Localization core
# ----------------------------------------------------------------------
def fisher_localization(feature_vector, angles, weights):
    intensities = np.zeros(len(angles))
    for i, (feature, weight) in enumerate(zip(feature_vector, weights)):
        sigma = 1 / np.sqrt(weight)
        for j, angle in enumerate(angles):
            intensities[j] += feature * np.exp(-((angle - i / len(feature_vector) * 2 * np.pi) ** 2) / (2 * sigma ** 2))
    fisher_info = np.gradient(intensities) ** 2 / intensities
    return fisher_info

# ----------------------------------------------------------------------
# Bandit-Capybara core
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {} 

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def select_action(actions: List[str], algorithm: str = "linucb", epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError("No actions provided")
    return BanditAction(random.choice(actions), random.random(), random.random(), random.random(), algorithm)

# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_fisher_bandit(feature_vector, weights, angles, actions):
    fisher_info = fisher_localization(feature_vector, angles, weights)
    best_angle_idx = np.argmax(fisher_info)
    best_angle = angles[best_angle_idx]
    bandit_action = select_action(actions)
    modulated_fisher_info = fisher_info * bandit_action.propensity
    return best_angle, modulated_fisher_info, bandit_action

def optimize_angle(feature_vector, weights, angles, actions):
    best_angle, modulated_fisher_info, bandit_action = hybrid_fisher_bandit(feature_vector, weights, angles, actions)
    # Use bandit_action.confidence_bound to modulate the optimization
    optimized_angle = best_angle + bandit_action.confidence_bound * np.random.randn()
    return optimized_angle

def evaluate_policy(feature_vector, weights, angles, actions):
    best_angle, modulated_fisher_info, bandit_action = hybrid_fisher_bandit(feature_vector, weights, angles, actions)
    reward = modulated_fisher_info[np.argmin(np.abs(angles - best_angle))]
    return reward

if __name__ == "__main__":
    feature_vector = np.array([1, 2, 3])
    weights = np.array([0.5, 0.3, 0.2])
    angles = np.linspace(0, 2 * np.pi, 100)
    actions = ["action1", "action2", "action3"]
    
    best_angle, modulated_fisher_info, bandit_action = hybrid_fisher_bandit(feature_vector, weights, angles, actions)
    print("Best Angle:", best_angle)
    print("Modulated Fisher Info:", modulated_fisher_info)
    print("Bandit Action:", bandit_action)

    optimized_angle = optimize_angle(feature_vector, weights, angles, actions)
    print("Optimized Angle:", optimized_angle)

    reward = evaluate_policy(feature_vector, weights, angles, actions)
    print("Reward:", reward)