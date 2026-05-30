# DARWIN HAMMER — match 1410, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s5.py (gen5)
# parent_b: hybrid_path_signature_kan_m30_s2.py (gen1)
# born: 2026-05-29T23:36:07Z

"""
Hybrid LeadLag Signature Router

This module fuses two parent algorithms:

* **Parent A** – hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s5.py: 
  a kernelised contextual bandit that combines Krampus brain-map vibes with 
  radial-basis-function (RBF) surrogates for contextual decision-making.
* **Parent B** – hybrid_path_signature_kan_m30_s2.py: 
  a path signature-based algorithm that extracts features from lead-lag 
  transformed paths using level-1 and level-2 signatures.

The mathematical bridge between the two parents lies in the treatment of the 
lead-lag transformed path as a context in the kernelised bandit. The level-1 
and level-2 signatures of the path are used as feature vectors in the bandit, 
enabling the algorithm to make decisions based on both the semantic and 
geometric information of the path.

The implementation below provides:
* `lead_lag_transform` – the lead-lag transformation of a path.
* `signature_level1` and `signature_level2` – the level-1 and level-2 
  signatures of a path.
* `gaussian_kernel` – the RBF similarity used in the kernelised bandit.
* `HybridLeadLagRouter.update_policy` – stores contexts, actions, and rewards.
* `HybridLeadLagRouter.select_action` – computes kernel-weighted reward 
  estimates and confidence bounds, returning a `BanditAction`.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridLeadLag"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    feature_vec

def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          
    running    = path[:-1] - path[0]            
    return running.T @ increments               

def gaussian_kernel(x, x_prime, epsilon=1.0):
    return math.exp(-epsilon**2 * np.linalg.norm(x - x_prime)**2)

class HybridLeadLagRouter:
    def __init__(self):
        self.contexts = []
        self.actions = []
        self.rewards = []
        self.propensities = []
        self.feature_vecs = []

    def update_policy(self, context, action, reward, propensity, feature_vec):
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)
        self.propensities.append(propensity)
        self.feature_vecs.append(feature_vec)

    def select_action(self, context, actions, epsilon=1.0, beta=1.0):
        action_values = {}
        for action in actions:
            numerator = 0
            denominator = 0
            for i in range(len(self.actions)):
                if self.actions[i] == action:
                    kernel = gaussian_kernel(context, self.feature_vecs[i], epsilon)
                    numerator += kernel * self.rewards[i]
                    denominator += kernel
            if denominator == 0:
                action_values[action] = 0
            else:
                action_values[action] = numerator / denominator + beta * math.sqrt(1 / denominator)
        best_action = max(action_values, key=action_values.get)
        propensity = action_values[best_action] / sum(action_values.values())
        expected_reward = action_values[best_action]
        confidence_bound = beta * math.sqrt(1 / sum([gaussian_kernel(context, self.feature_vecs[i], epsilon) for i in range(len(self.actions)) if self.actions[i] == best_action]))
        return BanditAction(best_action, propensity, expected_reward, confidence_bound)

    def extract_full_features(self, path):
        lead_lag_path = lead_lag_transform(path)
        level1_signature = signature_level1(lead_lag_path)
        level2_signature = signature_level2(lead_lag_path)
        return np.concatenate([level1_signature, level2_signature])

if __name__ == "__main__":
    router = HybridLeadLagRouter()
    path = np.random.rand(10, 2)
    feature_vec = router.extract_full_features(path)
    actions = ["action1", "action2"]
    router.update_policy("context1", "action1", 1.0, 0.5, feature_vec)
    action = router.select_action(feature_vec, actions)
    print(asdict(action))