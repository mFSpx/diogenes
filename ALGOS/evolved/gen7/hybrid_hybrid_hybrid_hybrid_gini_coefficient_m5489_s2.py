# DARWIN HAMMER — match 5489, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_path_signatur_m1410_s0.py (gen6)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-30T00:02:20Z

"""
Hybrid Gini-Krampu Router

This module fuses two parent algorithms:
* **Parent A** – Hybrid LeadLag Signature Router: a kernelised contextual bandit 
  that combines Krampus brain-map vibes with radial-basis-function (RBF) surrogates 
  for contextual decision-making.
* **Parent B** – Gini Coefficient: a measure of the inequality or dispersion of a 
  distribution.

The mathematical bridge between the two parents lies in the use of the Gini 
coefficient to quantify the inequality of the reward distributions in the 
kernelised bandit. This allows the algorithm to adapt to changing reward 
distributions and make decisions based on both the semantic and geometric 
information of the path.

The implementation below provides:
* `lead_lag_transform` – the lead-lag transformation of a path.
* `signature_level1` and `signature_level2` – the level-1 and level-2 
  signatures of a path.
* `gaussian_kernel` – the RBF similarity used in the kernelised bandit.
* `gini_coefficient` – the Gini coefficient of a distribution.
* `HybridGiniKrampuRouter.update_policy` – stores contexts, actions, and rewards.
* `HybridGiniKrampuRouter.select_action` – computes kernel-weighted reward 
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
    algorithm: str = "HybridGiniKrampu"

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

def gaussian_kernel(x, y, sigma=1.0):
    return math.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def gini_coefficient(values):
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))

class HybridGiniKrampuRouter:
    def __init__(self):
        self.contexts = {}
        self.actions = {}

    def update_policy(self, context_id, action_id, reward, propensity, feature_vec):
        if context_id not in self.contexts:
            self.contexts[context_id] = []
        self.contexts[context_id].append((action_id, reward, propensity, feature_vec))

    def select_action(self, context_id, feature_vec):
        if context_id not in self.contexts:
            return BanditAction(action_id="random", propensity=0.0, expected_reward=0.0, confidence_bound=0.0)

        rewards = [r for _, r, _, _ in self.contexts[context_id]]
        gini = gini_coefficient(rewards)

        kernel_weights = [gaussian_kernel(feature_vec, fv) for _, _, _, fv in self.contexts[context_id]]
        kernel_weights = np.array(kernel_weights) / sum(kernel_weights)

        expected_rewards = np.array([r for _, r, _, _ in self.contexts[context_id]])
        expected_reward = np.sum(expected_rewards * kernel_weights)

        confidence_bound = np.sqrt(np.sum((expected_rewards - expected_reward) ** 2 * kernel_weights))

        return BanditAction(action_id="selected", propensity=gini, expected_reward=expected_reward, confidence_bound=confidence_bound)

def test_hybrid_gini_krampu_router():
    router = HybridGiniKrampuRouter()
    context_id = "test_context"
    action_id = "test_action"
    reward = 1.0
    propensity = 0.5
    feature_vec = np.array([1.0, 2.0])

    router.update_policy(context_id, action_id, reward, propensity, feature_vec)
    action = router.select_action(context_id, feature_vec)

    print(asdict(action))

if __name__ == "__main__":
    test_hybrid_gini_krampu_router()