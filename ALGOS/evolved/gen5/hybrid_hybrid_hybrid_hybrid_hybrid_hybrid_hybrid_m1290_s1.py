# DARWIN HAMMER — match 1290, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m936_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# born: 2026-05-29T23:34:57Z

"""
Hybrid Algorithm: Koopman-TTT (K-TTT)
Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m936_s1.py`
  provides a Koopman operator update rule and a bandit policy update mechanism.
* **Parent B** – `hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py`
  implements Test-Time Training (TTT) with a Structural Similarity (SSIM) guided loss.

Mathematical Bridge
-------------------
The mathematical bridge between the two parents lies in the use of a 
scalar loss function to drive updates of parameter matrices. In Parent A, 
the Koopman operator update rule can be seen as a special case of 
gradient descent on a loss function. In Parent B, the SSIM loss is used 
to guide the TTT update. We fuse these two ideas by using the SSIM loss 
to guide the Koopman operator update.

The hybrid algorithm, K-TTT, treats the Koopman operator as a weight 
matrix that is updated using a combination of the reward signal from the 
bandit policy and the SSIM loss. This allows the algorithm to balance 
exploration and exploitation while enforcing structural similarity 
between the input and output signals.
"""

import numpy as np
import math
from typing import List, Dict, Tuple
import random
import sys
from pathlib import Path

class BanditAction:
    def __init__(self, action_id: str, propensity: float, expected_reward: float, confidence_bound: float, algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm

class BanditUpdate:
    def __init__(self, context_id: str, action_id: str, reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    global _POLICY
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def koopman_update(observable: float, observation: float) -> float:
    return observable + 0.1 * observation

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
    return ssim_val

def hybrid_update(hypothesis: np.ndarray, evidence: np.ndarray, observation: np.ndarray) -> np.ndarray:
    likelihood_ratio = math.exp(-0.5 * np.dot(evidence.T, np.dot(hypothesis, hypothesis.T)) - 0.5 * np.dot(observation.T, np.dot(hypothesis, hypothesis.T)))
    var = np.linalg.inv(np.dot(hypothesis.T, hypothesis) + np.eye(len(hypothesis)))
    return update_belief_mean(hypothesis, observation, var)

def update_belief_mean(mean: np.ndarray, observation: np.ndarray, var: np.ndarray) -> np.ndarray:
    return mean + np.dot(var, observation)

def k_ttt_update(W: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    l_rec = np.mean((W @ x - y) ** 2)
    l_ssim = 1 - ssim(x, y)
    loss = alpha * l_rec + beta * l_ssim
    grad = 2 * (W @ x - y) @ x.T + beta * (l_ssim / ssim(x, y)) * (2 * (W @ x - y) @ x.T)
    return W - 0.1 * grad

def k_ttt_forward(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return W @ x

def route_with_k_ttt(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    output = k_ttt_forward(W, x)
    ssim_val = ssim(x, output)
    return output, ssim_val

if __name__ == "__main__":
    np.random.seed(0)
    W = np.random.rand(3, 3)
    x = np.random.rand(3)
    y = np.random.rand(3)
    alpha = 0.1
    beta = 0.1
    for _ in range(10):
        W = k_ttt_update(W, x, y, alpha, beta)
        output, ssim_val = route_with_k_ttt(W, x, y)
        print(f"SSIM: {ssim_val:.4f}")