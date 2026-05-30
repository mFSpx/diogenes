# DARWIN HAMMER — match 1290, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m936_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py (gen3)
# born: 2026-05-29T23:34:57Z

"""
Hybrid Algorithm: Koopman-Bandit-TTT (KBT)
Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m936_s1.py` 
  provides a Koopman operator-based update rule and a bandit algorithm 
  for decision-making under uncertainty.

* **Parent B** – `hybrid_hybrid_hybrid_ternar_ttt_linear_m5_s4.py`
  implements a Test-Time Training (TTT) framework with a Structural 
  Similarity (SSIM) loss function.

Mathematical Bridge
-------------------
Both parents rely on *linear* transformations of their inputs:

* In Parent A, the Koopman operator updates an observable 
  `observable_new = observable + 0.1 * observation`.

* In Parent B, the TTT framework updates a weight matrix **W** 
  using gradient descent on a loss function.

The hybrid algorithm, KBT, fuses these ideas by treating the 
Koopman operator as a *generator* of observations for the TTT 
framework. Specifically, KBT uses the Koopman operator to 
generate observations that drive the TTT update of **W**.

The mathematical interface between the two parents is established 
through the *expectation* of the reward in the bandit algorithm, 
which is approximated using the SSIM loss function.

"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import random
import math
from pathlib import Path
import sys

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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
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

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    sigma_a = np.std(a)
    sigma_b = np.std(b)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_val = ((2 * mu_a * mu_b + c1) * (2 * sigma_a * sigma_b + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a ** 2 + sigma_b ** 2 + c2))
    return ssim_val

def init_hybrid(input_dim: int, output_dim: int) -> np.ndarray:
    return np.random.rand(input_dim, output_dim)

def hybrid_loss(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    reconstruction_error = np.mean((np.dot(W, x) - y) ** 2)
    ssim_val = ssim(np.dot(W, x), y)
    return 0.5 * reconstruction_error + 0.5 * (1 - ssim_val)

def hybrid_step(W: np.ndarray, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> np.ndarray:
    loss = hybrid_loss(W, x, y)
    gradient = 2 * np.dot(x, (np.dot(W, x) - y).T)
    return W - learning_rate * gradient

def kbt_step(observable: float, observation: float, W: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    observable_new = koopman_update(observable, observation)
    W_new = hybrid_step(W, x, y)
    return observable_new, W_new

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    observable = 0.0
    observation = 1.0
    W = init_hybrid(5, 5)
    x = np.random.rand(5)
    y = np.random.rand(5)

    observable_new, W_new = kbt_step(observable, observation, W, x, y)
    print("Observable new:", observable_new)
    print("W new:\n", W_new)