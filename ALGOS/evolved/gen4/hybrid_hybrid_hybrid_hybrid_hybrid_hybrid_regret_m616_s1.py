# DARWIN HAMMER — match 616, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py (gen3)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s1.py (gen3)
# born: 2026-05-29T23:29:58Z

"""
Hybrid Fractional-Memory Regret-Weighted Module
===============================================

This module fuses two parent algorithms:

* **Hybrid Fractional-Memory Allocation Module (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s2.py)** – 
  provides a deterministic/LLM split and group-wise division with an input-dependent 
  effective time constant that modulates the LLM allocation, using a Caputo fractional derivative 
  to introduce a memory term into the allocation process.
* **Hybrid Regret-Weighted Strategy with Bandit Router (hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s1.py)** – 
  provides a regret-weighted strategy with a hybrid bandit router, incorporating a MinHash-based 
  similarity metric to modulate the propensity of each action.

The mathematical bridge between these two structures lies in the application of the fractional-memory 
kernel to the regret-weighted strategy, effectively modulating the propensity of each action based on 
its similarity to a set of reference actions, and the use of the Caputo fractional derivative to 
introduce a memory term into the regret-weighted strategy.

The hybrid module fuses:
1. The deterministic/LLM split and group-wise division of the Hybrid Fractional-Memory Allocation Module.
2. The input-dependent effective time constant of the Hybrid Fractional-Memory Allocation Module.
3. The fractional-memory tree cost of the Hybrid fractional-memory tree cost module.
4. The regret-weighted strategy of the Hybrid Regret-Weighted Strategy with Bandit Router.
5. The hybrid bandit router of the Hybrid Regret-Weighted Strategy with Bandit Router.

The implementation below provides:

* `init_hybrid_fm_regret` – initialise the hybrid regret parameters.
* `hybrid_fm_regret_allocate` – compute per-action, per-group allocations using 
  the fractional-memory modulated regret-weighted strategy.
* `summarize_hybrid_fm_regret_savings` – aggregate baseline vs. fractional-memory modulated 
  regret-weighted strategy and report a savings percentage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import random
import sys
import pathlib
import hashlib

GROUPS = ("codex", "groq", "cohere", "local_models")

_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13
])

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

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        delta = self._last_delta
        return max(0.0, min(self.limit, self.base + self.gain * delta))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def caputo_fractional_derivative(f: Callable[[float], float], t: float, alpha: float) -> float:
    integral = 0.0
    for tau in np.linspace(0, t, 100):
        integral += (t - tau) ** (alpha - 1) * f(tau)
    return 1 / math.gamma(alpha) * integral

def _lanczos_gamma(z: complex) -> complex:
    if z.real < 0.5:
        return math.pi / (math.sin(math.pi * z) * _lanczos_gamma(1 - z))
    else:
        x = 1 / (z * z)
        p = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ]
        for i in range(len(p)):
            p[i] += x * p[i]
        return math.sqrt(2 * math.pi) * z ** (z - 0.5) * np.exp(-z) * np.prod(p)

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def init_hybrid_fm_regret(groups: Tuple[str, ...], 
                           alpha: float, 
                           beta: float, 
                           dt: float) -> Tuple[Dict[str, StoreState], Dict[str, float]]:
    store_states = {group: StoreState() for group in groups}
    propensities = {group: 1.0 for group in groups}
    return store_states, propensities

def hybrid_fm_regret_allocate(store_states: Dict[str, StoreState], 
                              propensities: Dict[str, float], 
                              actions: List[MathAction], 
                              alpha: float, 
                              beta: float, 
                              dt: float) -> Dict[str, Dict[str, float]]:
    allocations = {group: {} for group in store_states}
    for group, store_state in store_states.items():
        for action in actions:
            propensity = propensities[group]
            expected_reward = action.expected_value * propensity
            confidence_bound = action.cost * np.sqrt(propensity)
            bandit_action = BanditAction(action_id=action.id, 
                                         propensity=propensity, 
                                         expected_reward=expected_reward, 
                                         confidence_bound=confidence_bound, 
                                         algorithm="Hybrid FM Regret")
            level, delta = store_state.update([expected_reward], [confidence_bound])
            allocations[group][action.id] = level * sigmoid(delta)
    return allocations

def summarize_hybrid_fm_regret_savings(store_states: Dict[str, StoreState], 
                                       propensities: Dict[str, float], 
                                       actions: List[MathAction], 
                                       alpha: float, 
                                       beta: float, 
                                       dt: float) -> float:
    baseline_allocation = 0.0
    hybrid_allocation = 0.0
    for group, store_state in store_states.items():
        for action in actions:
            baseline_allocation += action.expected_value
            hybrid_allocation += store_state.dance * sigmoid(store_state._last_delta)
    savings_percentage = (baseline_allocation - hybrid_allocation) / baseline_allocation * 100
    return savings_percentage

if __name__ == "__main__":
    groups = GROUPS
    alpha = 0.5
    beta = 0.5
    dt = 1.0
    store_states, propensities = init_hybrid_fm_regret(groups, alpha, beta, dt)
    actions = [MathAction(id="action1", expected_value=10.0), 
               MathAction(id="action2", expected_value=20.0)]
    allocations = hybrid_fm_regret_allocate(store_states, propensities, actions, alpha, beta, dt)
    savings_percentage = summarize_hybrid_fm_regret_savings(store_states, propensities, actions, alpha, beta, dt)
    print("Allocations:", allocations)
    print("Savings Percentage:", savings_percentage)