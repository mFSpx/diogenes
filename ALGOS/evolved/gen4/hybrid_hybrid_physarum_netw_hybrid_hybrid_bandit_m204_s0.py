# DARWIN HAMMER — match 204, survivor 0
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s6.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s0.py (gen2)
# born: 2026-05-29T23:27:28Z

"""
Hybrid Physarum-Bandit-TTT Model
================================

This module fuses the Physarum conductance dynamics with the contextual bandit 
and TTT-Linear model. The mathematical bridge is the identification of 
**conductance** with the **propensity** (inflow rate) of a bandit action and 
**flux** with the **reward signal** that drives the bandit's learning update. 
The TTT-Linear model's output is used to update the bandit's policy and the 
conductance update rule is derived from the Physarum model.

*   Physarum: `q = G / L * (p_a - p_b)`  (flux on an edge)
*   Bandit   : `propensity` is the probability-like inflow for an action, 
    `confidence_bound` acts as an outflow.
*   TTT-Linear: The model's output is used to update the bandit's policy.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

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

_POLICY: dict[str, list[float]] = {}
_STORE: dict[str, float] = {}

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def flux(
    conductance: float,
    edge_length: float,
    pressure_a: float,
    pressure_b: float,
    eps: float = 1e-12,
) -> float:
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(
    conductance: float,
    q: float,
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> float:
    return conductance + gain * q * dt - decay * conductance * dt

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_in, d_out)) * scale

def select_action(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> BanditAction:
    if not actions:
        raise ValueError('actions required')
    rng = random.Random(seed)
    if algorithm == 'epsilon_greedy' and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == 'thompson':
        chosen = max(actions, key=lambda a: rng.betavariate(1 + max(0, _reward(a)), 1 + max(0, 1 - _reward(a))))
    else:
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        chosen = max(actions, key=lambda a: _reward(a) + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]))
    return BanditAction(chosen, 1.0 / len(actions), _reward(chosen), 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1]), algorithm)

def hybrid_update(context: dict[str, float], actions: list[str], algorithm: str = 'linucb', epsilon: float = 0.1, seed: int | str | None = 7) -> tuple[BanditAction, float]:
    action = select_action(context, actions, algorithm, epsilon, seed)
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 1.0
    pressure_b = 0.0
    flux_value = flux(conductance, edge_length, pressure_a, pressure_b)
    updated_conductance = update_conductance(conductance, flux_value)
    ttt_output = np.array([updated_conductance])
    return action, ttt_output[0]

def main() -> None:
    reset_policy()
    context = {'feature1': 1.0, 'feature2': 2.0}
    actions = ['action1', 'action2']
    action, ttt_output = hybrid_update(context, actions)
    print(f"Selected action: {action.action_id}, TTT output: {ttt_output}")

if __name__ == "__main__":
    main()