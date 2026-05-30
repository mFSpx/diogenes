# DARWIN HAMMER — match 11, survivor 4
# gen: 3
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:25:06Z

"""
Module for integrating physarum network flux-based conductance updates with a hybrid bandit router model.
The core mathematical bridge lies in mapping the bandit's propensity to inflow rates and confidence bounds to outflow rates,
then using these rates to update conductance in the physarum network.
This fusion enables adaptive, learning-based routing in the physarum network, influenced by the bandit's exploration-exploitation trade-offs.
Parent algorithms: physarum_network.py, hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def integrate_bandit_with_physarum(bandit_action: BanditAction, edge_length: float, pressure_a: float, pressure_b: float, conductance: float, eps: float = 1e-12) -> float:
    """Integrate bandit propensity and confidence bound with physarum flux-based conductance updates."""
    q = bandit_action.propensity - bandit_action.confidence_bound
    flux_value = flux(conductance, edge_length, pressure_a, pressure_b, eps)
    return update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05) + flux_value


def hybrid_bandit_router(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, bandit_action: BanditAction, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> Tuple[float, BanditAction]:
    """Update conductance using the hybrid bandit router model and return the updated bandit action."""
    q = bandit_action.propensity - bandit_action.confidence_bound
    updated_conductance = update_conductance(conductance, q, dt, gain, decay)
    updated_bandit_action = BanditAction(
        action_id=bandit_action.action_id,
        propensity=updated_conductance,
        expected_reward=bandit_action.expected_reward,
        confidence_bound=bandit_action.confidence_bound,
        algorithm=bandit_action.algorithm
    )
    return updated_conductance, updated_bandit_action


def simulate_hybrid_system(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, bandit_action: BanditAction, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05, num_steps: int = 10) -> List[Tuple[float, BanditAction]]:
    """Simulate the hybrid system for a specified number of steps."""
    trajectory = []
    for _ in range(num_steps):
        updated_conductance, updated_bandit_action = hybrid_bandit_router(conductance, edge_length, pressure_a, pressure_b, bandit_action, dt, gain, decay)
        trajectory.append((updated_conductance, updated_bandit_action))
        conductance = updated_conductance
        bandit_action = updated_bandit_action
    return trajectory


if __name__ == "__main__":
    bandit_action = BanditAction(
        action_id="action_1",
        propensity=0.5,
        expected_reward=0.8,
        confidence_bound=0.2,
        algorithm="hybrid_bandit_router"
    )
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 1.0
    pressure_b = 0.0
    updated_conductance = integrate_bandit_with_physarum(bandit_action, edge_length, pressure_a, pressure_b, conductance)
    print(f"Updated conductance: {updated_conductance}")
    trajectory = simulate_hybrid_system(conductance, edge_length, pressure_a, pressure_b, bandit_action)
    print(f"Trajectory: {trajectory}")