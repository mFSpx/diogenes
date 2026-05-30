# DARWIN HAMMER — match 5591, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py (gen3)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py (gen4)
# born: 2026-05-30T00:03:12Z

"""
HybridModelFusion
Combines:
- hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py (gen3)
- hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py (gen4)

The mathematical bridge between the two parent algorithms lies in the integration of 
the rectified-flow schedule and the pheromone infotaxis. Specifically, we define 
a hybrid allocation function that combines the morphology-driven priority metrics 
from the first parent with the pheromone-infused decision hygiene from the second parent.

Let **r** ∈ ℝ⁺ be the RAM footprint of a model, **p(m)** ∈ [0,1] be the recovery 
priority of an endpoint, and **pheromone** ∈ ℝ⁺ be the pheromone concentration. 
We define a scalar field  

    f(endpoint, model) = p(m) · (1 – r / R_max) · _phermone_infotaxis(pheromone, log_count_ratio)

The hybrid system fuses the rectified-flow algorithm with the pheromone-infused 
decision hygiene, enabling simultaneous consideration of RAM limits, endpoint health, 
morphology-aware loading decisions, and pheromone-driven action selection.

The module implements three core hybrid operations:
1. `hybrid_allocation` – computes the hybrid allocation using the rectified-flow schedule 
   and pheromone infotaxis.
2. `hybrid_priority` – computes the morphology-driven priority metrics.
3. `hybrid_pheromone_decision` – selects an action based on the hybrid pheromone decision 
   hygiene.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List, Any

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

_POLICY: dict = {}

def reset_policy() -> None:
    """Reset the bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Compute the reward for an action based on the bandit policy."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Count the number of times an action has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Compute the hybrid store factor."""
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float) -> float:
    """Compute the fold-change detection."""
    return math.log(max(x / eps, 1))

def _phermone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    """Compute the pheromone infotaxis."""
    return pheromone * log_count_ratio

def _decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    """Compute the decision hygiene shannon entropy."""
    infotaxis = _phermone_infotaxis(pheromone, log_count_ratio)
    return -infotaxis * math.log(max(infotaxis, 1e-10))

class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

def linear_interpolant(a0: float, a1: float, alpha: float) -> float:
    """Generic rectified-flow interpolation."""
    return (1 - alpha) * a0 + alpha * a1

def hybrid_priority(model: ModelTier, R_max: int) -> float:
    """Compute morphology-driven priority metrics."""
    ram_footprint = model.ram_mb
    return (1 - ram_footprint / R_max)

def hybrid_pheromone_decision(actions: List[BanditAction], log_count_ratio: float, pheromone: float) -> str:
    """Select an action based on the hybrid pheromone decision hygiene."""
    best_action = None
    best_value = float('-inf')
    for action in actions:
        count = _count(action.action_id)
        value = _hybrid_store_factor(action.action_id, count, log_count_ratio) + _reward(action.action_id) + _phermone_infotaxis(pheromone, log_count_ratio)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action.action_id

def hybrid_allocation(model: ModelTier, R_max: int, pheromone: float, log_count_ratio: float) -> float:
    """Compute hybrid allocation using rectified-flow schedule and pheromone infotaxis."""
    priority = hybrid_priority(model, R_max)
    a0 = 0.0
    a1 = priority * (1 - model.ram_mb / R_max) * _phermone_infotaxis(pheromone, log_count_ratio)
    alpha = 0.5
    return linear_interpolant(a0, a1, alpha)

if __name__ == "__main__":
    model = ModelTier("test_model", 1000, "test_tier")
    R_max = 6000
    pheromone = 1.0
    log_count_ratio = 1.0
    actions = [BanditAction("action1", 1.0, 1.0, 1.0, "algorithm1")]
    print(hybrid_allocation(model, R_max, pheromone, log_count_ratio))
    print(hybrid_pheromone_decision(actions, log_count_ratio, pheromone))