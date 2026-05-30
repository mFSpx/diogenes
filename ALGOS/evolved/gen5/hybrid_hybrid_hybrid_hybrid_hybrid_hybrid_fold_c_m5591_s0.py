# DARWIN HAMMER — match 5591, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py (gen3)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py (gen4)
# born: 2026-05-30T00:03:12Z

"""
HybridSystem: Fusing hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py and hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py

This module integrates the rectified-flow schedule and morphology-driven priority metrics from 
hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py with the hybrid bandit router and 
pheromone infotaxis from hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py.

The mathematical bridge is established by replacing the target allocation a1 in the rectified-flow 
schedule with a value proportional to the pheromone infotaxis and hybrid store factor. This 
couples the morphology-derived priority with the log-count ratio and store factor, enabling 
simultaneous optimization of RAM allocation, endpoint health, and morphology-aware loading decisions.
"""

import math
import random
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass(frozen=True)
class Model:
    name: str
    ram_mb: int

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

class HybridSystem:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, Model] = {}
        self._POLICY: dict = {}

    def _hybrid_store_factor(self, action_id: str, count: float, log_count_ratio: float) -> float:
        return log_count_ratio * count

    def _phermone_infotaxis(self, pheromone: float, log_count_ratio: float) -> float:
        return pheromone * log_count_ratio

    def _rectified_flow(self, a0: float, a1: float, alpha: float) -> float:
        return (1 - alpha) * a0 + alpha * a1

    def _compute_priority(self, model: Model, pheromone: float, log_count_ratio: float) -> float:
        ram_fraction = model.ram_mb / self.ram_ceiling_mb
        infotaxis = self._phermone_infotaxis(pheromone, log_count_ratio)
        return infotaxis * (1 - ram_fraction)

    def hybrid_load_model(self, model: Model, pheromone: float, log_count_ratio: float) -> bool:
        priority = self._compute_priority(model, pheromone, log_count_ratio)
        a0 = 0.0
        a1 = priority * self.ram_ceiling_mb
        alpha = 1.0
        target_allocation = self._rectified_flow(a0, a1, alpha)
        if target_allocation <= self.ram_ceiling_mb:
            self.loaded[model.name] = model
            return True
        return False

    def hybrid_select_action(self, actions: List[BanditAction], log_count_ratio: float) -> str:
        best_action = None
        best_value = float('-inf')
        for action in actions:
            count = self._POLICY.get(action.action_id, 0.0)
            value = self._hybrid_store_factor(action.action_id, count, log_count_ratio) + action.expected_reward
            if value > best_value:
                best_value = value
                best_action = action
        self._POLICY[action.action_id] = self._POLICY.get(action.action_id, 0.0) + 1.0
        return best_action.action_id

if __name__ == "__main__":
    system = HybridSystem()
    model = Model("test_model", 1000)
    actions = [BanditAction("action1", 0.5, 10.0, 0.1, "algorithm1"), 
               BanditAction("action2", 0.3, 20.0, 0.2, "algorithm2")]
    pheromone = 0.8
    log_count_ratio = 0.2
    print(system.hybrid_load_model(model, pheromone, log_count_ratio))
    print(system.hybrid_select_action(actions, log_count_ratio))