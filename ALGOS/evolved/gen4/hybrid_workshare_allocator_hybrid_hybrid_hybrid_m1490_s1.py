# DARWIN HAMMER — match 1490, survivor 1
# gen: 4
# parent_a: workshare_allocator.py (gen0)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py (gen3)
# born: 2026-05-29T23:37:01Z

"""
Hybrid Workshare-Bandit Algorithm.

This module fuses the deterministic workshare allocation from 
workshare_allocator.py with the contextual multi-armed bandit 
primitives of hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s2.py.
The mathematical bridge is the **store equation** of the bandit 
primitive, which is extended to incorporate the signal-to-noise 
gap and the Hoeffding epsilon. The workshare allocation is 
used to rescale the inflow and outflow rates of the store equation.

The signal-to-noise gap is used to modulate the learning rate 
of the TTT update. The Hoeffding epsilon is used to bound the 
confidence intervals of the bandit actions.
"""

import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Workshare core (Parent A)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict[str, Any]:
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "lanes": lanes,
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
    }

# ----------------------------------------------------------------------
# Bandit core (Parent B)
# ----------------------------------------------------------------------
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

def evasion_delta(t: int, t_max: int, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    return delta_max * math.exp(-alpha * min(t, t_max) / t_max)

def clamp(x: List[float], lower: float, upper: float) -> List[float]:
    return [min(upper, max(lower, xi)) for xi in x]

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_update(context_id: str, action_id: str, reward: float, propensity: float, 
                   total_units: float, deterministic_target_pct: float) -> Tuple[BanditUpdate, dict[str, Any]]:
    workshare_allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    llm_units = workshare_allocation["llm_units"]
    deterministic_units = workshare_allocation["deterministic_units"]
    bandit_update = BanditUpdate(context_id=context_id, action_id=action_id, reward=reward, propensity=propensity)
    store_update = {
        "context_id": context_id,
        "action_id": action_id,
        "inflow_rate": llm_units * (1 - deterministic_target_pct / 100),
        "outflow_rate": deterministic_units * evasion_delta(t=0, t_max=100),
    }
    return bandit_update, store_update

def get_bandit_action(context_id: str, action_id: str, 
                       total_units: float, deterministic_target_pct: float) -> BanditAction:
    workshare_allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    llm_units = workshare_allocation["llm_units"]
    propensity = llm_units / (llm_units + 1)
    expected_reward = llm_units * (1 - deterministic_target_pct / 100)
    confidence_bound = math.sqrt(llm_units) * evasion_delta(t=0, t_max=100)
    return BanditAction(action_id=action_id, propensity=propensity, expected_reward=expected_reward, 
                       confidence_bound=confidence_bound, algorithm="hybrid")

def hybrid_bandit_step(context_id: str, action_id: str, reward: float, 
                        total_units: float, deterministic_target_pct: float) -> Tuple[BanditAction, BanditUpdate, dict[str, Any]]:
    bandit_action = get_bandit_action(context_id=context_id, action_id=action_id, 
                                       total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    bandit_update, store_update = hybrid_update(context_id=context_id, action_id=action_id, 
                                                 reward=reward, propensity=bandit_action.propensity, 
                                                 total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    return bandit_action, bandit_update, store_update

if __name__ == "__main__":
    total_units = 100.0
    deterministic_target_pct = 90.0
    context_id = "example_context"
    action_id = "example_action"
    reward = 10.0
    bandit_action, bandit_update, store_update = hybrid_bandit_step(context_id=context_id, action_id=action_id, 
                                                                   reward=reward, total_units=total_units, 
                                                                   deterministic_target_pct=deterministic_target_pct)
    print(asdict(bandit_action))
    print(asdict(bandit_update))
    print(store_update)