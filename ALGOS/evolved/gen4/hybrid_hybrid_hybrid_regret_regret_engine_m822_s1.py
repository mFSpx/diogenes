# DARWIN HAMMER — match 822, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s2.py (gen3)
# parent_b: regret_engine.py (gen0)
# born: 2026-05-29T23:31:07Z

"""
This module integrates the Regret-Weighted Strategy from regret_engine.py 
with the Hybrid Bandit Router from hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s2.py.
The mathematical bridge between these two structures lies in the application of the regret-weighted 
strategy to modulate the propensity scores in the Hybrid Bandit Router. Specifically, we use the 
regret-weighted strategy to re-weight the expected rewards in the bandit router, allowing it to 
consider both the expected reward and the regret associated with each action.

The governing equations of the regret-weighted strategy are used to update the propensity scores 
in the bandit router, creating a hybrid algorithm that leverages the strengths of both parents.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import random
import sys
import pathlib

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
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> dict[str,float]:
    if not actions: return {}
    cf={c.action_id:c.outcome_value*c.probability for c in counterfactuals}
    vals={a.id:a.expected_value-a.cost-a.risk+cf.get(a.id,0.0) for a in actions}
    best=max(vals.values()); w={k:math.exp(v-best) for k,v in vals.items()}; total=sum(w.values()) or 1.0
    return {k:v/total for k,v in w.items()}

def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value-a.cost-a.risk), a.id))

def hybrid_bandit_router(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> list[BanditAction]:
    regret_weights = compute_regret_weighted_strategy(actions, counterfactuals)
    bandit_actions = []
    for action in actions:
        propensity = regret_weights.get(action.id, 0.0)
        expected_reward = action.expected_value - action.cost - action.risk
        confidence_bound = np.sqrt(expected_reward * propensity)
        bandit_action = BanditAction(action.id, propensity, expected_reward, confidence_bound, "Hybrid")
        bandit_actions.append(bandit_action)
    return bandit_actions

def update_bandit_policy(bandit_actions: list[BanditAction], updates: list[BanditUpdate]) -> list[BanditAction]:
    for update in updates:
        for bandit_action in bandit_actions:
            if bandit_action.action_id == update.action_id:
                bandit_action.propensity = update.propensity
                bandit_action.expected_reward = update.reward
                bandit_action.confidence_bound = np.sqrt(bandit_action.expected_reward * bandit_action.propensity)
    return bandit_actions

def evaluate_hybrid_algorithm(actions: list[MathAction], counterfactuals: list[MathCounterfactual], updates: list[BanditUpdate]) -> list[BanditAction]:
    bandit_actions = hybrid_bandit_router(actions, counterfactuals)
    updated_bandit_actions = update_bandit_policy(bandit_actions, updates)
    return updated_bandit_actions

if __name__ == "__main__":
    actions = [MathAction("A", 10.0), MathAction("B", 20.0), MathAction("C", 30.0)]
    counterfactuals = [MathCounterfactual("A", 5.0), MathCounterfactual("B", 10.0), MathCounterfactual("C", 15.0)]
    updates = [BanditUpdate("context1", "A", 10.0, 0.5), BanditUpdate("context2", "B", 20.0, 0.6)]
    hybrid_bandit_actions = evaluate_hybrid_algorithm(actions, counterfactuals, updates)
    for action in hybrid_bandit_actions:
        print(action)