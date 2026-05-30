# DARWIN HAMMER — match 822, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s2.py (gen3)
# parent_b: regret_engine.py (gen0)
# born: 2026-05-29T23:31:07Z

"""
This module fuses the Regret-Weighted Strategy from regret_engine.py 
with the Hybrid Bandit Router from hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s2.py.
The mathematical bridge between these two structures lies in the application of the regret-weighted 
strategy to modulate the propensity scores in the Hybrid Bandit Router. Specifically, we use the 
regret-weighted strategy to compute the expected values of actions in the bandit router, 
which are then used to update the propensity scores.

The governing equations of the regret engine are used to compute the regret-weighted strategy, 
which is then used to update the propensity scores in the bandit router. The bandit router's 
governing equations are used to select actions based on the updated propensity scores.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import hashlib
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

@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation and recompute the dance duration.

        Returns
        -------
        new_level, delta
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ (computed lazily)."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> dict[str,float]:
    if not actions: return {}
    cf={c.action_id:c.outcome_value*c.probability for c in counterfactuals}
    vals={a.id:a.expected_value-a.cost-a.risk+cf.get(a.id,0.0) for a in actions}
    best=max(vals.values()); w={k:math.exp(v-best) for k,v in vals.items()}; total=sum(w.values()) or 1.0
    return {k:v/total for k,v in w.items()}

def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value-a.cost-a.risk), a.id))

def hybrid_bandit_router(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> BanditAction:
    regret_weights = compute_regret_weighted_strategy(actions, counterfactuals)
    action_id = np.random.choice([a.id for a in actions], p=[regret_weights.get(a.id, 0.0) for a in actions])
    action = next(a for a in actions if a.id == action_id)
    propensity = regret_weights.get(action_id, 0.0)
    expected_reward = action.expected_value - action.cost - action.risk
    confidence_bound = np.sqrt(np.log(len(actions)) / (propensity * len(actions)))
    return BanditAction(action_id, propensity, expected_reward, confidence_bound, "Hybrid")

def update_policy(store_state: StoreState, bandit_update: BanditUpdate) -> StoreState:
    inflow = [bandit_update.reward * bandit_update.propensity]
    outflow = [store_state.level]
    store_state.update(inflow, outflow)
    return store_state

def smoke_test():
    actions = [MathAction("a", 10.0), MathAction("b", 20.0), MathAction("c", 30.0)]
    counterfactuals = [MathCounterfactual("a", 5.0), MathCounterfactual("b", 10.0), MathCounterfactual("c", 15.0)]
    bandit_action = hybrid_bandit_router(actions, counterfactuals)
    store_state = StoreState()
    bandit_update = BanditUpdate("context", bandit_action.action_id, 10.0, bandit_action.propensity)
    updated_store_state = update_policy(store_state, bandit_update)

if __name__ == "__main__":
    smoke_test()