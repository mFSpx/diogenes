# DARWIN HAMMER — match 1430, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s4.py (gen3)
# born: 2026-05-29T23:37:48Z

"""
Hybrid Regret‑Weighted Pheromone Bandit

This module fuses the core mathematics of two parent algorithms:

* **Parent A – Hybrid Regret‑Weighted Bandit with Honeybee Store and MinHash Bridge**  
  The regret‑weighted value of each action is computed from expected value,
  cost, risk and counterfactual outcomes.  A MinHash signature of the
  action’s token set provides a similarity metric that can modulate those
  values.

* **Parent B – Hybrid Pheromone Bandit System**  
  A tighter integration of a pheromone‑based decay model with a
  multi‑armed bandit (UCB1) algorithm.

The mathematical bridge between the two parents lies in the use of the 
MinHash similarity as a *multiplicative factor* that scales the 
regret‑weighted utility, which in turn biases the pheromone‑based 
exploration in the bandit algorithm. The pheromone signal is used as 
a *prior* for the expected reward of each arm, and the decayed 
pheromone value is combined with the regret‑weighted utility to 
compute a hybrid score.
"""

import sys
import pathlib
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict

import numpy as np

@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"

@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy (not used directly in the demo)."""
    context_id: str
    action_id: str
    reward: float

class HybridRegretPheromoneBandit:
    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms

        # Pheromone store: surface_key → dict with decay parameters
        self.pheromones: Dict[str, Dict[str, any]] = {}

        # Bandit statistics
        self.counts = np.zeros(n_arms, dtype=int)          # pulls per arm
        self.values = np.zeros(n_arms, dtype=float)        # average reward per arm

        # Global statistics for scaling
        self.total_pulls = 0
        self.store = 0.0                                     # cumulative pheromone mass

    def minhash(self, tokens: Tuple[str, ...], n_buckets: int = 10) -> int:
        """Compute MinHash signature for a token set."""
        hash_values = []
        for token in tokens:
            hash_object = hashlib.md5(token.encode())
            hash_value = int(hash_object.hexdigest(), 16)
            hash_values.append(hash_value)
        min_hash = min(hash_values) % n_buckets
        return min_hash

    def regret_weighted_utility(self, action: MathAction, counterfactuals: List[MathCounterfactual]) -> float:
        """Compute regret-weighted utility for an action."""
        expected_value = action.expected_value
        cost = action.cost
        risk = action.risk
        regret = 0.0
        for counterfactual in counterfactuals:
            regret += (counterfactual.outcome_value - expected_value) * counterfactual.probability
        utility = expected_value - cost - risk - regret
        return utility

    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _decayed_signal(self, created: datetime, value: float, half_life: float) -> float:
        """
        Return the exponentially decayed signal value.
        """
        if half_life <= 0:
            raise ValueError("half_life_seconds must be positive")
        elapsed = (self._current_utc() - created).total_seconds()
        decay_factor = 0.5 ** (elapsed / half_life)
        return value * decay_factor

    def update_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """
        Insert or update a pheromone signal.
        """
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {}
        self.pheromones[surface_key][signal_kind] = (signal_value, self._current_utc())
        return self._decayed_signal(self.pheromones[surface_key][signal_kind][1], signal_value, half_life_seconds)

    def hybrid_score(self, action: MathAction, counterfactuals: List[MathCounterfactual], surface_key: str, half_life_seconds: float) -> float:
        """Compute hybrid score for an action."""
        utility = self.regret_weighted_utility(action, counterfactuals)
        min_hash = self.minhash(action.tokens)
        pheromone_value = self.update_pheromone(surface_key, action.id, utility, half_life_seconds)
        hybrid_utility = utility * (1 + min_hash * pheromone_value)
        return hybrid_utility

    def select_action(self, actions: List[MathAction], counterfactuals: List[MathCounterfactual], surface_key: str, half_life_seconds: float) -> BanditAction:
        """Select an action based on hybrid scores."""
        hybrid_utilities = [self.hybrid_score(action, counterfactuals, surface_key, half_life_seconds) for action in actions]
        propensities = np.softmax(hybrid_utilities)
        action_id = random.choices([action.id for action in actions], weights=propensities)[0]
        action = next(action for action in actions if action.id == action_id)
        expected_reward = self.regret_weighted_utility(action, counterfactuals)
        confidence_bound = np.sqrt(np.log(len(actions)) / (self.counts[[action.id for action in actions].index(action_id)] + 1))
        return BanditAction(action_id, propensities[[action.id for action in actions].index(action_id)], expected_reward, confidence_bound)

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    actions = [
        MathAction("action1", ("token1", "token2"), 10.0),
        MathAction("action2", ("token3", "token4"), 20.0),
        MathAction("action3", ("token5", "token6"), 30.0),
    ]

    counterfactuals = [
        MathCounterfactual("action1", 15.0),
        MathCounterfactual("action2", 25.0),
        MathCounterfactual("action3", 35.0),
    ]

    bandit = HybridRegretPheromoneBandit()
    selected_action = bandit.select_action(actions, counterfactuals, "surface_key", 10.0)
    print(selected_action)