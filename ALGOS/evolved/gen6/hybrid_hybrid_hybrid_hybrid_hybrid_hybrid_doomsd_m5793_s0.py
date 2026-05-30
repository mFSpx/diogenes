# DARWIN HAMMER — match 5793, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s0.py (gen5)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_regret_engine_m1429_s3.py (gen3)
# born: 2026-05-30T00:04:42Z

"""
Hybrid Module: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s0 + hybrid_hybrid_doomsday_cale_hybrid_regret_engine_m1429_s3

Mathematical Bridge
-------------------
Parent A manages a *ModelPool* whose internal state is quantified by an **energy**
value.  Parent B produces a *regret‑weighted probability distribution* over
actions and a **Gini coefficient** that measures inequality of regret across
weekdays.

The fusion treats the regret‑weighted distribution as an **entropy source**.
Shannon entropy `H` of that distribution is injected into the pool’s energy
(`ΔE = α·H`).  Simultaneously, the Gini coefficient of the supplied regret
vector is used as a multiplicative penalty (`ΔE = β·G·E₀`).  The resulting
energy‑adjusted pool guides model selection: the model whose descriptor aligns
with the highest‑probability action is loaded, while respecting RAM limits and
the pool’s energy budget.

Thus the core topology of both parents is merged:
* probability → entropy → energy adjustment,
* regret → Gini → energy scaling,
* energy → model‑loading policy.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Dict, List

# ---------- Parent A: Model pool and energy handling ----------

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"


class ModelPool:
    """Manages a collection of models with an energy metric."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        """Add a model, applying penalties for tier conflicts or RAM overflow."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10  # penalty for conflicting tiers
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6   # penalty for high memory usage
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """Direct load without eviction (rewarded)."""
        self._energy -= 1e4  # reward for loading a model
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load with eviction of the largest RAM consumer until space is freed."""
        self._energy -= 1e3  # reward for evicting an old model
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = max(self.loaded, key=lambda n: self.loaded[n].ram_mb)
            evicted = self.loaded.pop(evicted_name)
            self._energy += 1e2  # penalty for evicting a model
        self.load(model)

    @property
    def energy(self) -> float:
        return self._energy

    def __repr__(self) -> str:
        return f"<ModelPool used={self._used()}MB energy={self._energy:.2e} models={list(self.loaded)}>"


# ---------- Parent B: Regret weighted strategy & Gini ----------

@dataclass(frozen=True)
class Action:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class Counterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class Day:
    weekday: int  # 1=Monday ... 7=Sunday
    count: int


def compute_regret_weighted_strategy(actions: List[Action],
                                     counterfactuals: List[Counterfactual]) -> Dict[str, float]:
    """Return a softmax over (expected‑value – cost – risk + cf) for each action."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions}
    best = max(vals.values())
    weights = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def gini_coefficient(values: List[float]) -> float:
    """Standard Gini coefficient for a non‑negative list."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def weekday_distribution(year: int, month: int, num_days: int) -> List[Day]:
    """Count occurrences of each weekday in a month."""
    def doomsday(y: int, m: int, d: int) -> int:
        return (date(y, m, d).weekday() + 1) % 7  # 0=Sunday → 7, 1=Monday →1, ...

    weekdays = [doomsday(year, month, day) for day in range(1, num_days + 1)]
    counts = np.bincount(weekdays, minlength=7)
    # shift index: 0 → Sunday (7), 1 → Monday (1), … we map to 1‑7
    return [Day(i if i != 0 else 7, int(counts[i])) for i, _ in enumerate(counts)]


# ---------- Fusion utilities ----------

def shannon_entropy(prob_dist: Dict[str, float]) -> float:
    """Compute Shannon entropy H = -∑ p·log(p)."""
    return -sum(p * math.log(p) for p in prob_dist.values() if p > 0.0)


def hybrid_energy_adjustment(pool: ModelPool,
                             actions: List[Action],
                             counterfactuals: List[Counterfactual],
                             regret_values: List[float],
                             alpha: float = 1.0,
                             beta: float = 1.0) -> None:
    """
    Adjust the pool's internal energy using:
    * α·H where H is the entropy of the regret‑weighted strategy,
    * β·G·E₀ where G is the Gini coefficient of regret_values and E₀ is the current energy.
    The function mutates `pool._energy` in place.
    """
    # 1) entropy from regret‑weighted strategy
    strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    H = shannon_entropy(strategy)
    pool._energy += alpha * H

    # 2) Gini‑based scaling
    G = gini_coefficient(regret_values)
    pool._energy += beta * G * pool._energy  # multiplicative penalty/reward


def hybrid_select_model(pool: ModelPool,
                        models: List[ModelTier],
                        actions: List[Action],
                        counterfactuals: List[Counterfactual]) -> ModelTier:
    """
    Select a model whose name matches the highest‑probability action identifier.
    If no exact match, fall back to the model with the smallest additional RAM
    requirement that keeps the pool under its ceiling.
    """
    strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    if not strategy:
        raise ValueError("Empty strategy – cannot select a model.")

    # Action with maximal probability
    best_action_id = max(strategy, key=strategy.get)

    # Direct match
    for m in models:
        if m.name == best_action_id:
            return m

    # No direct match → choose minimal‑ram model that fits after possible eviction
    feasible = [m for m in models if m.ram_mb + pool._used() <= pool.ram_ceiling_mb]
    if feasible:
        return min(feasible, key=lambda m: m.ram_mb)

    # As last resort, return the smallest model (eviction will happen later)
    return min(models, key=lambda m: m.ram_mb)


def hybrid_load_policy(pool: ModelPool,
                       models: List[ModelTier],
                       actions: List[Action],
                       counterfactuals: List[Counterfactual],
                       regret_values: List[float]) -> None:
    """
    Full hybrid operation:
    1. Adjust pool energy with entropy and Gini penalties.
    2. Pick a model using the regret‑weighted strategy.
    3. Load the model, evicting if necessary.
    """
    hybrid_energy_adjustment(pool, actions, counterfactuals, regret_values,
                             alpha=5.0, beta=0.3)
    chosen = hybrid_select_model(pool, models, actions, counterfactuals)

    # Decide whether we can load directly or need eviction
    if chosen.ram_mb + pool._used() <= pool.ram_ceiling_mb:
        pool.load(chosen)
    else:
        pool.load_with_eviction(chosen)


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Create a pool
    pool = ModelPool(ram_ceiling_mb=8000)

    # Define some mock models
    model_catalog = [
        ModelTier(name="A1", ram_mb=1500, tier="T1"),
        ModelTier(name="B2", ram_mb=2500, tier="T2"),
        ModelTier(name="C3", ram_mb=3500, tier="T3"),
        ModelTier(name="D4", ram_mb=1200, tier="T1"),
    ]

    # Define actions and counterfactuals
    actions = [
        Action(id="A1", expected_value=120.0, cost=20.0, risk=5.0),
        Action(id="B2", expected_value=150.0, cost=30.0, risk=10.0),
        Action(id="X9", expected_value=80.0, cost=10.0, risk=2.0),
    ]
    counterfactuals = [
        Counterfactual(action_id="A1", outcome_value=15.0, probability=0.8),
        Counterfactual(action_id="B2", outcome_value=-5.0, probability=0.5),
        Counterfactual(action_id="X9", outcome_value=20.0, probability=0.3),
    ]

    # Regret values per weekday (dummy example)
    regret_vals = [3.2, 1.5, 2.8, 4.0, 0.9, 2.1, 3.3]

    # Run the hybrid loading policy
    hybrid_load_policy(pool, model_catalog, actions, counterfactuals, regret_vals)

    # Output final state
    print(pool)
    print("Final energy:", pool.energy)
    print("Loaded models:", list(pool.loaded.keys()))