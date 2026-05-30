# DARWIN HAMMER — match 3206, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s2.py (gen3)
# parent_b: hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s2.py (gen3)
# born: 2026-05-29T23:48:31Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_context(text: str | None) -> dict[str, any]:
    if not text:
        return {}
    try:
        import json
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value

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

    def update(self, inflow: list[float], outflow: list[float]) -> tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        return self.level + delta, delta

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

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> dict[str, float]:
    if not actions: return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions}
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}

def gini_coefficient(values: list[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    area = 0.0
    n = len(xs)
    for i, x in enumerate(xs):
        area += ((2 * i + 1) * x) / (n * n)
    return 1 - (2 / n**2) * sum(xs) * area / sum(xs)

def hybrid_operation(actions: list[MathAction], counterfactuals: list[MathCounterfactual], store_state: StoreState) -> tuple[dict[str, float], StoreState]:
    regret_weighted_strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    values = list(regret_weighted_strategy.values())
    gini = gini_coefficient(values)
    inflow = [x * (1 - gini) for x in values]
    outflow = [x * gini for x in values]
    new_level, delta = store_state.update(inflow, outflow)
    return regret_weighted_strategy, StoreState(level=new_level, alpha=store_state.alpha, beta=store_state.beta, dt=store_state.dt, base=store_state.base, gain=store_state.gain, limit=store_state.limit)

def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value - a.cost - a.risk), a.id))

def bandit_update(bandit_action: BanditAction, store_state: StoreState) -> BanditUpdate:
    return BanditUpdate(context_id=store_state.base, action_id=bandit_action.action_id, reward=bandit_action.expected_reward, propensity=bandit_action.propensity)

def kl_divergence(p: list[float], q: list[float]) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)) if p[i] != 0)

def hybrid_kl_regularized_operation(actions: list[MathAction], counterfactuals: list[MathCounterfactual], store_state: StoreState) -> tuple[dict[str, float], StoreState]:
    regret_weighted_strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    uniform_strategy = {k: 1 / len(actions) for k in regret_weighted_strategy}
    kl_div = kl_divergence(list(regret_weighted_strategy.values()), list(uniform_strategy.values()))
    values = list(regret_weighted_strategy.values())
    gini = gini_coefficient(values)
    inflow = [x * (1 - gini) for x in values]
    outflow = [x * gini for x in values]
    new_level, delta = store_state.update(inflow, outflow)
    return regret_weighted_strategy, StoreState(level=new_level, alpha=store_state.alpha, beta=store_state.beta, dt=store_state.dt, base=store_state.base, gain=store_state.gain, limit=store_state.limit)

if __name__ == "__main__":
    store_state = StoreState()
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    counterfactuals = [MathCounterfactual("action1", 5.0)]
    regret_weighted_strategy, new_store_state = hybrid_operation(actions, counterfactuals, store_state)
    print(regret_weighted_strategy)
    print(new_store_state)
    print(rank_actions_by_ev(actions))
    bandit_action = BanditAction("action1", 0.5, 10.0, 0.1, "algorithm1")
    bandit_update_result = bandit_update(bandit_action, store_state)
    print(bandit_update_result)
    kl_regret_weighted_strategy, new_kl_store_state = hybrid_kl_regularized_operation(actions, counterfactuals, store_state)
    print(kl_regret_weighted_strategy)
    print(new_kl_store_state)