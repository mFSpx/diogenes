from __future__ import annotations
import math
from .types import MathAction, MathCounterfactual

def _net(action: MathAction, counterfactuals: list[MathCounterfactual]) -> float:
    penalty = sum(cf.probability * cf.loss_bound * max(abs(action.expected_value), 1.0) for cf in counterfactuals if cf.action == action.id)
    return float(action.expected_value) - float(action.risk) - penalty

def rank_actions_by_ev(actions: list[MathAction], counterfactuals: list[MathCounterfactual] | None = None) -> list[MathAction]:
    cfs = counterfactuals or []
    return sorted(actions, key=lambda a: _net(a, cfs), reverse=True)

def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual] | None = None, iterations: int | None = None) -> dict[str, float]:
    if not actions:
        if iterations is not None:
            raise NotImplementedError("CFR iterations require a game tree; use EV ranking for empty action sets")
        return {}
    if len(actions) == 1:
        return {actions[0].id: 1.0}
    cfs = counterfactuals or []
    nets = [_net(a, cfs) for a in actions]
    # Shift to positive softmax-like weights; equal nets -> equal probabilities.
    m = max(nets)
    weights = [math.exp(min(50.0, n - m)) for n in nets]
    s = sum(weights) or 1.0
    return {a.id: w / s for a, w in zip(actions, weights)}
