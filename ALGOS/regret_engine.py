#!/usr/bin/env python3
"""Regret-weighted strategy and EV ranking."""
from __future__ import annotations
import math
from dataclasses import dataclass
@dataclass(frozen=True)
class MathAction:
    id: str; expected_value: float; cost: float=0.0; risk: float=0.0
@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str; outcome_value: float; probability: float=1.0
def compute_regret_weighted_strategy(actions: list[MathAction], counterfactuals: list[MathCounterfactual]) -> dict[str,float]:
    if not actions: return {}
    cf={c.action_id:c.outcome_value*c.probability for c in counterfactuals}
    vals={a.id:a.expected_value-a.cost-a.risk+cf.get(a.id,0.0) for a in actions}
    best=max(vals.values()); w={k:math.exp(v-best) for k,v in vals.items()}; total=sum(w.values()) or 1.0
    return {k:v/total for k,v in w.items()}
def rank_actions_by_ev(actions: list[MathAction]) -> list[MathAction]:
    return sorted(actions, key=lambda a: (-(a.expected_value-a.cost-a.risk), a.id))
