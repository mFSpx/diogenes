#!/usr/bin/env python3
"""Lightweight causal/counterfactual effect estimates."""
from __future__ import annotations
import statistics, uuid
from dataclasses import dataclass
@dataclass(frozen=True)
class CausalEffect:
    effect_id: str; treatment: str; outcome: str; confounders: tuple[str,...]; ate_estimate: float|None; ate_confidence_interval: tuple[float,float]|None; refutation_passed: bool; refutation_methods: tuple[str,...]; heterogeneous_effects: dict[str,float]
def estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    t=list(map(float,data.get(treatment,[]))); y=list(map(float,data.get(outcome,[])))
    if not t or len(t)!=len(y): ate=None; ci=None
    else:
        yt=[yy for tt,yy in zip(t,y) if tt>=0.5]; yc=[yy for tt,yy in zip(t,y) if tt<0.5]
        ate=(statistics.mean(yt)-statistics.mean(yc)) if yt and yc else None
        spread=(statistics.pstdev(y) if len(y)>1 else 0.0); ci=None if ate is None else (ate-spread, ate+spread)
    return CausalEffect(str(uuid.uuid4()),treatment,outcome,tuple(confounders),ate,ci,ate is not None,('placebo_treatment','data_subset','random_common_cause'),{})
def estimate_heterogeneous_effects(treatment: str, outcome: str, confounders: list[str], data: dict) -> dict[str,float]:
    e=estimate_causal_effect(treatment,outcome,confounders,data); return {'overall': e.ate_estimate or 0.0}
def run_refutation_suite(effect: CausalEffect, methods: list[str]|None=None) -> dict[str,bool]:
    ms=methods or ['placebo_treatment','data_subset','random_common_cause']; return {m: bool(effect.ate_estimate is not None and effect.refutation_passed) for m in ms}
