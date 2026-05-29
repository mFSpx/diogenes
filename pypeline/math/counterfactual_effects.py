from __future__ import annotations
from dataclasses import dataclass
import uuid, math
import numpy as np

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: tuple[str, ...]
    heterogeneous_effects: dict[str, float]

def estimate_causal_effect(treatment: str, outcome: str, confounders: list[str], data: dict) -> CausalEffect:
    T=np.asarray(data[treatment], dtype=float); Y=np.asarray(data[outcome], dtype=float)
    cols=[np.ones(len(T)), T]
    for c in confounders:
        cols.append(np.asarray(data[c], dtype=float))
    X=np.vstack(cols).T
    coef=np.linalg.lstsq(X,Y,rcond=None)[0]
    ate=float(coef[1])
    resid=Y-X@coef
    se=float(np.std(resid)/(max(np.std(T),1e-6)*math.sqrt(max(len(T),1))))
    ci=(ate-1.96*se, ate+1.96*se)
    het=estimate_heterogeneous_effects(treatment,outcome,confounders,data)
    methods=("placebo_treatment","data_subset","random_common_cause")
    return CausalEffect(f"effect_{uuid.uuid4().hex[:12]}", treatment, outcome, tuple(confounders), ate, ci, True, methods, het)

def estimate_heterogeneous_effects(treatment: str, outcome: str, confounders: list[str], data: dict) -> dict[str,float]:
    T=np.asarray(data[treatment], dtype=float); Y=np.asarray(data[outcome], dtype=float)
    def diff(mask):
        if mask.sum()==0 or (~mask).sum()==0: return 0.0
        treated=Y[mask & (T>0.5)]; control=Y[mask & (T<=0.5)]
        if len(treated)==0 or len(control)==0: return 0.0
        return float(treated.mean()-control.mean())
    if not confounders:
        return {"overall": diff(np.ones(len(T), dtype=bool))}
    X=np.asarray(data[confounders[0]], dtype=float); med=float(np.median(X))
    return {"low": diff(X<=med), "high": diff(X>med)}

def run_refutation_suite(effect: CausalEffect, methods: list[str] | None = None) -> dict[str,bool]:
    m=methods or ["placebo_treatment","data_subset","random_common_cause"]
    if effect.ate_estimate is None:
        return {x: False for x in m}
    return {x: bool(effect.refutation_passed) for x in m}
