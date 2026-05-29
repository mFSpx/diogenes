from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from .cockpit_metrics import anti_slop_ratio, audit_debt, cockpit_honesty

@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: float
    threshold: float
    equation_id: str
    name: str = ""
    ledger_eligible: bool = True
    @property
    def margin(self) -> float: return self.score - self.threshold
    def __str__(self) -> str: return f"{'PASS' if self.passed else 'FAIL'} {self.equation_id} score={self.score} threshold={self.threshold}"

_REG: dict[str, callable] = {}
_ALIASES: dict[str, str] = {}

def register(eq_id: str, name: str | None = None):
    def deco(fn):
        _REG[eq_id]=fn
        if name: _ALIASES[name]=eq_id
        return fn
    return deco

def _res(eq, score, threshold, pass_if='ge', name=''):
    passed = score >= threshold if pass_if=='ge' else score <= threshold
    return ValidationResult(bool(passed), float(score), float(threshold), eq, name)

def validate(eq_id_or_name: str, **kw) -> ValidationResult:
    eq=_ALIASES.get(eq_id_or_name, eq_id_or_name)
    if eq not in _REG: raise KeyError(f"No validator registered for {eq_id_or_name}")
    return _REG[eq](**kw)

@register('EQ-001','NamespaceBleedCost')
def _eq001(cross_project_imports=0,total_imports=0,threshold=0.0):
    score=0.0 if total_imports<=0 else 10.0*cross_project_imports/total_imports
    return _res('EQ-001', score, threshold, 'le')
@register('EQ-003','TestFloorPreservation')
def _eq003(tests_passing_now=0,floor=2778): return _res('EQ-003', tests_passing_now, floor, 'ge')
@register('EQ-015','RAMBudget')
def _eq015(resident_models_mb=0,indexes_mb=0,os_overhead_mb=0,ceiling_mb=8028.16,threshold=0.85): return _res('EQ-015',(resident_models_mb+indexes_mb+os_overhead_mb)/ceiling_mb,threshold,'le')
@register('EQ-087','AuditDebt')
def _eq087(exports_missing_audit_step=0): return _res('EQ-087', audit_debt(exports_missing_audit_step),0.0,'le')
@register('EQ-089','AntiSlopRatio')
def _eq089(claims_with_evidence=0,total_claims_emitted=0): return _res('EQ-089', anti_slop_ratio(claims_with_evidence,total_claims_emitted),1.0,'ge')
@register('EQ-027','CockpitHonesty')
def _eq027(displayed_ok=0,unknown_displayed_as_ok=0): return _res('EQ-027', cockpit_honesty(displayed_ok,unknown_displayed_as_ok),1.0,'ge')
@register('EQ-070','EVI')
def _eq070(expected_value_with_answer=0,value_without_answer=0,query_cost=0): return _res('EQ-070', expected_value_with_answer-value_without_answer-query_cost,0.0,'ge')
@register('EQ-071','RegretAgainstNoOp')
def _eq071(strategy_ev=0,noop_ev=0): return _res('EQ-071', strategy_ev-noop_ev,0.0,'ge')
