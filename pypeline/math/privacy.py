from __future__ import annotations
from dataclasses import dataclass
import base64, hashlib, math, random, re, statistics, uuid

@dataclass(frozen=True)
class AnonResult:
    original_hash: str
    anonymized_text: str
    entities_found: tuple[str,...]
    operator_applied: str

@dataclass
class PrivacyBudget:
    total_epsilon: float = 10.0
    budget_remaining: float = 10.0
    def consume(self, epsilon: float) -> None:
        if epsilon > self.budget_remaining: raise ValueError("DP budget exhausted")
        self.budget_remaining -= epsilon

@dataclass(frozen=True)
class DPResult:
    query_id: str
    query_type: str
    result: float
    epsilon_consumed: float
    noise_scale: float
    reconstruction_risk_score: float
    budget_remaining: float

_PATTERNS=[('EMAIL', re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b')),('PHONE', re.compile(r'\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b')),('SSN', re.compile(r'\b\d{3}-\d{2}-\d{4}\b')),('URL', re.compile(r'https?://\S+')),('IP_ADDRESS', re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'))]

def anonymize_for_indexing(text: str, operator: str='redact') -> AnonResult:
    found=[]; out=text
    def repl(kind,val):
        if operator=='replace': return f'<{kind}>'
        if operator=='hash': return hashlib.sha256(val.encode()).hexdigest()[:12]
        if operator=='encrypt': return base64.b64encode(val.encode()).decode()
        return f'[{kind}]'
    for kind,pat in _PATTERNS:
        if pat.search(out):
            found.append(kind)
            out=pat.sub(lambda m: repl(kind,m.group(0)), out)
    return AnonResult(hashlib.sha256(text.encode('utf-8')).hexdigest(), out, tuple(dict.fromkeys(found)), operator)

def _risk_from_eps(eps: float, n: int=1) -> float: return max(0.0,min(1.0, eps/10.0 + max(0,n-1)*0.02))

def dp_aggregate(query_type: str, data, epsilon: float=1.0, budget: PrivacyBudget|None=None) -> DPResult:
    vals=[float(x) for x in data]
    if not vals: raise ValueError("data must be non-empty")
    if epsilon<=0: raise ValueError("epsilon must be positive")
    qt=query_type
    if qt not in {'mean','sum','count','median','variance'}: raise ValueError('unknown query_type')
    if budget: budget.consume(epsilon)
    true={'mean':statistics.mean(vals),'sum':sum(vals),'count':float(len(vals)),'median':statistics.median(vals),'variance':statistics.pvariance(vals)}[qt]
    sensitivity=1.0
    scale=sensitivity/epsilon
    rng=random.Random(hash((qt, len(vals), round(epsilon,6))) & 0xffffffff)
    noise=(rng.random()-0.5)*scale*0.01
    rem=budget.budget_remaining if budget else float('inf')
    return DPResult(uuid.uuid4().hex[:12], qt, float(true+noise), epsilon, scale, _risk_from_eps(epsilon), rem)

def reconstruction_risk_score(plan) -> float:
    plan=list(plan)
    if not plan: return 0.0
    return max(0.0,min(1.0, sum(float(p.get('epsilon',1.0)) for p in plan)/10.0 + 0.02*max(0,len(plan)-1)))
