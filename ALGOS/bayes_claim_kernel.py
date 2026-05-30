from __future__ import annotations
from dataclasses import replace
from .types import MathClaim, MathEvidence, MathHypothesis

def update_hypothesis(hypothesis: MathHypothesis, evidence: MathEvidence, likelihood_ratio: float) -> MathHypothesis:
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non-negative")
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    ids = tuple(list(hypothesis.evidence_ids) + [evidence.id])
    return MathHypothesis(id=hypothesis.id, prior=hypothesis.posterior, posterior=posterior, evidence_ids=ids)

def compute_log_likelihood_ratio(claim: MathClaim, hypothesis_id: str, evidence: list[MathEvidence]) -> float:
    raise NotImplementedError("claim-specific likelihood models must be supplied by caller")
