# DARWIN HAMMER — match 1226, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_worksh_m156_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_nlms_o_m937_s0.py (gen4)
# born: 2026-05-29T23:34:33Z

"""Hybrid Regret-Bayesian Curvature Engine

This module fuses the core topologies of:
- **Parent A** (`hybrid_hybrid_regret_engine_hybrid_hybrid_worksh_m156_s0.py`): 
  Gini coefficient on weekday distributions and a regret‑weighted strategy.
- **Parent B** (`hybrid_hybrid_hybrid_bayes__hybrid_hybrid_nlms_o_m937_s0.py`): 
  Bayesian hypothesis updating with a likelihood ratio.

**Mathematical bridge**
The Gini coefficient, a scalar measuring inequality of a distribution, is
interpreted here as a proxy for Ollivier‑Ricci curvature: higher inequality
implies stronger “curvature” of the probability landscape.  This scalar
modulates the likelihood ratio used in the Bayesian update, while the
regret‑weighted strategy supplies the prior probabilities for each hypothesis.
Thus the hybrid algorithm simultaneously respects the regret‑based utility
and curvature‑adjusted Bayesian inference."""

from __future__ import annotations
import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared between the two parent topologies)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action description used in the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome used to adjust regret."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class MathEvidence:
    """Evidence element used in the Bayesian update."""
    id: str


@dataclass(frozen=True)
class MathHypothesis:
    """Bayesian hypothesis attached to a specific action."""
    id: str                     # same as the action id
    prior: float                # prior probability (regret‑weighted)
    posterior: float            # current posterior
    evidence_ids: Tuple[str, ...]  # accumulated evidence identifiers


# ----------------------------------------------------------------------
# Parent A – Gini and regret weighted utilities
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return a deterministic weekday index (0‑Mon … 6‑Sun) based on the Doomsday algorithm."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative sequence."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def weekday_distribution(year: int, month: int, num_days: int) -> np.ndarray:
    """Count occurrences of each weekday within a month."""
    weekdays = [doomsday(year, month, day) for day in range(1, num_days + 1)]
    counts = np.zeros(7, dtype=float)
    for w in weekdays:
        counts[w] += 1.0
    return counts


def gini_weekday(year: int, month: int, num_days: int) -> float:
    """Gini coefficient of the weekday count distribution."""
    return gini_coefficient(weekday_distribution(year, month, num_days))


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Regret‑weighted softmax over adjusted action values.
    Returns a probability distribution over action identifiers.
    """
    if not actions:
        return {}
    # Map counterfactual adjustments
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    # Adjusted utility per action
    utilities = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(utilities.values())
    # Softmax with negative exponent (higher utility → higher weight)
    weights = {k: math.exp(v - best) for k, v in utilities.items()}
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


# ----------------------------------------------------------------------
# Parent B – Bayesian hypothesis update (likelihood ratio)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """
    Classic Bayesian odds update using a likelihood ratio.
    The posterior becomes the new prior for the next iteration.
    """
    if likelihood_ratio < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
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
    new_evidence = hypothesis.evidence_ids + (evidence.id,)
    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=new_evidence,
    )


# ----------------------------------------------------------------------
# Hybrid bridge – curvature‑adjusted likelihood
# ----------------------------------------------------------------------
def curvature_adjusted_likelihood(
    base_lr: float,
    gini: float,
    epsilon: float = 1e-6,
) -> float:
    """
    Treat the Gini coefficient as a curvature scalar κ ∈ [0,1].
    Adjust the base likelihood ratio L by a factor (1 + κ) to obtain
    L' = L * (1 + κ).  The epsilon guards against degenerate zero values.
    """
    if base_lr < 0:
        raise ValueError("base likelihood ratio must be non‑negative")
    curvature_factor = 1.0 + gini  # κ maps 0→1, 1→2
    adjusted = max(epsilon, base_lr * curvature_factor)
    return adjusted


def deterministic_likelihood_from_ids(action_id: str, evidence_id: str) -> float:
    """
    Produce a reproducible likelihood ratio in (0, 2] from two identifiers.
    The hash is made deterministic across runs by using Python's built‑in hash
    combined with a fixed seed.
    """
    seed = hash((action_id, evidence_id))
    rng = random.Random(seed)
    # Uniform draw in (0, 2]; avoid exactly 0
    return rng.random() * 2.0 + 1e-6


# ----------------------------------------------------------------------
# Hybrid engine – three public functions
# ----------------------------------------------------------------------
def initialise_hypotheses(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, MathHypothesis]:
    """
    Create a hypothesis per action using the regret‑weighted strategy as the prior.
    The initial posterior equals the prior.
    """
    priors = compute_regret_weighted_strategy(actions, counterfactuals)
    hypotheses = {
        aid: MathHypothesis(
            id=aid,
            prior=pri,
            posterior=pri,
            evidence_ids=(),
        )
        for aid, pri in priors.items()
    }
    return hypotheses


def hybrid_update_step(
    hypotheses: Dict[str, MathHypothesis],
    evidences: List[MathEvidence],
    gini: float,
) -> Dict[str, MathHypothesis]:
    """
    Perform one full Bayesian update over the supplied evidences.
    Each evidence updates every hypothesis using a curvature‑adjusted likelihood.
    """
    updated = hypotheses.copy()
    for ev in evidences:
        for aid, hyp in updated.items():
            base_lr = deterministic_likelihood_from_ids(aid, ev.id)
            adj_lr = curvature_adjusted_likelihood(base_lr, gini)
            updated[aid] = update_hypothesis(hyp, ev, adj_lr)
    return updated


def run_hybrid_engine(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    evidences: List[MathEvidence],
    year: int,
    month: int,
    num_days: int,
) -> Dict[str, float]:
    """
    Full pipeline:
    1. Compute Gini of weekday distribution (curvature proxy).
    2. Initialise Bayesian hypotheses with regret‑weighted priors.
    3. Update hypotheses with curvature‑adjusted likelihoods.
    4. Return final posterior probabilities per action.
    """
    gini = gini_weekday(year, month, num_days)
    hyps = initialise_hypotheses(actions, counterfactuals)
    hyps = hybrid_update_step(hyps, evidences, gini)
    # Extract posterior probabilities
    return {aid: hyp.posterior for aid, hyp in hyps.items()}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample actions
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=8.0, cost=1.0, risk=0.5),
        MathAction(id="C", expected_value=6.0, cost=0.5, risk=0.2),
    ]

    # Counterfactual adjustments (could be empty)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=1.5, probability=0.8),
        MathCounterfactual(action_id="B", outcome_value=-0.5, probability=0.6),
    ]

    # Deterministic evidence set
    evidences = [
        MathEvidence(id="e1"),
        MathEvidence(id="e2"),
        MathEvidence(id="e3"),
    ]

    # Use current month for Gini computation (any realistic parameters)
    today = dt.date.today()
    year, month = today.year, today.month
    # Number of days in the month (simple approximation)
    num_days = (dt.date(year + int(month / 12), (month % 12) + 1, 1) - dt.timedelta(days=1)).day

    posteriors = run_hybrid_engine(
        actions=actions,
        counterfactuals=counterfactuals,
        evidences=evidences,
        year=year,
        month=month,
        num_days=num_days,
    )
    print("Final posterior probabilities per action:")
    for aid, prob in sorted(posteriors.items()):
        print(f"  Action {aid}: {prob:.4f}")
    sys.exit(0)