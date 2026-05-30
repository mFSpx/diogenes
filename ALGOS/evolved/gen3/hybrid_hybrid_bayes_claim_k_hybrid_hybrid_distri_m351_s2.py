# DARWIN HAMMER — match 351, survivor 2
# gen: 3
# parent_a: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s0.py (gen2)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s1.py (gen2)
# born: 2026-05-29T23:28:23Z

"""Hybrid Bayesian–Strike Algorithm

This module fuses two parent algorithms:

* **Parent A** – Bayesian claim kernel with a decreasing‑rate pruning schedule.
* **Parent B** – Chelydrid ambush‑strike kinematics used for burst‑admission scoring.

**Mathematical bridge**

For each piece of evidence we compute a *selection score* using the
ambush‑strike model (`integrate_strike` → `burst_admission_score`).  
The score acts as a *dynamic likelihood modifier* that is multiplied by
the standard Bayesian likelihood ratio.  The resulting effective
likelihood is then subject to the time‑dependent pruning probability
from Parent A.  In this way the drag‑based cost from the strike model
modulates the Bayesian update while the exponential pruning schedule
controls evidence retention.

The hybrid therefore consists of:
1. Computing a physics‑based admission score for evidence.
2. Scaling the Bayesian likelihood ratio by that score.
3. Applying the pruning schedule to decide whether the evidence
   participates in the posterior update.

The three core functions below illustrate this combined workflow.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Mapping, Iterable, Hashable, Tuple
import numpy as np
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass

# ---------- Parent A structures ----------
class MathClaim:
    def __init__(self, id: str):
        self.id = id

class MathEvidence:
    def __init__(self, id: str, strength: float):
        """`strength` is a raw signal value that will be turned into a likelihood ratio."""
        self.id = id
        self.strength = strength

class MathHypothesis:
    def __init__(self, id: str, prior: float, posterior: float, evidence_ids: List[str]):
        self.id = id
        self.prior = prior
        self.posterior = posterior
        self.evidence_ids = evidence_ids

def update_hypothesis(hypothesis: MathHypothesis,
                     evidence: MathEvidence,
                     likelihood_ratio: float) -> MathHypothesis:
    """Standard Bayesian odds update."""
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
    ids = tuple(list(hypothesis.evidence_ids) + [evidence.id])
    return MathHypothesis(id=hypothesis.id,
                          prior=hypothesis.posterior,
                          posterior=posterior,
                          evidence_ids=list(ids))

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decay pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))

def prune_evidence(evidence: List[MathEvidence],
                   t: float,
                   lam: float = 1.0,
                   alpha: float = 0.2,
                   seed: int | str | None = None) -> List[MathEvidence]:
    """Randomly keep each evidence item with probability given by `prune_probability`."""
    rng = random.Random(seed)
    keep_prob = prune_probability(t, lam, alpha)
    return [e for e in evidence if rng.random() < keep_prob]

# ---------- Parent B structures ----------
Node = Hashable
Graph = MappingABC[Node, set[Node]]

@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float

def integrate_strike(force_series: Iterable[float],
                    dt: float,
                    m_head: float,
                    drag_cd: float = 0.3,
                    fluid_density: float = 1000.0,
                    area: float = 0.001,
                    v0: float = 0.0) -> StrikeState:
    """Integrate 1‑D motion under a piecewise constant force with quadratic drag."""
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)

def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse of `steps` samples, symmetric around the centre."""
    if steps <= 0:
        return []
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)]

def burst_admission_score(work_value: float,
                          cost_drag: float,
                          urgency_force: float,
                          steps: int = 12,
                          dt: float = 0.01,
                          m_head: float = 0.01) -> float:
    """
    Compute a scalar admission score using the ambush‑strike integration.
    The `work_value` plays the role of a baseline likelihood; `cost_drag`
    reduces it, while `urgency_force` injects additional force.
    """
    # Build a force series: base work + urgency, attenuated by drag cost.
    base = max(0.0, work_value - cost_drag)
    force_series = pulse_force(base + max(0.0, urgency_force), steps)
    state = integrate_strike(force_series, dt, m_head)
    # Normalise by the maximum possible peak (when drag = 0 and urgency = work_value)
    max_state = integrate_strike(pulse_force(work_value, steps), dt, m_head)
    norm = (state.distance / max_state.distance) if max_state.distance > 0 else 0.0
    return norm

# ---------- Hybrid core ----------
def compute_selection_score(evidence: MathEvidence,
                            hypothesis: MathHypothesis,
                            urgency_factor: float,
                            steps: int = 12) -> float:
    """
    Translate the raw `strength` of an evidence item into a physics‑based
    admission score.  The hypothesis posterior provides a dynamic drag
    cost (higher confidence → larger drag).
    """
    # Convert raw strength to a likelihood ratio (simple linear scaling)
    base_lr = max(0.0, min(1.0, evidence.strength))
    # Drag cost grows with the current posterior (more certain hypotheses are
    # harder to overturn).
    drag_cost = hypothesis.posterior * 0.5  # tunable factor
    # Urgency is an external knob that can amplify the force.
    urgency = urgency_factor * base_lr
    score = burst_admission_score(work_value=base_lr,
                                  cost_drag=drag_cost,
                                  urgency_force=urgency,
                                  steps=steps)
    return score

def prune_and_select(evidence_set: List[MathEvidence],
                     hypothesis: MathHypothesis,
                     t: float,
                     lam: float,
                     alpha: float,
                     urgency_factor: float,
                     selection_threshold: float = 0.3) -> List[MathEvidence]:
    """
    Apply the exponential pruning schedule, then keep only those evidence
    items whose physics‑based selection score exceeds `selection_threshold`.
    """
    # Stage 1: stochastic pruning based on time.
    pruned = prune_evidence(evidence_set, t, lam, alpha)
    # Stage 2: deterministic selection using the ambush‑strike score.
    selected = []
    for ev in pruned:
        score = compute_selection_score(ev, hypothesis, urgency_factor)
        if score >= selection_threshold:
            selected.append(ev)
    return selected

def hybrid_update(hypotheses: List[MathHypothesis],
                  evidence_pool: List[MathEvidence],
                  t: float,
                  lam: float = 1.0,
                  alpha: float = 0.2,
                  urgency_factor: float = 1.0,
                  steps: int = 12,
                  selection_threshold: float = 0.3) -> List[MathHypothesis]:
    """
    For each hypothesis, select evidence via the hybrid prune/score pipeline
    and perform a Bayesian update where the likelihood ratio is modulated by
    the selection score.
    """
    updated = []
    for hyp in hypotheses:
        # Choose evidence relevant to this hypothesis (here we treat the pool as common)
        selected = prune_and_select(evidence_pool, hyp, t, lam, alpha,
                                    urgency_factor, selection_threshold)
        new_hyp = hyp
        for ev in selected:
            # Base likelihood from evidence strength
            base_lr = max(0.0, min(1.0, ev.strength))
            # Physics‑based modifier
            modifier = compute_selection_score(ev, hyp, urgency_factor, steps)
            effective_lr = base_lr * (1.0 + modifier)  # boost proportional to score
            new_hyp = update_hypothesis(new_hyp, ev, effective_lr)
        updated.append(new_hyp)
    return updated

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create a tiny hypothesis set
    hyps = [
        MathHypothesis(id="H1", prior=0.5, posterior=0.5, evidence_ids=[]),
        MathHypothesis(id="H2", prior=0.2, posterior=0.2, evidence_ids=[])
    ]

    # Generate synthetic evidence with random strengths
    rng = random.Random(42)
    ev_pool = [MathEvidence(id=f"E{i}", strength=rng.random()) for i in range(10)]

    # Run the hybrid update at time t=3.0
    updated_hyps = hybrid_update(
        hypotheses=hyps,
        evidence_pool=ev_pool,
        t=3.0,
        lam=1.0,
        alpha=0.15,
        urgency_factor=0.8,
        steps=12,
        selection_threshold=0.25
    )

    for h in updated_hyps:
        print(f"Hypothesis {h.id}: posterior={h.posterior:.4f}, evidence={h.evidence_ids}")