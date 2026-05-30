# DARWIN HAMMER — match 4061, survivor 1
# gen: 4
# parent_a: bayes_claim_kernel.py (gen0)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_temporal_moti_m1701_s0.py (gen3)
# born: 2026-05-29T23:53:20Z

"""Hybrid Bayesian-Workshare Engine
Parents:
- bayes_claim_kernel.py (Algorithm A): Bayesian posterior update using likelihood ratios.
- hybrid_hybrid_ternar_hybrid_temporal_moti_m1701_s0.py (Algorithm B): SSIM similarity, temporal motif support, and workshare allocation.

Mathematical Bridge:
Algorithm B provides a similarity metric (SSIM) between a claim vector and temporal‑motif vectors.
Algorithm A consumes a *likelihood ratio* to update a hypothesis posterior.
We convert the SSIM (∈[0,1]) into a likelihood ratio via the odds transform  

    LR = SSIM / (1‑SSIM + ε)

and feed it to the Bayesian update.  
The resulting posterior then weights the support counts of the motifs when allocating
work units across predefined groups.  This creates a single pipeline:

claim → SSIM → LR → Bayesian update → weighted support → workshare allocation

The module implements this fused workflow with three core functions and a smoke test.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import date
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Simple data models (stand‑alone replacements for .types)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathClaim:
    id: str
    vector: np.ndarray  # representation of the claim


@dataclass(frozen=True)
class MathEvidence:
    id: str
    vector: np.ndarray  # representation used for similarity (temporal motif)


@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float       # previous posterior (used as prior for next step)
    posterior: float   # current posterior probability
    evidence_ids: Tuple[str, ...] = ()


# ----------------------------------------------------------------------
# Algorithm B – similarity and allocation utilities
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index between two 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator)


GROUPS = ("codex", "groq", "cohere", "local_models")


def allocate_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
    weighted_support: Dict[str, float],
) -> Dict[str, float]:
    """
    Allocate ``total_units`` among ``groups`` proportionally to the
    ``weighted_support`` values.  A deterministic target percentage forces
    a minimum share for each group to avoid zero allocation.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("at least one group required")

    # Ensure every group appears in the dict; missing groups get zero weight.
    for g in groups:
        weighted_support.setdefault(g, 0.0)

    total_weight = sum(weighted_support.values())
    if total_weight == 0:
        # Uniform fallback when all weights are zero.
        base = total_units / len(groups)
        return {g: base for g in groups}

    # Deterministic floor for each group.
    floor_share = total_units * (deterministic_target_pct / 100.0) / len(groups)
    remaining = total_units - floor_share * len(groups)

    allocation = {}
    for g in groups:
        weight = weighted_support[g]
        proportional = (weight / total_weight) * remaining if total_weight else 0.0
        allocation[g] = floor_share + proportional
    return allocation


# ----------------------------------------------------------------------
# Algorithm A – Bayesian update (unchanged core logic)
# ----------------------------------------------------------------------
def update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    likelihood_ratio: float,
) -> MathHypothesis:
    """Bayesian update of a hypothesis posterior given a likelihood ratio."""
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
    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=ids,
    )


# ----------------------------------------------------------------------
# Hybrid bridge: SSIM → likelihood ratio → Bayesian → weighted allocation
# ----------------------------------------------------------------------
def compute_log_likelihood_ratio(
    claim: MathClaim,
    hypothesis_id: str,
    evidences: List[MathEvidence],
    epsilon: float = 1e-12,
) -> float:
    """
    Convert SSIM similarity between the claim and each evidence vector into a
    log‑likelihood ratio.  The average SSIM is used as a proxy for the
    evidence strength for the given hypothesis.
    """
    if not evidences:
        raise ValueError("evidence list cannot be empty")
    ssim_values = [compute_ssim(claim.vector, ev.vector) for ev in evidences]
    avg_ssim = float(np.mean(ssim_values))
    # Odds transform to a likelihood ratio; guard against division by zero.
    lr = avg_ssim / (1.0 - avg_ssim + epsilon)
    return math.log(lr + epsilon)


def hybrid_update_and_allocate(
    hypothesis: MathHypothesis,
    claim: MathClaim,
    evidences: List[MathEvidence],
    total_units: float,
) -> Tuple[MathHypothesis, Dict[str, float]]:
    """
    1. Compute a likelihood ratio from SSIM similarity (bridge).
    2. Perform Bayesian posterior update.
    3. Weight each group’s support by the updated posterior.
    4. Allocate work units proportionally.
    """
    # Step 1 – similarity to likelihood ratio (log‑space then exp back)
    log_lr = compute_log_likelihood_ratio(claim, hypothesis.id, evidences)
    lr = math.exp(log_lr)

    # For simplicity we treat the first evidence as the one that triggered the update.
    evidence = evidences[0]

    # Step 2 – Bayesian update
    updated_hyp = update_hypothesis(hypothesis, evidence, lr)

    # Step 3 – build weighted support per group.
    # Assume each evidence carries an implicit group label encoded in its id prefix.
    weighted_support: Dict[str, float] = {g: 0.0 for g in GROUPS}
    for ev in evidences:
        # Extract group name from id (e.g., "codex_42")
        prefix = ev.id.split("_")[0]
        if prefix in weighted_support:
            # Support contributed proportionally to current posterior.
            weighted_support[prefix] += updated_hyp.posterior

    # Step 4 – allocation
    allocation = allocate_workshare(
        total_units=total_units,
        deterministic_target_pct=90.0,
        groups=GROUPS,
        weighted_support=weighted_support,
    )
    return updated_hyp, allocation


# ----------------------------------------------------------------------
# Additional demonstration function
# ----------------------------------------------------------------------
def simulate_hybrid_process(seed: int = 42) -> None:
    """Run a small Monte‑Carlo style demo of the hybrid pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    # Create a random claim vector.
    claim = MathClaim(id="claim_1", vector=np.random.rand(10))

    # Generate synthetic evidences (temporal motifs) with group tags.
    evidences = []
    for i in range(8):
        group = random.choice(GROUPS)
        ev = MathEvidence(
            id=f"{group}_{i}",
            vector=np.random.rand(10) * (0.5 + 0.5 * random.random()),
        )
        evidences.append(ev)

    # Initial hypothesis with a neutral prior.
    hypothesis = MathHypothesis(
        id="hyp_001", prior=0.5, posterior=0.5, evidence_ids=()
    )

    updated_hyp, allocation = hybrid_update_and_allocate(
        hypothesis=hypothesis,
        claim=claim,
        evidences=evidences,
        total_units=1000.0,
    )

    print("Updated hypothesis posterior:", updated_hyp.posterior)
    print("Evidence ids used:", updated_hyp.evidence_ids)
    print("Workshare allocation:")
    for grp, units in allocation.items():
        print(f"  {grp}: {units:.2f} units")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    simulate_hybrid_process()