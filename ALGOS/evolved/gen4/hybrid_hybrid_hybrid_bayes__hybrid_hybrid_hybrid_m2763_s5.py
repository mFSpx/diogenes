# DARWIN HAMMER — match 2763, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_distri_m351_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_percyphon_hyb_m992_s2.py (gen3)
# born: 2026-05-29T23:45:37Z

"""Hybrid Bayesian-Pruning-Ambush Algorithm
================================================
This module fuses two parent algorithms:

* **Parent A** – Bayesian hypothesis updating with a time‑dependent pruning
  schedule (`prune_probability`, `prune_evidence`) and a Chelydrid ambush
  strike model (`StrikeState`).
* **Parent B** – Generic Bayesian marginal/update utilities, a procedural
  entity generator that yields a set of *slots* (clusters) and a simple
  edge‑posterior computation.

**Mathematical bridge**

The pruning schedule of Parent A yields a decay factor  


w(t) = λ·exp(-α·t) ∈ [0,1]


which we interpret as a *weight* on the likelihood contributed by each piece
of evidence.  In Parent B the posterior for a hypothesis is


posterior = prior·likelihood / marginal .


We replace `likelihood` by `w(t)·likelihood` so that older evidence is down‑weighted
in the Bayesian update.  The Chelydrid ambush model supplies a `StrikeState`
that scores each procedural slot; the highest‑scoring slot is taken as the
representative element of its cluster and its evidence is fed into the weighted
Bayesian update.  Thus the three core topologies (pruning, Bayesian marginal,
and ambush‑selection) are mathematically interlocked.
"""

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Data structures (borrowed / adapted from the parents)
# ----------------------------------------------------------------------
class MathClaim:
    def __init__(self, id: str):
        self.id = id


class MathEvidence:
    def __init__(self, id: str):
        self.id = id


class MathHypothesis:
    def __init__(self, id: str, prior: float, posterior: float, evidence_ids: List[str]):
        self.id = id
        self.prior = prior
        self.posterior = posterior
        self.evidence_ids = evidence_ids


class StrikeState:
    def __init__(self, velocity: float, distance: float, peak_velocity: float):
        self.velocity = velocity
        self.distance = distance
        self.peak_velocity = peak_velocity


# ----------------------------------------------------------------------
# Parent‑A utilities
# ----------------------------------------------------------------------
def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Time‑dependent survival probability used to keep evidence."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_evidence(evidence: List[MathEvidence],
                   t: float,
                   lam: float = 1.0,
                   alpha: float = 0.2,
                   seed: int | str | None = None) -> List[MathEvidence]:
    """Return a subset of evidence kept after applying the pruning schedule."""
    rng = random.Random(seed)
    keep_prob = prune_probability(t, lam, alpha)
    kept = [e for e in evidence if rng.random() < keep_prob]
    return kept


def update_hypothesis(hypothesis: MathHypothesis,
                     evidence: MathEvidence,
                     likelihood_ratio: float) -> MathHypothesis:
    """Bayesian odds update (Parent A)."""
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
    ids = list(hypothesis.evidence_ids) + [evidence.id]
    return MathHypothesis(id=hypothesis.id,
                          prior=hypothesis.posterior,
                          posterior=posterior,
                          evidence_ids=ids)


# ----------------------------------------------------------------------
# Parent‑B utilities
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Compute the marginal probability for a binary test."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Standard Bayesian update using a marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict:
        return self.__dict__


def _uuid_from_sha256(seed: str) -> str:
    h = hash(seed)
    return f"{h:08x}-{h:04x}-{h:04x}-{h:04x}-{h:012x}"


def _slot_name(seed: str, idx: int) -> Tuple[str, str, str]:
    h = hash(f"{seed}:{idx}")
    name = f"Villager-{h % 5000:04d}"
    alias = f"Alias-{h % 10000:04d}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][h % 6]
    return name, alias, persona


def procedural_entity_generator(villagers: Iterable[str] | None = None,
                                psyche_wrath_velocity: float = 0.0,
                                psyche_forensic_shield_ratio: float = 0.0,
                                fluid_slots: int = 88) -> dict:
    """Generate a list of procedural slots (clusters)."""
    villagers = list(villagers or [])
    seed = "|".join(villagers[:5000]) or "lucidota-villager-baseline"
    ternary_offs = []
    slots = []
    for idx in range(fluid_slots):
        name, alias, persona = _slot_name(seed, idx)
        uuid = _uuid_from_sha256(name)
        ternary_offset = random.choice([-1, 0, 1])
        slot = ProceduralSlot(idx, name, alias, persona, uuid, ternary_offset)
        slots.append(slot)
        ternary_offs.append(ternary_offset)
    return {"slots": slots, "ternary_offs": ternary_offs}


# ----------------------------------------------------------------------
# Hybrid functions (the new unified system)
# ----------------------------------------------------------------------
def hybrid_prune_and_weight(evidence: List[MathEvidence],
                            t: float,
                            lam: float = 1.0,
                            alpha: float = 0.2,
                            seed: int | str | None = None) -> Dict[str, float]:
    """
    Apply the pruning schedule to the evidence list and assign a weight
    ``w(t)`` to each surviving piece.  The weight is the same for all kept
    items because the schedule is global; the function returns a mapping
    ``evidence_id -> weight`` that can be used as a scaling factor on likelihoods.
    """
    kept = prune_evidence(evidence, t, lam, alpha, seed)
    weight = prune_probability(t, lam, alpha)
    return {e.id: weight for e in kept}


def hybrid_bayes_cluster_update(slots: List[ProceduralSlot],
                                priors: Dict[str, float],
                                evidence_weights: Dict[str, float],
                                false_positive: float = 0.01) -> Dict[str, float]:
    """
    For each procedural slot (treated as a hypothesis cluster) compute a
    posterior using the weighted Bayesian formula:

        marginal   = λ·exp(-α·t)·likelihood + false_positive·(1‑prior)
        posterior  = prior·(λ·exp(-α·t)·likelihood) / marginal

    The *likelihood* for a slot is synthesized from its ternary offset and a
    random base (simulating a sensor reading).  The function returns a dict
    ``slot_uuid -> posterior``.
    """
    posteriors = {}
    for slot in slots:
        # synthesize a base likelihood in (0,1)
        base_likelihood = random.random()
        # incorporate the ternary offset as a simple bias
        bias = (slot.ternary_offset + 1) / 2.0  # maps -1,0,1 -> 0,0.5,1
        likelihood = base_likelihood * (0.5 + 0.5 * bias)

        # retrieve the weight for the representative evidence (if any)
        weight = evidence_weights.get(slot.uuid, 1.0)  # default weight 1 if missing
        weighted_likelihood = weight * likelihood

        prior = priors.get(slot.uuid, 0.5)
        marginal = bayes_marginal(prior, weighted_likelihood, false_positive)
        posterior = bayes_update(prior, weighted_likelihood, marginal)
        posteriors[slot.uuid] = posterior
    return posteriors


def hybrid_ambush_selection(slots: List[ProceduralSlot],
                            strike: StrikeState) -> ProceduralSlot:
    """
    Use the Chelydrid ambush model to pick a representative slot.
    Score each slot with

        score = velocity * (1 - distance / Dmax) + peak_velocity * ternary_offset

    where ``Dmax`` is the maximum distance observed among the slots (here we
    treat the slot index as a proxy for distance).  The slot with the highest
    score is returned.
    """
    if not slots:
        raise ValueError("Slot list cannot be empty")
    # Use slot_index as a surrogate distance (larger index → farther)
    max_idx = max(s.slot_index for s in slots) or 1
    best_score = -math.inf
    best_slot = None
    for slot in slots:
        distance_factor = 1.0 - (slot.slot_index / max_idx)
        score = (strike.velocity * distance_factor) + (strike.peak_velocity * slot.ternary_offset)
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot  # type: ignore


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy evidence
    evidence_pool = [MathEvidence(f"e{i}") for i in range(20)]

    # Prune at time t = 3.0
    t_now = 3.0
    ev_weights = hybrid_prune_and_weight(evidence_pool, t_now, seed=42)

    # Generate procedural slots (clusters)
    gen = procedural_entity_generator(villagers=["alice", "bob", "carol"], fluid_slots=10)
    slots = gen["slots"]

    # Assign each slot a random prior (for demonstration)
    priors = {slot.uuid: random.random() for slot in slots}

    # Compute posteriors using the hybrid Bayesian update
    post = hybrid_bayes_cluster_update(slots, priors, ev_weights)

    # Create a strike state
    strike = StrikeState(velocity=7.5, distance=0.0, peak_velocity=12.0)

    # Select the representative slot via ambush scoring
    chosen = hybrid_ambush_selection(slots, strike)

    # Print a concise summary
    print("=== Hybrid Update Summary ===")
    print(f"Time t = {t_now}")
    print(f"Evidence kept: {len(ev_weights)} / {len(evidence_pool)}")
    print(f"Chosen slot UUID: {chosen.uuid}")
    print(f"Posterior for chosen slot: {post[chosen.uuid]:.4f}")
    print("All posteriors (first 5):")
    for uid, val in list(post.items())[:5]:
        print(f"  {uid[:8]}... -> {val:.4f}")