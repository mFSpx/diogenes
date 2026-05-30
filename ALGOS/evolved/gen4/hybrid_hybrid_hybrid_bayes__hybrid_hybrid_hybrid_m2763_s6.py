# DARWIN HAMMER — match 2763, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_distri_m351_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_percyphon_hyb_m992_s2.py (gen3)
# born: 2026-05-29T23:45:37Z

"""Hybrid algorithm merging:
- Parent A: Bayesian claim update with pruning schedule and chelydrid ambush strike model.
- Parent B: Standard Bayesian marginal/update utilities and procedural slot generation.

Mathematical bridge:
The pruning probability `p_prune(t)` from Parent A is used as a scaling factor on the
likelihood ratio supplied by Parent B’s `bayes_marginal`/`bayes_update`.  The scaled
ratio modulates the odds update in `update_hypothesis`.  Additionally, the procedural
slots generated in Parent B provide per‑slot velocity/offset data that feed the
chelydrid ambush model (Parent A) to select a representative piece of evidence from
each similarity cluster before the Bayesian update is performed.
"""

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Data structures (shared)
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
# Parent A utilities (pruning & Bayesian odds update)
# ----------------------------------------------------------------------
def update_hypothesis(hypothesis: MathHypothesis,
                     evidence: MathEvidence,
                     likelihood_ratio: float) -> MathHypothesis:
    """Odds‑based Bayesian update using a likelihood ratio."""
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
    ids = hypothesis.evidence_ids + [evidence.id]
    return MathHypothesis(id=hypothesis.id,
                          prior=hypothesis.posterior,
                          posterior=posterior,
                          evidence_ids=ids)

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decay pruning schedule."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))

def prune_evidence(evidence: List[MathEvidence],
                   t: float,
                   lam: float = 1.0,
                   alpha: float = 0.2,
                   seed: int | str | None = None) -> List[MathEvidence]:
    """Randomly keep each piece of evidence with probability given by the schedule."""
    if seed is not None:
        random.seed(seed)
    keep_prob = prune_probability(t, lam, alpha)
    return [e for e in evidence if random.random() < keep_prob]

# ----------------------------------------------------------------------
# Parent B utilities (Bayesian marginal/update & procedural slots)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
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
                                fluid_slots: int = 88) -> Dict[str, List]:
    """Generate a list of procedural slots together with their ternary offsets."""
    villagers = list(villagers or [])
    seed = "|".join(villagers[:5000]) or "lucidota-villager-baseline"
    slots = []
    ternary_offs = []
    for idx in range(fluid_slots):
        name, alias, persona = _slot_name(seed, idx)
        uuid = _uuid_from_sha256(name)
        ternary_offset = random.choice([-1, 0, 1])
        slot = ProceduralSlot(idx, name, alias, persona, uuid, ternary_offset)
        slots.append(slot)
        ternary_offs.append(ternary_offset)
    return {"slots": slots, "ternary_offs": ternary_offs}

def bayes_edge_posterior(node: str,
                         edge: Tuple[str, str],
                         priors: Dict[str, float],
                         likelihoods: Dict[Tuple[str, str], float]) -> float:
    """Posterior probability for a node given an incident edge."""
    prior = priors.get(node, 0.5)
    likelihood = likelihoods.get(edge, 0.5)
    # Assume a simple false‑positive rate of 0.1 for demonstration
    false_positive = 0.1
    marginal = bayes_marginal(prior, likelihood, false_positive)
    return bayes_update(prior, likelihood, marginal)

# ----------------------------------------------------------------------
# Hybrid core functions (mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_prune_and_update(hypothesis: MathHypothesis,
                            evidence_pool: List[MathEvidence],
                            t: float,
                            lam: float = 1.0,
                            alpha: float = 0.2,
                            false_positive: float = 0.1,
                            seed: int | str | None = None) -> MathHypothesis:
    """
    1. Prune the evidence pool using Parent A's schedule.
    2. For each retained evidence compute a likelihood ratio via Parent B's
       marginal/update formulas.
    3. Scale the ratio by the pruning probability (the bridge) and perform the
       odds update from Parent A.
    Returns the updated hypothesis after processing the first retained evidence
    (for simplicity – the function can be called iteratively).
    """
    # Step 1 – prune
    pruned = prune_evidence(evidence_pool, t, lam, alpha, seed)

    if not pruned:
        return hypothesis  # nothing to update

    # Take the first piece for demonstration
    ev = pruned[0]

    # Step 2 – compute likelihood ratio
    # Use a dummy likelihood based on hash of evidence id (normalised)
    raw_likelihood = (hash(ev.id) % 1000) / 1000.0
    raw_likelihood = max(0.0, min(1.0, raw_likelihood))

    marginal = bayes_marginal(hypothesis.posterior, raw_likelihood, false_positive)
    posterior = bayes_update(hypothesis.posterior, raw_likelihood, marginal)

    # Likelihood ratio for odds update = posterior / prior (if prior > 0)
    if hypothesis.posterior > 0:
        lr = posterior / hypothesis.posterior
    else:
        lr = 0.0

    # Step 3 – bridge: scale by pruning probability at current time
    scale = prune_probability(t, lam, alpha)
    scaled_lr = lr * scale

    # Perform odds‑based update
    return update_hypothesis(hypothesis, ev, scaled_lr)

def generate_procedural_strike_state(num_slots: int = 10,
                                     base_velocity: float = 5.0) -> List[StrikeState]:
    """
    Create procedural slots (Parent B) and translate each slot into a StrikeState
    (Parent A).  The ternary offset influences the velocity; the distance is a
    random draw; the peak_velocity is the maximum of velocity and distance.
    """
    gen = procedural_entity_generator(fluid_slots=num_slots)
    slots: List[ProceduralSlot] = gen["slots"]
    strike_states: List[StrikeState] = []
    for slot in slots:
        # Velocity perturbed by ternary offset
        velocity = base_velocity + slot.ternary_offset * random.uniform(0.5, 1.5)
        distance = random.uniform(1.0, 10.0)
        peak = max(velocity, distance)
        strike_states.append(StrikeState(velocity, distance, peak))
    return strike_states

def select_representative_evidence(evidence_pool: List[MathEvidence],
                                   strike_states: List[StrikeState]) -> List[MathEvidence]:
    """
    Cluster evidence by the first character of their id (a cheap similarity proxy).
    For each cluster, use the corresponding StrikeState to compute a "strike strength"
    = velocity * distance.  The evidence whose hash-derived score is closest to the
    strike strength is chosen as the representative of that cluster.
    """
    # Build clusters
    clusters: Dict[str, List[MathEvidence]] = {}
    for ev in evidence_pool:
        key = ev.id[0] if ev.id else ""
        clusters.setdefault(key, []).append(ev)

    representatives: List[MathEvidence] = []
    for idx, (key, members) in enumerate(clusters.items()):
        # Cycle through strike states if fewer than clusters
        ss = strike_states[idx % len(strike_states)]
        target_strength = ss.velocity * ss.distance

        # Choose member whose hash score is nearest to target_strength
        best_ev = min(members,
                      key=lambda e: abs((hash(e.id) % 1000) - target_strength))
        representatives.append(best_ev)
    return representatives

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy hypothesis
    hyp = MathHypothesis(id="H1", prior=0.5, posterior=0.5, evidence_ids=[])

    # Build a pool of synthetic evidence
    ev_pool = [MathEvidence(id=f"E{chr(65 + i%26)}{i}") for i in range(30)]

    # Hybrid prune‑update step
    updated_hyp = hybrid_prune_and_update(hypothesis=hyp,
                                          evidence_pool=ev_pool,
                                          t=2.0,
                                          lam=1.0,
                                          alpha=0.3,
                                          false_positive=0.1,
                                          seed=42)

    print(f"Updated hypothesis posterior: {updated_hyp.posterior:.4f}")
    print(f"Evidence incorporated: {updated_hyp.evidence_ids}")

    # Generate strike states and pick representatives
    strikes = generate_procedural_strike_state(num_slots=8, base_velocity=4.0)
    reps = select_representative_evidence(ev_pool, strikes)

    print(f"Selected {len(reps)} representative evidences:")
    for r in reps:
        print(f"  {r.id}")

    # Verify that a posterior can be computed for an arbitrary edge
    priors = {"nodeA": 0.6}
    likelihoods = {("nodeA", "nodeB"): 0.8}
    post = bayes_edge_posterior("nodeA", ("nodeA", "nodeB"), priors, likelihoods)
    print(f"Edge posterior (nodeA→nodeB): {post:.4f}")