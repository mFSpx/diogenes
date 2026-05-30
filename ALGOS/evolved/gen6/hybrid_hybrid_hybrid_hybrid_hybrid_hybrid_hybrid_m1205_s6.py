# DARWIN HAMMER — match 1205, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_cockpit_metri_m1060_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py (gen5)
# born: 2026-05-29T23:34:25Z

"""
Hybrid Algorithm Fusion of:
- PARENT ALGORITHM A: hybrid_hybrid_hybrid_distri_hybrid_cockpit_metri_m1060_s2.py
- PARENT ALGORITHM B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py

Mathematical Bridge:
The acceptance probability from simulated‑annealing (A) determines whether a
model transition is accepted.  The temperature controlling that probability is
modulated by a trust‑weighted linguistic similarity measure (B) computed between
the candidate model and the currently loaded models.  High similarity (i.e. a
trusted, linguistically close model) lowers the effective temperature, making
acceptance more likely, while low similarity raises the temperature,
encouraging exploration.  This creates a unified decision rule that fuses
probabilistic annealing with trust‑aware model‑pool management.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Mapping, Hashable, Set, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Core types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

# ----------------------------------------------------------------------
# Functions from Parent A
# ----------------------------------------------------------------------
def broadcast_probability(phase: int, step: int) -> float:
    """Probability that a pruning broadcast occurs at a given phase/step."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Standard Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


# ----------------------------------------------------------------------
# Functions from Parent B (trust‑weighted linguistic similarity)
# ----------------------------------------------------------------------
def _tokenize(text: str) -> Set[str]:
    """Very simple whitespace tokenizer."""
    return set(text.lower().split())


def trust_weighted_similarity(
    text_a: str,
    text_b: str,
    trust_a: float,
    trust_b: float,
) -> float:
    """
    Jaccard‑like similarity weighted by trust scores.
    similarity = (|A ∩ B| * sqrt(trust_a * trust_b)) / |A ∪ B|
    Returns a value in [0, 1].
    """
    if not (0 <= trust_a <= 1 and 0 <= trust_b <= 1):
        raise ValueError("trust scores must be in [0, 1]")
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    weight = math.sqrt(trust_a * trust_b)
    return (len(intersection) * weight) / len(union)


def aggregate_similarity_to_pool(
    candidate_text: str,
    candidate_trust: float,
    pool: "ModelPool",
) -> float:
    """
    Average trust‑weighted similarity between a candidate model and all models
    currently resident in the pool.  If the pool is empty, returns 0.0.
    """
    if not pool.loaded:
        return 0.0
    sims = [
        trust_weighted_similarity(
            candidate_text,
            m.text,
            candidate_trust,
            m.trust,
        )
        for m in pool.loaded.values()
    ]
    return float(np.mean(sims))


# ----------------------------------------------------------------------
# Data structures (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    text: str               # descriptive text for similarity
    trust: float = 0.5      # trust score in [0,1]


class ModelPool:
    """
    Simple RAM‑bounded pool with mutual‑exclusivity rule for T2/T3 tiers.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Evicts least‑recently‑added models until enough RAM is freed, then loads.
        """
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # pop arbitrary (FIFO) element
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)


# ----------------------------------------------------------------------
# Hybrid Operations (the new fused logic)
# ----------------------------------------------------------------------
def hybrid_temperature(
    base_temp: float,
    similarity: float,
    beta: float = 0.7,
) -> float:
    """
    Modulates the annealing temperature with linguistic similarity.
    similarity ∈ [0,1]; higher similarity → lower temperature.
    T_eff = base_temp * (1 - beta * similarity)
    """
    if not (0 <= similarity <= 1):
        raise ValueError("similarity must be in [0,1]")
    return max(1e-8, base_temp * (1.0 - beta * similarity))


def hybrid_decision(
    delta_e: float,
    step: int,
    phase: int,
    base_temp: float,
    candidate: ModelTier,
    pool: ModelPool,
) -> bool:
    """
    Returns True if the candidate model should be admitted.
    1. Compute a cooling temperature for the current step.
    2. Adjust it with similarity to the pool.
    3. Compute acceptance probability.
    4. Optionally enforce a pruning broadcast probability.
    """
    # 1. Cooling schedule
    temp = cooling_temperature(step, t0=base_temp)

    # 2. Similarity‑driven modulation
    sim = aggregate_similarity_to_pool(candidate.text, candidate.trust, pool)
    temp_eff = hybrid_temperature(temp, sim)

    # 3. Acceptance probability
    acc_prob = acceptance_probability(delta_e, temp_eff)

    # 4. Broadcast‑driven pruning chance
    broadcast_prob = broadcast_probability(phase, step)

    # Final decision: accept if a random draw falls under both probabilities
    decision_threshold = acc_prob * broadcast_prob
    return random.random() < decision_threshold


def process_candidate_model(
    candidate: ModelTier,
    pool: ModelPool,
    delta_e: float,
    step: int,
    phase: int,
    base_temp: float = 1.0,
) -> str:
    """
    Attempts to admit *candidate* into *pool* using the hybrid decision rule.
    Returns a short status message.
    """
    try:
        if hybrid_decision(delta_e, step, phase, base_temp, candidate, pool):
            pool.load_with_eviction(candidate)
            return f"Model {candidate.name} admitted."
        else:
            return f"Model {candidate.name} rejected by hybrid policy."
    except Exception as exc:
        return f"Model {candidate.name} failed to load: {exc}"


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Existing model (acts as similarity anchor)
    existing = ModelTier(
        name="base_v1",
        ram_mb=500,
        tier="T1",
        text="the quick brown fox jumps over the lazy dog",
        trust=0.9,
    )
    pool.load(existing)

    # Candidate models
    candidates = [
        ModelTier(
            name="candidate_a",
            ram_mb=600,
            tier="T2",
            text="the quick brown fox leaps over a lazy dog",
            trust=0.8,
        ),
        ModelTier(
            name="candidate_b",
            ram_mb=1200,
            tier="T3",
            text="quantum entanglement and relativistic frames",
            trust=0.4,
        ),
        ModelTier(
            name="candidate_c",
            ram_mb=900,
            tier="T1",
            text="the lazy dog sleeps under the quick brown fox",
            trust=0.95,
        ),
    ]

    # Simulated energy differences for each candidate
    delta_es = [0.2, 1.5, -0.1]

    for i, (cand, de) in enumerate(zip(candidates, delta_es), start=1):
        msg = process_candidate_model(
            candidate=cand,
            pool=pool,
            delta_e=de,
            step=i,
            phase=2,
            base_temp=1.0,
        )
        print(msg)

    # Final pool summary
    print("\nFinal pool contents:")
    for m in pool.loaded.values():
        print(f"- {m.name} (RAM {m.ram_mb} MB, tier {m.tier}, trust {m.trust})")