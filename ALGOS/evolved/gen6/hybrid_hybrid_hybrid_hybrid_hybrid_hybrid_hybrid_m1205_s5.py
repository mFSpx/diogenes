# DARWIN HAMMER — match 1205, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_cockpit_metri_m1060_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s0.py (gen5)
# born: 2026-05-29T23:34:25Z

"""Hybrid Algorithm Fusion of PARENT ALGORITHM A and PARENT ALGORITHM B.

Mathematical Bridge
-------------------
Algorithm A provides a simulated‑annealing style acceptance probability  
    p_accept = exp(-ΔE / T)                     (1)

where the temperature *T* can be shaped by an *anti‑slop ratio* (a schedule
derived from broadcast_probability).  

Algorithm B introduces a trust‑weighted linguistic similarity  

    sim_trust = trust * Jaccard(text, action_id) (2)

which is used to rank models for eviction in a pool.

The fusion replaces the raw temperature *T* of (1) by a *hybrid temperature*
that is scaled by the similarity score (2).  Consequently the acceptance
probability becomes

    p_hybrid = exp(-ΔE / (T·sim_trust))         (3)

If *sim_trust* is small the system is more conservative (lower acceptance);
if it is large the system accepts more aggressively.  The same hybrid score
drives the eviction policy of the model pool: the model with the smallest
(p_hybrid, Hoeffding bound) product is evicted first.

The code below implements the fused mathematics together with three
demonstrative functions and a smoke‑test.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Mapping, Set, Hashable, Dict, Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Core utilities taken from Algorithm A
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]

def broadcast_probability(phase: int, step: int) -> float:
    """Schedule used as an anti‑slop ratio."""
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Standard Metropolis acceptance."""
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

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for empirical mean estimation."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Core utilities taken from Algorithm B
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
    text: str                       # textual description used for similarity
    trust: float = 1.0              # trust score ∈ (0, 1]

class ModelPool:
    """A RAM‑bounded pool that evicts models using the hybrid score."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model respecting tier exclusivity and RAM ceiling."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier,
                           trust_lookup: Dict[str, float],
                           phase: int,
                           step: int) -> None:
        """
        Load a model, evicting the least promising resident according to the
        hybrid score (see `hybrid_eviction_score`).
        """
        while (model.ram_mb + self._used() > self.ram_ceiling_mb) and self.loaded:
            # pick victim with minimal hybrid score
            victim_name = min(
                self.loaded,
                key=lambda n: hybrid_eviction_score(
                    self.loaded[n],
                    trust_lookup.get(n, 1.0),
                    phase,
                    step
                )
            )
            del self.loaded[victim_name]
        self.load(model)

# ----------------------------------------------------------------------
# Fusion mathematics
# ----------------------------------------------------------------------
def trust_weighted_similarity(action: MathAction, model: ModelTier) -> float:
    """
    Jaccard similarity between the token sets of `action.id` and `model.text`,
    scaled by the model's trust score.
    """
    act_tokens = set(action.id.lower().split('_'))
    txt_tokens = set(model.text.lower().split())
    if not act_tokens and not txt_tokens:
        base = 1.0
    else:
        base = len(act_tokens & txt_tokens) / len(act_tokens | txt_tokens)
    return model.trust * base

def hybrid_temperature(base_temp: float,
                       anti_slop: float,
                       similarity: float) -> float:
    """
    Modulate the base temperature by the anti‑slop ratio and the similarity.
    The formula mirrors (3) where temperature is effectively divided by the
    similarity factor.
    """
    if similarity <= 0:
        # avoid division by zero – fall back to a very low temperature
        return 1e-9
    return base_temp * anti_slop / similarity

def hybrid_acceptance(delta_e: float,
                      k: int,
                      phase: int,
                      step: int,
                      action: MathAction,
                      model: ModelTier) -> float:
    """
    Compute the hybrid acceptance probability using:
        T   = cooling_temperature(k)
        a   = broadcast_probability(phase, step)   # anti‑slop
        s   = trust_weighted_similarity(action, model)
        T'  = hybrid_temperature(T, a, s)
    """
    T = cooling_temperature(k)
    a = broadcast_probability(phase, step)
    s = trust_weighted_similarity(action, model)
    T_prime = hybrid_temperature(T, a, s)
    return acceptance_probability(delta_e, T_prime)

def hybrid_eviction_score(model: ModelTier,
                          trust: float,
                          phase: int,
                          step: int,
                          r: float = 1.0,
                          delta: float = 0.05,
                          n: int = 100) -> float:
    """
    A scalar that combines the Hoeffding bound (uncertainty) with the inverse
    of the similarity‑derived temperature.  Smaller scores are less valuable
    and are evicted first.
    """
    # surrogate action to reuse similarity logic; the actual id is irrelevant
    dummy_action = MathAction(id="eviction_dummy", expected_value=0.0)
    # temporary model with provided trust (overrides its own trust)
    tmp_model = ModelTier(name=model.name,
                          ram_mb=model.ram_mb,
                          tier=model.tier,
                          text=model.text,
                          trust=trust)
    sim = trust_weighted_similarity(dummy_action, tmp_model) + 1e-12
    anti = broadcast_probability(phase, step)
    temp = hybrid_temperature(1.0, anti, sim)          # base temperature = 1.0
    bound = hoeffding_bound(r, delta, n)
    return bound / temp

# ----------------------------------------------------------------------
# Demonstrative functions
# ----------------------------------------------------------------------
def simulate_step(delta_e: float,
                  k: int,
                  phase: int,
                  step: int,
                  action: MathAction,
                  model: ModelTier) -> bool:
    """Run a single Monte‑Carlo step returning whether the move is accepted."""
    prob = hybrid_acceptance(delta_e, k, phase, step, action, model)
    return random.random() < prob

def evaluate_pool(pool: ModelPool,
                  candidate: ModelTier,
                  trust_lookup: Dict[str, float],
                  phase: int,
                  step: int) -> Tuple[bool, List[str]]:
    """
    Attempt to load `candidate` into `pool`.  Returns a tuple:
        (load_successful, list_of_evicted_model_names)
    """
    evicted = []
    if candidate.ram_mb + pool._used() <= pool.ram_ceiling_mb:
        pool.load(candidate)
        return True, evicted

    # perform eviction loop manually to record evicted names
    while candidate.ram_mb + pool._used() > pool.ram_ceiling_mb and pool.loaded:
        victim = min(
            pool.loaded,
            key=lambda n: hybrid_eviction_score(
                pool.loaded[n],
                trust_lookup.get(n, 1.0),
                phase,
                step
            )
        )
        evicted.append(victim)
        del pool.loaded[victim]

    try:
        pool.load(candidate)
        return True, evicted
    except Exception:
        return False, evicted

def batch_hybrid_decision(delta_es: List[float],
                          actions: List[MathAction],
                          model: ModelTier,
                          k_start: int = 0,
                          phase: int = 3,
                          step: int = 1) -> List[bool]:
    """
    Process a batch of energy deltas with increasing cooling steps.
    Returns a list of booleans indicating acceptance for each entry.
    """
    results = []
    k = k_start
    for delta_e, act in zip(delta_es, actions):
        results.append(simulate_step(delta_e, k, phase, step, act, model))
        k += 1
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create dummy actions
    actions = [
        MathAction(id="move_left", expected_value=1.2),
        MathAction(id="jump_up", expected_value=0.5),
        MathAction(id="shoot_enemy", expected_value=2.0)
    ]

    # Energy differences for the actions
    delta_es = [0.3, -0.1, 1.5]

    # Model tier with textual description
    model = ModelTier(
        name="AlphaNet",
        ram_mb=1500,
        tier="T1",
        text="move left jump up shoot enemy",
        trust=0.85
    )

    # Run batch decision
    decisions = batch_hybrid_decision(delta_es, actions, model)
    print("Batch acceptance decisions:", decisions)

    # Initialise pool and attempt to load models
    pool = ModelPool(ram_ceiling_mb=4000)
    trust_map = {"AlphaNet": 0.85, "BetaNet": 0.60}
    pool.load(model)

    beta = ModelTier(
        name="BetaNet",
        ram_mb=2600,
        tier="T2",
        text="run fast dodge left",
        trust=0.60
    )

    success, evicted = evaluate_pool(pool, beta, trust_map, phase=3, step=1)
    print("Load BetaNet success:", success, "Evicted:", evicted)

    # Verify that hybrid_acceptance returns a sensible probability
    prob = hybrid_acceptance(delta_e=0.8, k=5, phase=3, step=1,
                             action=MathAction(id="run_fast", expected_value=0.9),
                             model=model)
    print("Hybrid acceptance probability:", prob)