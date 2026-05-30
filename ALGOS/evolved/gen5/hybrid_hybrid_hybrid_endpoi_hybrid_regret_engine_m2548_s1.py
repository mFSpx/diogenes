# DARWIN HAMMER — match 2548, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2.py (gen4)
# parent_b: hybrid_regret_engine_hybrid_liquid_time_c_m13_s1.py (gen2)
# born: 2026-05-29T23:42:50Z

"""Hybrid Algorithm: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s2 + hybrid_regret_engine_hybrid_liquid_time_c_m13_s1
This module fuses two parent algorithms:

* **Parent A** – circuit‑breaker primitives with a morphological Fisher score that
  modulates pruning probabilities.
* **Parent B** – regret‑weighted strategy whose hidden state is projected via MinHash
  similarity.

**Mathematical bridge**

1. The Fisher score `F(morph)` (a scalar derived from the `Morphology` object) is used
   as a multiplicative factor on the regret‑weighted values `R_i` of each action.
2. MinHash similarity `S(i, R)` between the tokenised identifier of an action `i`
   and a reference signature `R` (computed from all actions) is used as an
   additional scaling factor.
3. The final hybrid weight for action `i` is  

   `W_i = R_i * (1 + F) * (1 + S(i,R))`.

4. The circuit‑breaker `EndpointCircuitBreaker` consumes the hybrid weight:
   if `W_i` falls below a dynamic threshold derived from the prune probability,
   a failure is recorded, otherwise a success resets the breaker.

The three core functions below implement this fused mathematics.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict
import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be a positive number")

class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = ""
        if self.failures >= self.failure_threshold:
            self.open = True

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str                     # tokenisable identifier
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Helper functions (shared)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

# ----------------------------------------------------------------------
# 1. Fisher score derived from morphology
# ----------------------------------------------------------------------
def fisher_score(morph: Morphology) -> float:
    """
    Compute a dimensionless Fisher‑like score.
    The formula uses the volume‑to‑mass ratio, normalised to (0,1].

    F = (V / mass) / max(V / mass) over a reasonable range.
    For simplicity we bound the raw ratio by 1.0.
    """
    volume = morph.length * morph.width * morph.height
    raw = volume / morph.mass
    # Clamp to [0,1] to keep the scaling stable
    return min(1.0, raw)

# ----------------------------------------------------------------------
# 2. Prune probability that uses the Fisher score
# ----------------------------------------------------------------------
def prune_probability(breaker: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Logistic‑style probability that the breaker should open.
    Incorporates the current failure count and the Fisher score.

    p = sigmoid( (failures - threshold) + α·F )
    where α is a scaling constant (chosen = 2.0).
    """
    F = fisher_score(morph)
    α = 2.0
    x = (breaker.failures - breaker.failure_threshold) + α * F
    # Numerically stable sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)

# ----------------------------------------------------------------------
# 3. Regret‑weighted strategy (parent B core)
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    """
    Base regret‑weighted value for each action:
    V_i = expected_value - cost - risk + Σ_cf outcome_value·probability
    """
    cf_map = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf_map.get(a.id, 0.0)
        for a in actions
    }
    return vals

# ----------------------------------------------------------------------
# 4. Hybrid operation combining both parents
# ----------------------------------------------------------------------
def compute_hybrid_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> Dict[str, float]:
    """
    Produce hybrid weights W_i = R_i * (1+F) * (1+S_i)
    where:
        R_i – regret‑weighted value,
        F   – Fisher score from morphology,
        S_i – MinHash similarity of action i to the collective signature.
    The circuit‑breaker is updated based on a dynamic threshold derived from
    prune_probability().
    """
    # 1) Base regret values
    regret_vals = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2) Fisher scaling factor
    F = fisher_score(morph)

    # 3) Global MinHash signature built from all action tokens
    all_tokens = set()
    for a in actions:
        all_tokens.update(a.id.split('_'))
    global_sig = signature(all_tokens)

    # 4) Compute per‑action similarity and final weight
    hybrid_weights: Dict[str, float] = {}
    for a in actions:
        tokens = a.id.split('_')
        act_sig = signature(tokens)
        S = similarity(act_sig, global_sig)  # between 0 and 1
        W = regret_vals[a.id] * (1.0 + F) * (1.0 + S)
        hybrid_weights[a.id] = W

    # 5) Update circuit‑breaker based on a threshold
    threshold = np.percentile(list(hybrid_weights.values()), 25)  # 25th percentile
    avg_weight = float(np.mean(list(hybrid_weights.values())))

    # If the average hybrid weight is below the prune probability, record failure
    if avg_weight < prune_probability(breaker, morph):
        breaker.record_failure()
    else:
        breaker.record_success()

    return hybrid_weights

# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def simulate_step(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    morph: Morphology,
    breaker: EndpointCircuitBreaker,
) -> None:
    """Run one simulation step and print the hybrid weights and breaker state."""
    weights = compute_hybrid_weights(actions, counterfactuals, morph, breaker)
    print("Hybrid weights:")
    for aid, w in weights.items():
        print(f"  {aid}: {w:.4f}")
    print(f"CircuitBreaker – failures: {breaker.failures}, open: {breaker.open}")

def generate_random_actions(n: int) -> List[MathAction]:
    """Utility to create n random actions with tokenised identifiers."""
    actions = []
    for i in range(n):
        # identifier composed of three random tokens
        tokens = [random.choice(['alpha', 'beta', 'gamma', 'delta', 'epsilon']) for _ in range(3)]
        aid = "_".join(tokens) + f"_{i}"
        ev = random.uniform(0, 10)
        cost = random.uniform(0, 3)
        risk = random.uniform(0, 2)
        actions.append(MathAction(id=aid, expected_value=ev, cost=cost, risk=risk))
    return actions

def generate_counterfactuals(actions: List[MathAction]) -> List[MathCounterfactual]:
    """Create a simple counterfactual for each action."""
    cfs = []
    for a in actions:
        outcome = random.uniform(-5, 5)
        prob = random.uniform(0.5, 1.0)
        cfs.append(MathCounterfactual(action_id=a.id, outcome_value=outcome, probability=prob))
    return cfs

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Instantiate morphology and breaker
    morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)
    breaker = EndpointCircuitBreaker(failure_threshold=3)

    # Generate synthetic actions and counterfactuals
    actions = generate_random_actions(6)
    counterfactuals = generate_counterfactuals(actions)

    # Run a few simulation steps
    for step in range(4):
        print(f"\n--- Simulation step {step + 1} ---")
        simulate_step(actions, counterfactuals, morph, breaker)