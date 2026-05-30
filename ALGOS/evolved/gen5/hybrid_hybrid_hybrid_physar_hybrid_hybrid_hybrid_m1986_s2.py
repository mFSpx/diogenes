# DARWIN HAMMER — match 1986, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_regret_regret_engine_m822_s5.py (gen4)
# born: 2026-05-29T23:40:15Z

"""Hybrid Physarum‑Gliner Regret Engine (HPGRE)

This module fuses the two parent algorithms:

* **hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s0.py** – provides a
  flux‑based conductance dynamics and the `Span` data structure whose
  confidence `score` modulates the update.

* **hybrid_hybrid_hybrid_regret_regret_engine_m822_s5.py** – offers a
  regret‑weighted action selection mechanism that uses a MinHash‑based
  similarity metric between the current context and a set of reference
  contexts.

**Mathematical bridge** – The bridge is the *similarity* between contexts.
The MinHash similarity computed in the regret engine is used as an
additional multiplicative factor on the flux‐derived quantity `q` before
the conductance update.  Consequently the conductance of an edge grows not
only with the physical flux but also with how well the current textual
context matches previously observed contexts, linking the physarum
network to the regret‑weighted bandit router.  The same similarity weight
also scales the inflow term of the store dynamics, creating a unified
feedback loop across all three subsystems."""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# ----------------------------------------------------------------------
# Core data structures from Parent A
# ----------------------------------------------------------------------
class Span:
    """Represents a labeled text span with an associated confidence score."""
    def __init__(self, start: int, end: int, text: str, label: str, score: float):
        self.start = start
        self.end = end
        self.text = text
        self.label = label
        self.score = score

def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physical flux through an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Euler update of conductance with growth proportional to |q|."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ----------------------------------------------------------------------
# Core data structures from Parent B
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
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    """Honeybee‑style store that converts inflow/outflow into a bounded signal."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and recompute the dance duration."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        # Store the last delta for the lazy property
        setattr(self, "_last_delta", delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

# ----------------------------------------------------------------------
# Fusion utilities
# ----------------------------------------------------------------------
def minhash_signature(tokens: List[str], num_perm: int = 64) -> np.ndarray:
    """
    Compute a simple MinHash signature for a list of tokens.
    The signature is the minimum hash value observed for each of `num_perm`
    independent hash functions derived from SHA‑256 seeds.
    """
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode('utf‑8')
        base_hash = hashlib.sha256(token_bytes).digest()
        for i in range(num_perm):
            # Derive a per‑perm hash by mixing the base hash with the perm index
            mixed = hashlib.sha256(base_hash + i.to_bytes(1, 'little')).digest()
            hv = int.from_bytes(mixed[:8], 'little')
            if hv < sig[i]:
                sig[i] = hv
    return sig

def minhash_similarity(text_a: str, text_b: str, num_perm: int = 64) -> float:
    """
    Approximate Jaccard similarity using MinHash signatures.
    Returns a value in [0, 1].
    """
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if not tokens_a or not tokens_b:
        return 0.0
    sig_a = minhash_signature(tokens_a, num_perm)
    sig_b = minhash_signature(tokens_b, num_perm)
    return float(np.mean(sig_a == sig_b))

def hybrid_conductance_update(conductance: float,
                              edge_length: float,
                              pressure_a: float,
                              pressure_b: float,
                              span: Span,
                              similarity: float,
                              dt: float = 1.0,
                              gain: float = 1.0,
                              decay: float = 0.05) -> float:
    """
    Unified update:
    1. Compute physical flux `q`.
    2. Modulate `q` by both the Span confidence and the MinHash similarity.
    3. Apply the standard conductance growth/decay rule.
    """
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    weighted_q = q * span.score * similarity
    return update_conductance(conductance, weighted_q, dt, gain, decay)

def select_action_with_regret(context: str,
                              reference_contexts: Dict[str, str],
                              actions: List[MathAction],
                              temperature: float = 0.5) -> BanditAction:
    """
    Choose an action using regret‑weighted propensity.
    Similarity between `context` and each reference context scales the
    propensity (higher similarity → higher chance).
    The final propensity is normalized to a probability distribution.
    """
    # Compute raw scores
    raw = []
    for act in actions:
        # Use a dummy reference keyed by action id if available
        ref = reference_contexts.get(act.id, "")
        sim = minhash_similarity(context, ref)
        # Regret term: lower expected value → higher regret weight
        regret_weight = 1.0 / (1.0 + act.expected_value)
        score = sim * regret_weight
        raw.append(score)

    total = sum(raw)
    if total == 0:
        probs = [1.0 / len(actions)] * len(actions)
    else:
        probs = [r / total for r in raw]

    # Softmax temperature scaling
    scaled = np.exp(np.log(probs) / max(temperature, 1e-12))
    scaled /= scaled.sum()

    chosen_idx = np.random.choice(len(actions), p=scaled)
    chosen = actions[chosen_idx]
    propensity = scaled[chosen_idx]

    # Confidence bound (simple optimistic estimate)
    cb = math.sqrt(-math.log(propensity + 1e-12) / (2 * (chosen_idx + 1)))

    return BanditAction(
        action_id=chosen.id,
        propensity=propensity,
        expected_reward=chosen.expected_value,
        confidence_bound=cb,
        algorithm="HPGRE"
    )

def hybrid_store_update(store: StoreState,
                        bandit_update: BanditUpdate,
                        similarity: float) -> Tuple[float, float]:
    """
    Update the StoreState where the inflow term is the reward weighted by
    the similarity between the current context and its reference.
    The outflow term is the propensity (cost of taking the action).
    """
    inflow = [bandit_update.reward * similarity]
    outflow = [bandit_update.propensity]
    return store.update(inflow, outflow)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Create a random Span
    text = "The quick brown fox jumps over the lazy dog."
    labels = ["ANIMAL", "ACTION", "OBJECT"]
    span = Span(start=4, end=9, text=text, label=random.choice(labels), score=random.random())

    # 2. Initialise conductance parameters
    conductance = 0.5
    edge_len = 2.0
    p_a, p_b = 1.0, 0.3

    # 3. Compute similarity between two contexts
    ctx = "fox jumps"
    ref_ctx = "quick brown fox"
    sim = minhash_similarity(ctx, ref_ctx)

    # 4. Hybrid conductance update
    new_cond = hybrid_conductance_update(conductance, edge_len, p_a, p_b, span, sim)
    print(f"Updated conductance: {new_cond:.4f}")

    # 5. Define actions and reference contexts for bandit selection
    actions = [
        MathAction(id="a1", expected_value=0.2),
        MathAction(id="a2", expected_value=0.5),
        MathAction(id="a3", expected_value=0.1)
    ]
    reference_contexts = {
        "a1": "quick fox",
        "a2": "lazy dog",
        "a3": "brown jumps"
    }

    # 6. Select an action
    bandit_act = select_action_with_regret(ctx, reference_contexts, actions)
    print(f"Selected action: {bandit_act.action_id}, propensity={bandit_act.propensity:.3f}")

    # 7. Create a store and apply hybrid update
    store = StoreState()
    bandit_upd = BanditUpdate(
        context_id="c1",
        action_id=bandit_act.action_id,
        reward=bandit_act.expected_reward,
        propensity=bandit_act.propensity
    )
    level, delta = hybrid_store_update(store, bandit_upd, sim)
    print(f"Store level: {level:.3f}, delta: {delta:.3f}, dance: {store.dance:.3f}")

    # 8. Verify that no exception was raised
    sys.exit(0)