# DARWIN HAMMER — match 3773, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_bayes__m2655_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_regret_engine_m2548_s0.py (gen5)
# born: 2026-05-29T23:52:55Z

"""Hybrid Algorithm: Pheromone‑Bayesian & Regret‑Circuit‑MinHash Fusion
====================================================================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – a pheromone system whose *expected entropy* modulates the
  likelihood of selecting evidence for a Bayesian update.
* **Parent B** – a regret‑weighted decision engine whose *Fisher score* tunes
  circuit‑breaker thresholds and whose *MinHash* similarity adjusts action
  values.

**Mathematical Bridge**

The bridge is built on three coupled quantities:

1. **Expected Entropy (E)** of the pheromone distribution on a surface.
2. **Fisher Information (F)** derived from the morphology of the entity,
   used to scale the circuit‑breaker failure threshold.
3. **MinHash similarity (M)** between the current action vector and a set of
   reference actions, which weights the regret‑adjusted action values.

The hybrid update equations are:


# Evidence selection weight
w_e = (1 - E) * M

# Bayesian posterior for hypothesis h
posterior_h = prior_h * likelihood_h(w_e) / Z

# Regret‑weighted value for action a
V_a = Σ_h posterior_h * reward_h(a) * exp(-R_a) * M_a

# Adaptive circuit‑breaker threshold
threshold = base_threshold * (1 + λ * F) * (1 + κ * E)


where `R_a` is cumulative regret for action `a`,
`λ` and `κ` are tunable scaling constants,
and `Z` normalises the posterior.

The implementation below provides three public functions that realise
these coupled updates, together with lightweight data‑structures needed for
the hybrid operation.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core components from Parent A
# ----------------------------------------------------------------------


class PheromoneSystem:
    """Tracks pheromone signals with exponential decay (simplified)."""

    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {}
        entry = self.pheromones[surface_key].get(signal_kind)
        if entry is None:
            entry = {"value": signal_value, "timestamp": now}
            self.pheromones[surface_key][signal_kind] = entry
        else:
            # exponential decay towards new value
            elapsed = (now - entry["timestamp"]).total_seconds()
            decay = 0.5 ** (elapsed / half_life_seconds)
            entry["value"] = decay * entry["value"] + (1 - decay) * signal_value
            entry["timestamp"] = now
        return entry["value"]

    def get_signal_distribution(self, surface_key: str) -> List[float]:
        """Return the list of current signal values for a surface."""
        return [
            info["value"]
            for info in self.pheromones.get(surface_key, {}).values()
        ]


def expected_entropy(probabilities: List[float]) -> float:
    """Shannon entropy of a normalised probability vector."""
    if not probabilities:
        return 0.0
    probs = np.array(probabilities, dtype=float)
    probs_sum = probs.sum()
    if probs_sum == 0:
        return 0.0
    probs = probs / probs_sum
    # avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Core components from Parent B
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""

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
    """Failure counter that opens when a scaled threshold is exceeded."""

    def __init__(self, base_threshold: int = 3):
        if base_threshold <= 0:
            raise ValueError("base_threshold must be positive")
        self.base_threshold = base_threshold
        self.failures = 0
        self.open = False

    def record_failure(self, scale: float = 1.0) -> None:
        """Increment failures; open if scaled threshold crossed."""
        threshold = int(self.base_threshold * scale)
        self.failures += 1
        if self.failures >= threshold:
            self.open = True

    def reset(self) -> None:
        self.failures = 0
        self.open = False


def fisher_information(morph: Morphology) -> float:
    """
    Simple proxy for Fisher information: variance of the normalized
    morphological parameters. Higher variance → more information.
    """
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 0.0
    normalized = vec / norm
    return float(np.var(normalized))


def minhash_signature(vector: List[float], num_perm: int = 64) -> np.ndarray:
    """
    Very lightweight MinHash: for each permutation we take the index of the
    minimal (hash(value), perm_index) pair. Returns a binary signature.
    """
    rng = np.random.default_rng(42)
    perms = rng.integers(0, 2 ** 31 - 1, size=(num_perm, len(vector)), dtype=np.int64)
    hashed = np.array([hash(v) for v in vector], dtype=np.int64)
    combined = (hashed[None, :] ^ perms)  # shape (num_perm, len(vector))
    mins = np.min(combined, axis=1)
    # Convert to bits: 1 if min is even, else 0
    return (mins % 2).astype(np.int8)


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Jaccard similarity between two binary MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signatures must have the same length")
    return float(np.sum(sig1 == sig2) / sig1.size)


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


class MathHypothesis:
    """Simple hypothesis with prior/posterior and linked evidence."""

    def __init__(self, id_: str, prior: float):
        self.id = id_
        self.prior = prior
        self.posterior = prior
        self.evidence_ids: List[str] = []


class MathEvidence:
    """Evidence object identified by a UUID."""

    def __init__(self, id_: str = None):
        self.id = id_ or str(uuid.uuid4())


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------


def hybrid_evidence_selection(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    half_life: float,
    evidence_pool: List[MathEvidence],
) -> Tuple[MathEvidence, float]:
    """
    Select a piece of evidence weighted by (1 - expected entropy) of the
    pheromone distribution on ``surface_key``.
    Returns the chosen evidence and the selection weight.
    """
    # Update pheromone signals (dummy values for illustration)
    for i, ev in enumerate(evidence_pool):
        pheromone_system.calculate_pheromone_signal(
            surface_key,
            f"signal_{i}",
            signal_value=random.random(),
            half_life_seconds=half_life,
        )
    probs = pheromone_system.get_signal_distribution(surface_key)
    entropy = expected_entropy(probs)
    weight = 1.0 - entropy  # higher weight when distribution is peaked
    # Choose evidence proportionally to the underlying signal magnitude
    if not probs:
        chosen = random.choice(evidence_pool)
        return chosen, weight
    probs = np.array(probs)
    probs = probs / probs.sum()
    idx = np.random.choice(len(evidence_pool), p=probs)
    return evidence_pool[idx], weight


def hybrid_regret_weighted_action(
    hypotheses: List[MathHypothesis],
    action_vector: List[float],
    reference_actions: List[List[float]],
    regret_history: Dict[int, float],
    lambda_regret: float = 0.5,
) -> float:
    """
    Compute a regret‑weighted value for an action.
    The value is the posterior‑averaged reward scaled by:
        exp(-regret) * MinHash similarity.
    """
    # 1. Build MinHash signature for the current action
    cur_sig = minhash_signature(action_vector)
    # 2. Compute maximum similarity to any reference action
    sims = [
        minhash_similarity(cur_sig, minhash_signature(ref))
        for ref in reference_actions
    ]
    max_sim = max(sims) if sims else 1.0

    # 3. Aggregate posterior probabilities
    total_posterior = sum(h.posterior for h in hypotheses) or 1.0
    weighted_reward = 0.0
    for h in hypotheses:
        # Dummy reward model: reward = prior (placeholder for real model)
        reward = h.prior
        weighted_reward += (h.posterior / total_posterior) * reward

    # 4. Apply regret weighting
    action_id = hash(tuple(action_vector))  # deterministic identifier
    regret = regret_history.get(action_id, 0.0)
    regret_factor = math.exp(-lambda_regret * regret)

    return weighted_reward * regret_factor * max_sim


def hybrid_circuit_breaker_update(
    breaker: EndpointCircuitBreaker,
    morph: Morphology,
    pheromone_system: PheromoneSystem,
    surface_key: str,
    base_lambda: float = 0.3,
    base_kappa: float = 0.2,
) -> None:
    """
    Adjust the circuit‑breaker failure threshold using:
        * Fisher information from morphology,
        * Expected entropy from pheromone signals.
    Then record a failure (for demo purposes) using the scaled threshold.
    """
    # Fisher information component
    F = fisher_information(morph)

    # Expected entropy component
    probs = pheromone_system.get_signal_distribution(surface_key)
    E = expected_entropy(probs)

    # Compute scaling factor
    scale = (1.0 + base_lambda * F) * (1.0 + base_kappa * E)

    # Record a failure using the computed scale
    breaker.record_failure(scale=scale)


# ----------------------------------------------------------------------
# Utility for Bayesian update (uses evidence weight from hybrid_evidence_selection)
# ----------------------------------------------------------------------


def bayesian_update(
    hypotheses: List[MathHypothesis],
    evidence_weight: float,
    likelihood_func,
) -> None:
    """
    Perform a Bayesian update on the list of hypotheses.
    ``likelihood_func`` receives a hypothesis and returns a likelihood value.
    The evidence weight (derived from pheromone entropy) modulates the likelihood.
    """
    total = 0.0
    for h in hypotheses:
        likelihood = likelihood_func(h)
        # Modulate by evidence weight
        h.posterior = h.prior * (likelihood ** evidence_weight)
        total += h.posterior
    if total == 0:
        return
    for h in hypotheses:
        h.posterior /= total


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Initialise components
    pheromone = PheromoneSystem()
    surface = "test_surface"
    half_life = 30.0

    # Create a pool of dummy evidence objects
    evidence_pool = [MathEvidence() for _ in range(5)]

    # Select evidence using hybrid logic
    chosen_evidence, sel_weight = hybrid_evidence_selection(
        pheromone, surface, half_life, evidence_pool
    )
    print(f"Chosen evidence ID: {chosen_evidence.id}, weight: {sel_weight:.3f}")

    # Initialise hypotheses
    hyps = [MathHypothesis(f"H{i}", prior=random.uniform(0.1, 0.9)) for i in range(3)]

    # Simple likelihood function (placeholder)
    def dummy_likelihood(hyp):
        return random.uniform(0.5, 1.0)

    # Perform Bayesian update
    bayesian_update(hyps, sel_weight, dummy_likelihood)
    for h in hyps:
        print(f"Hypothesis {h.id}: prior={h.prior:.3f}, posterior={h.posterior:.3f}")

    # Define an action and reference actions
    action = [random.random() for _ in range(8)]
    refs = [[random.random() for _ in range(8)] for _ in range(4)]
    regret_hist = {}

    # Compute regret‑weighted value
    value = hybrid_regret_weighted_action(hyps, action, refs, regret_hist)
    print(f"Regret‑weighted action value: {value:.4f}")

    # Morphology and circuit breaker
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)
    breaker = EndpointCircuitBreaker(base_threshold=3)

    # Update circuit breaker using hybrid scaling
    hybrid_circuit_breaker_update(breaker, morph, pheromone, surface)

    print(
        f"Circuit breaker state: failures={breaker.failures}, open={breaker.open}"
    )
    # Record another failure to see scaling effect
    hybrid_circuit_breaker_update(breaker, morph, pheromone, surface)
    print(
        f"After second update: failures={breaker.failures}, open={breaker.open}"
    )
    sys.exit(0)