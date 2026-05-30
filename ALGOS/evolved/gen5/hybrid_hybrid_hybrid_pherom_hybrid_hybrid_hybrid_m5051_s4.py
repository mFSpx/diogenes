# DARWIN HAMMER — match 5051, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s0.py (gen4)
# born: 2026-05-29T23:59:29Z

"""
Hybrid Pheromone‑Regret Engine with Entropic‑Annealed Leader Election

Parents:
- hybrid_hybrid_pheromone_inf_privacy_m54_s0.py (pheromone system & entropy)
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_regret_m25_s0.py (Hoeffding bound,
  tropical max‑plus, regret‑engine leader election)

Mathematical bridge:
Both parents manipulate *uncertainty* to decide whether a structural change
should be kept.  The pheromone module quantifies uncertainty with Shannon
entropy of a probability vector; the regret engine treats the Hoeffding bound
ε as a temperature‑like quantity and uses a simulated‑annealing acceptance rule
based on the tropical gain G.  In the hybrid we:
1. Use the pheromone probabilities as the action‑distribution p_i.
2. Compute the entropy H(p) (A).
3. Compute the Hoeffding bound ε from observed gains (B).
4. Compute the tropical max‑plus gain G (B).
5. Form ΔE = ε – G and define an annealed acceptance probability
   α = exp( -ΔE / (H(p)+δ) ), δ≪1 to avoid division by zero.
   This unifies entropy (A) with the temperature‑like Hoeffding bound (B).
The hybrid engine therefore decides to keep a split (or promote a leader) by
sampling a Bernoulli trial with probability α.
"""

import math
import random
import sys
import pathlib
from datetime import datetime, timezone
from typing import Any, List, Mapping, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def calculate_entropy(probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    norm = [p / total for p in probabilities if p > 0]
    return -sum(p * math.log(max(p, eps)) for p in norm)


def decay_pheromone(
    previous_value: float,
    previous_half_life: float,
    elapsed_seconds: float,
) -> float:
    """Exponential decay of a pheromone signal."""
    return previous_value * math.pow(0.5, elapsed_seconds / previous_half_life)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def compute_hoeffding_bound(
    n_samples: int, confidence: float = 0.95
) -> float:
    """Hoeffding bound ε = sqrt( 2 * ln(1/δ) / n ).  δ = 1‑confidence."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    delta = 1.0 - confidence
    if delta <= 0:
        delta = 1e-12
    return math.sqrt(2.0 * math.log(1.0 / delta) / n_samples)


def tropical_max_plus_evaluate(coeffs: Sequence[float], gain: float) -> float:
    """Tropical max‑plus polynomial: max_i (coeff_i + gain)."""
    if not coeffs:
        raise ValueError("coefficients list cannot be empty")
    return max(c + gain for c in coeffs)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_acceptance_probability(
    action_probs: Sequence[float],
    n_observations: int,
    tropical_coeffs: Sequence[float],
    observed_gain: float,
    confidence: float = 0.95,
    epsilon: float = 1e-9,
) -> float:
    """
    Compute the acceptance probability α that unifies entropy (A) with the
    Hoeffding‑tropical acceptance rule (B).

    Steps:
    1. H = entropy(action_probs)
    2. ε = Hoeffding bound from n_observations
    3. G = tropical max‑plus evaluation
    4. ΔE = ε – G
    5. α = exp( -ΔE / (H + ε) ), clipped to [0, 1]
    """
    H = calculate_entropy(action_probs)
    hoeffding_eps = compute_hoeffding_bound(n_observations, confidence)
    G = tropical_max_plus_evaluate(tropical_coeffs, observed_gain)
    delta_E = hoeffding_eps - G
    # The denominator uses entropy as a temperature surrogate.
    denom = H + epsilon
    raw_alpha = math.exp(-delta_E / denom)
    # Clamp to valid probability range.
    return max(0.0, min(1.0, raw_alpha))


def update_pheromone_system(
    pheromone_store: dict,
    surface_key: Any,
    signal_kind: str,
    signal_value: float,
    half_life_seconds: float,
) -> float:
    """
    Update or create a pheromone entry.  Returns the (potentially decayed) signal
    that will be used as the new action probability for the hybrid engine.
    """
    now = datetime.now(timezone.utc)
    if surface_key not in pheromone_store:
        pheromone_store[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": half_life_seconds,
            "created_time": now,
        }
        return signal_value

    entry = pheromone_store[surface_key]
    elapsed = (now - entry["created_time"]).total_seconds()
    decayed = decay_pheromone(
        entry["signal_value"], entry["half_life_seconds"], elapsed
    )
    # Blend old decayed value with the new observation (simple averaging)
    blended = 0.5 * decayed + 0.5 * signal_value
    pheromone_store[surface_key] = {
        "signal_kind": signal_kind,
        "signal_value": blended,
        "half_life_seconds": half_life_seconds,
        "created_time": now,
    }
    return blended


def hybrid_leader_election(
    pheromone_store: dict,
    surface_key: Any,
    action_probs: Sequence[float],
    observed_gains: List[float],
    tropical_coeffs: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[bool, float]:
    """
    Perform a leader‑election decision for a given surface (e.g., a split node).

    Returns a tuple (keep_split, acceptance_probability).
    """
    # 1. Update pheromone to obtain a fresh probability estimate.
    latest_signal = update_pheromone_system(
        pheromone_store,
        surface_key,
        signal_kind="leader_score",
        signal_value=sum(action_probs),  # crude proxy; any scalar works
        half_life_seconds=60.0,
    )
    # 2. Normalise action probabilities (they may not sum to 1).
    norm_probs = np.array(action_probs, dtype=float)
    if norm_probs.sum() == 0:
        norm_probs = np.ones_like(norm_probs) / len(norm_probs)
    else:
        norm_probs = norm_probs / norm_probs.sum()
    # 3. Compute acceptance probability via the hybrid rule.
    alpha = hybrid_acceptance_probability(
        norm_probs,
        n_observations=len(observed_gains),
        tropical_coeffs=tropical_coeffs,
        observed_gain=sum(observed_gains) / max(1, len(observed_gains)),
        confidence=confidence,
    )
    # 4. Sample decision.
    keep = random.random() < alpha
    return keep, alpha


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    # Initialise a minimal pheromone store.
    pheromones: dict = {}

    # Example surface key.
    key = "node_42"

    # Dummy action probabilities (e.g., from a regret engine).
    actions = [0.2, 0.5, 0.3]

    # Simulated observed gains from a split evaluation.
    gains = [0.12, 0.15, 0.09, 0.11]

    # Tropical coefficients (could be learned offsets).
    tropical_coeffs = [0.05, 0.10, 0.02]

    # Run the hybrid leader election.
    keep, prob = hybrid_leader_election(
        pheromone_store=pheromones,
        surface_key=key,
        action_probs=actions,
        observed_gains=gains,
        tropical_coeffs=tropical_coeffs,
        confidence=0.95,
    )

    print(f"Decision to keep split for '{key}': {keep} (acceptance prob = {prob:.4f})")
    print(f"Pheromone state after update: {pheromones[key]}")


if __name__ == "__main__":
    _smoke_test()