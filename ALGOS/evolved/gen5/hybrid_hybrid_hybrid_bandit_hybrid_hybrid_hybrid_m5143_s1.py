# DARWIN HAMMER — match 5143, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_hybrid_m896_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s4.py (gen4)
# born: 2026-05-30T00:00:14Z

"""
Hybrid Algorithm combining:
- Parent A: Schoolfield temperature‑dependent developmental rate used to
  modulate pruning probability in a Bayesian update (Fisher‑Krampus style).
- Parent B: Sinusoidal weekday weight vector that supplies stochastic
  weights for a pheromone‑based entropy calculation.

Mathematical bridge:
The developmental rate ρ(T) (kelvin) is employed as a scaling factor for the
pheromone half‑life, thereby linking temperature to pheromone decay.
Conversely, the weekday‑derived weight vector w(d) is used to weight the
priors of hypotheses before the Bayesian update.  The hybrid system thus
propagates temperature information through both the Bayesian pruning
mechanism and the pheromone entropy measure, yielding a unified
exploration‑exploitation controller.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants (shared)
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal mol⁻¹ K⁻¹
K25 = 298.15   # reference temperature (K)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (K25)
    delta_h_activation: float = 12_000.0  # J mol⁻¹
    t_low: float = 283.15             # K
    t_high: float = 307.15            # K
    delta_h_low: float = -45_000.0    # J mol⁻¹
    delta_h_high: float = 65_000.0    # J mol⁻¹
    r_cal: float = R_CAL

@dataclass
class Hypothesis:
    id: str
    prior: float
    posterior: float = 0.0
    evidence_ids: List[str] = None

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.

    ρ(T) = ρ25 * (T / T25) *
           exp[ -ΔH_A / (R·T) + ΔH_A / (R·T25) ] /
           ( 1 + exp[ ΔH_L / (R·T_low) - ΔH_L / (R·T) ] +
                 exp[ ΔH_H / (R·T) - ΔH_H / (R·T_high) ] )
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")

    # activation term
    act = math.exp(
        -params.delta_h_activation / (params.r_cal * temp_k) +
         params.delta_h_activation / (params.r_cal * K25)
    )

    # low‑temperature deactivation term
    low = math.exp(
        params.delta_h_low / (params.r_cal * params.t_low) -
        params.delta_h_low / (params.r_cal * temp_k)
    )

    # high‑temperature deactivation term
    high = math.exp(
        params.delta_h_high / (params.r_cal * temp_k) -
        params.delta_h_high / (params.r_cal * params.t_high)
    )

    denom = 1.0 + low + high
    rate = params.rho_25 * (temp_k / K25) * act / denom
    return rate

def weekday_weight_vector(groups: List[str], dow: int,
                          amplitude: float = 0.3) -> np.ndarray:
    """
    Produce a row‑stochastic weight vector using a sinusoidal rotation.

    weight_i = 1 + amplitude * sin( 2π * i / N + phase )
    where phase = 2π * dow / 7.
    The vector is finally normalised to sum to 1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups list cannot be empty")

    base_angles = 2.0 * math.pi * np.arange(n) / n
    phase = 2.0 * math.pi * dow / 7.0
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec

def pheromone_signal(initial: float, half_life_sec: float,
                    elapsed_sec: float) -> float:
    """
    Exponential decay of a pheromone signal.

    signal(t) = initial * 2^{ -t / half_life }
    """
    if half_life_sec <= 0:
        raise ValueError("half_life_sec must be positive")
    decay_factor = 2 ** (-elapsed_sec / half_life_sec)
    return initial * decay_factor

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_pruning_probability(temp_k: float,
                                min_prob: float = 0.01,
                                max_prob: float = 0.5) -> float:
    """
    Map developmental rate to a pruning probability in [min_prob, max_prob].
    Higher temperature → higher developmental rate → higher pruning chance.
    """
    rate = developmental_rate(temp_k)
    # Normalise rate to [0,1] using a simple logistic stretch
    norm = 1.0 / (1.0 + math.exp(-0.01 * (rate - 0.5)))
    prob = min_prob + (max_prob - min_prob) * norm
    return prob

def bayesian_update_weighted(hypotheses: List[Hypothesis],
                             likelihoods: Dict[str, float],
                             temp_k: float,
                             groups: List[str],
                             dow: int) -> List[Hypothesis]:
    """
    Perform a Bayesian update where:
    - Priors are multiplied by weekday‑derived weights.
    - Likelihoods are attenuated by a pruning probability derived from
      the developmental rate.
    - Posterior is renormalised to sum to 1.
    """
    # 1. Compute weekday weights for the hypothesis identifiers
    weight_vec = weekday_weight_vector(groups, dow)
    weight_map = {hyp.id: w for hyp, w in zip(hypotheses, weight_vec)}

    # 2. Pruning probability from temperature
    prune_prob = compute_pruning_probability(temp_k)

    # 3. Apply weighting and pruning
    unnorm = {}
    for hyp in hypotheses:
        prior = hyp.prior * weight_map.get(hyp.id, 1.0)
        likelihood = likelihoods.get(hyp.id, 0.0)
        # prune (i.e., down‑weight) unlikely evidence
        adjusted_likelihood = likelihood * (1.0 - prune_prob)
        unnorm[hyp.id] = prior * adjusted_likelihood

    total = sum(unnorm.values())
    if total == 0:
        # fallback to uniform posterior
        uniform = 1.0 / len(hypotheses)
        for hyp in hypotheses:
            hyp.posterior = uniform
        return hypotheses

    for hyp in hypotheses:
        hyp.posterior = unnorm[hyp.id] / total
    return hypotheses

def pheromone_entropy(groups: List[str],
                      dow: int,
                      temp_c: float,
                      elapsed_sec: float = 10.0,
                      base_signal: float = 1.0) -> float:
    """
    Compute the Shannon entropy of a pheromone system whose half‑life
    is scaled by the developmental rate at the given temperature.

    Steps:
    1. Generate weekday‑based weights w_i.
    2. Scale the pheromone half‑life τ = τ₀ / ρ(T) where τ₀ is a reference
       half‑life (here 30 s) and ρ(T) is the developmental rate.
    3. Decay each group's pheromone signal using the scaled half‑life.
    4. Normalise the decayed signals to a probability distribution.
    5. Return H = -∑ p_i log₂ p_i.
    """
    # 1. Weekday weights (row‑stochastic)
    w = weekday_weight_vector(groups, dow)

    # 2. Temperature‑scaled half‑life
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k)
    reference_half_life = 30.0  # seconds
    half_life_scaled = reference_half_life / max(rate, 1e-6)  # avoid div‑zero

    # 3. Decay each pheromone component
    decayed = np.array([
        pheromone_signal(base_signal * wi, half_life_scaled, elapsed_sec)
        for wi in w
    ])

    # 4. Normalise to probabilities
    prob = decayed / decayed.sum()

    # 5. Shannon entropy (bits)
    entropy = -np.sum(prob * np.log2(prob + 1e-12))
    return float(entropy)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small scenario
    groups = ["codex", "groq", "cohere", "local_models"]
    today = datetime.now(timezone.utc)
    dow = (today.weekday() + 1) % 7  # 0 = Sunday … 6 = Saturday as in parent B

    # Temperature for the test (°C)
    temp_c = 22.0

    # Create hypotheses
    hyps = [
        Hypothesis(id="codex", prior=0.25),
        Hypothesis(id="groq", prior=0.25),
        Hypothesis(id="cohere", prior=0.25),
        Hypothesis(id="local_models", prior=0.25),
    ]

    # Simulated likelihoods (could come from any model)
    lik = {
        "codex": 0.6,
        "groq": 0.2,
        "cohere": 0.1,
        "local_models": 0.1,
    }

    # Perform hybrid Bayesian update
    updated = bayesian_update_weighted(
        hypotheses=hyps,
        likelihoods=lik,
        temp_k=c_to_k(temp_c),
        groups=groups,
        dow=dow,
    )

    print("Posterior after hybrid Bayesian update:")
    for h in updated:
        print(f"  {h.id}: {h.posterior:.4f}")

    # Compute pheromone entropy
    ent = pheromone_entropy(
        groups=groups,
        dow=dow,
        temp_c=temp_c,
        elapsed_sec=15.0,
        base_signal=1.0,
    )
    print(f"\nPheromone system entropy (bits): {ent:.4f}")

    # Demonstrate developmental rate alone
    dr = developmental_rate(c_to_k(temp_c))
    print(f"\nDevelopmental rate at {temp_c} °C: {dr:.6f}")

    sys.exit(0)