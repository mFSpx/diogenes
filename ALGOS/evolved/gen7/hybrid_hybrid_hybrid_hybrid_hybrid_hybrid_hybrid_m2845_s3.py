# DARWIN HAMMER — match 2845, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1876_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_hybrid_cockpit_metri_m1060_s1.py (gen4)
# born: 2026-05-29T23:46:23Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Schoolfield model and pheromone handling
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield model for temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def calculate_pheromone_probabilities(surface_data: List[float], temp_k: float) -> List[float]:
    """Scale raw surface values by the developmental rate and normalise."""
    rate = developmental_rate(temp_k)
    pheromones = [r * rate for r in surface_data]
    total = sum(pheromones)
    if total == 0:
        # avoid division by zero – fall back to uniform distribution
        n = len(surface_data)
        return [1.0 / n] * n
    return [p / total for p in pheromones]


def shannon_entropy(probabilities: List[float]) -> float:
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


# ----------------------------------------------------------------------
# Parent B – Hoeffding bound, acceptance, cooling, and metrics
# ----------------------------------------------------------------------


def broadcast_probability(phase: int, step: int) -> float:
    """Probability that a pruning broadcast is active at a given phase/step."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability using a temperature term."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for the difference between two empirical means."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> bool:
    """Decision rule for splitting a node based on Hoeffding bound."""
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    return gap > eps or eps < tie_threshold


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims to total emitted claims."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                                                          claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Metric of honesty in a cockpit‑style reporting system."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total == 0 else displayed_ok / total


# ----------------------------------------------------------------------
# Hybrid operations – three core functions
# ----------------------------------------------------------------------


def hybrid_temperature(temp_c: float) -> float:
    """
    Compute an effective temperature for acceptance probability.
    The developmental rate from the Schoolfield model is used as a scaling
    factor for a base temperature (set to 1.0 Kelvin for convenience).
    """
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k)
    # Map rate (typically ~0‑1) to a temperature in (0, 2] Kelvin
    return max(0.01, min(2.0, rate))


def hybrid_split_decision(surface_data: List[float],
                          temp_c: float,
                          best_gain: float,
                          second_best_gain: float,
                          r: float,
                          delta: float,
                          n: int) -> Tuple[bool, float]:
    """
    Combine pheromone‑derived evidence with Hoeffding split logic.
    Returns a tuple (should_split, combined_score) where combined_score
    incorporates entropy as a regulariser.
    """
    temp_k = c_to_k(temp_c)
    pheromone_probs = calculate_pheromone_probabilities(surface_data, temp_k)
    entropy = shannon_entropy(pheromone_probs)

    # Hoeffding decision
    split = should_split(best_gain, second_best_gain, r, delta, n)

    # Entropy regularisation: higher entropy (more uncertainty) lowers
    # confidence in splitting.  We blend the binary split decision with a
    # continuous score in [0,1].
    confidence = (1.0 - entropy)  # 1 = low uncertainty, 0 = high uncertainty
    combined_score = confidence * (1.0 if split else 0.0)

    return split, combined_score


def hybrid_pruning_probability(surface_data: List[float],
                               temp_c: float,
                               claims_with_evidence: int,
                               total_claims_emitted: int,
                               displayed_ok: int,
                               unknown_displayed_as_ok: int,
                               phase: int,
                               step: int) -> float:
    """
    Compute a unified pruning probability that blends:
    * broadcast probability (phase/step schedule)
    * anti‑slop ratio
    * cockpit honesty
    * pheromone entropy (from Parent A)
    * temperature‑scaled acceptance probability (using a dummy ΔE)
    """
    # 1. Base broadcast schedule
    base = broadcast_probability(phase, step)

    # 2. Quality metrics
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)

    # 3. Entropy from pheromones
    temp_k = c_to_k(temp_c)
    probs = calculate_pheromone_probabilities(surface_data, temp_k)
    entropy = shannon_entropy(probs)

    # 4. Acceptance probability
    temp = hybrid_temperature(temp_c)
    delta_e = 0.0  # dummy energy difference
    acceptance = acceptance_probability(delta_e, temp)

    # Combine factors into a unified pruning probability
    pruning_prob = base * slop * honesty * (1.0 - entropy) * acceptance
    return max(0.0, min(1.0, pruning_prob))  # clamp to [0,1]


def improved_hybrid_pruning_probability(surface_data: List[float],
                                        temp_c: float,
                                        claims_with_evidence: int,
                                        total_claims_emitted: int,
                                        displayed_ok: int,
                                        unknown_displayed_as_ok: int,
                                        phase: int,
                                        step: int,
                                        alpha: float = 0.5) -> float:
    """
    Improved version of hybrid_pruning_probability with more flexible
    combination of factors using a weighted geometric mean.
    """
    # 1. Base broadcast schedule
    base = broadcast_probability(phase, step)

    # 2. Quality metrics
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)

    # 3. Entropy from pheromones
    temp_k = c_to_k(temp_c)
    probs = calculate_pheromone_probabilities(surface_data, temp_k)
    entropy = shannon_entropy(probs)

    # 4. Acceptance probability
    temp = hybrid_temperature(temp_c)
    delta_e = 0.0  # dummy energy difference
    acceptance = acceptance_probability(delta_e, temp)

    # Combine factors into a unified pruning probability using weighted geometric mean
    factors = [base, slop, honesty, 1.0 - entropy, acceptance]
    weights = [alpha] * 5
    weights = [w / sum(weights) for w in weights]  # normalise weights
    pruning_prob = math.prod(f ** w for f, w in zip(factors, weights))
    return max(0.0, min(1.0, pruning_prob))  # clamp to [0,1]