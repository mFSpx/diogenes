# DARWIN HAMMER — match 4229, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_hybrid_hybrid_m1243_s3.py (gen5)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_ternar_m868_s1.py (gen4)
# born: 2026-05-29T23:54:21Z

"""Hybrid Pheromone‑VRAM Scheduler with Bayesian Utilities.

Parents:
- hybrid_hybrid_pheromone_inf_hybrid_hybrid_hybrid_m1243_s3.py (Pheromone decay, noisy input, sigmoid dynamics)
- hybrid_hybrid_model_vram_sc_hybrid_hybrid_ternar_m868_s1.py (VRAM linear scheduler, Bayesian marginal utilities)

Mathematical bridge:
The pheromone entry’s decay factor is interpreted as a *prior* probability 𝑃(H) for a memory‑allocation candidate.
A Bayesian marginal probability 𝑃(E) derived from the VRAM usage model acts as a *likelihood*.
The posterior 𝑃(H│E) = 𝑃(E)·𝑃(H) / [𝑃(E)·𝑃(H) + (1‑𝑃(E))·(1‑𝑃(H))] becomes the adaptive weight that drives the final VRAM allocation.
Noisy feature injection (σ‑diffusion schedule) from the pheromone side enriches the candidate feature vector before the linear VRAM schedule is evaluated.

The resulting hybrid system therefore evolves as:

    prior_i   = signal_i · decay_i
    likelihood_i = σ( a·usage + b )
    posterior_i  = bayes_update(prior_i, likelihood_i)
    alloc_i      = (budget – reserve) · posterior_i
"""

import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def linear_schedule(usage: float, a: float = 0.5, b: float = 0.1) -> float:
    """
    Linear VRAM allocation probability as a function of current usage (0‑1).
    Mirrors the simple linear model from the VRAM scheduler parent.
    """
    return max(0.0, min(1.0, a * usage + b))


def diffusion_alpha(t: int, T: int = 100) -> float:
    """
    Cumulative diffusion schedule ᾱ[t] ∈ [0,1].
    Simple quadratic ramp: ᾱ[t] = (t/T)^2 .
    """
    return (t / T) ** 2 if T else 0.0


def noisy_input(signal: float, t_idx: int, T: int = 100) -> float:
    """
    Implements x_noisy_i = √ᾱ[t_i]·I_i + √(1-ᾱ[t_i])·ε_i
    where ε_i ~ N(0,1).
    """
    alpha = diffusion_alpha(t_idx, T)
    noise = random.gauss(0.0, 1.0)
    return math.sqrt(alpha) * signal + math.sqrt(1.0 - alpha) * noise


# ----------------------------------------------------------------------
# Pheromone data structure (from Parent A)
# ----------------------------------------------------------------------
class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """
        Exponential decay based on half‑life:
        factor = 0.5 ** (age / half_life)
        """
        age = self.age_seconds()
        if self.half_life_seconds <= 0:
            return 1.0
        return 0.5 ** (age / self.half_life_seconds)

    def prior(self) -> float:
        """Prior probability derived from signal value and decay."""
        return max(0.0, min(1.0, self.signal_value * self.decay_factor()))


# ----------------------------------------------------------------------
# Bayesian update (from Parent B)
# ----------------------------------------------------------------------
def bayes_update(prior: float, likelihood: float) -> float:
    """
    Posterior for binary hypothesis:
        P(H|E) = P(E)·P(H) / [P(E)·P(H) + (1‑P(E))·(1‑P(H))]
    Both prior and likelihood are assumed to be in [0,1].
    """
    numerator = likelihood * prior
    denominator = numerator + (1.0 - likelihood) * (1.0 - prior)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid allocation core
# ----------------------------------------------------------------------
def compute_candidate_weights(
    pheromones: List[PheromoneEntry],
    usage_fraction: float,
    time_index: int,
    total_time_steps: int = 100,
) -> List[float]:
    """
    For each pheromone entry:
        1. Generate a noisy signal (diffusion schedule).
        2. Convert to a prior via decay().
        3. Obtain a likelihood from the VRAM linear schedule.
        4. Return the Bayesian posterior as the candidate weight.
    """
    weights: List[float] = []
    likelihood = linear_schedule(usage_fraction)  # same for all candidates in this simple model
    for entry in pheromones:
        noisy_sig = noisy_input(entry.signal_value, time_index, total_time_steps)
        # Clamp noisy signal to [0,1] before treating as probability
        noisy_sig = max(0.0, min(1.0, noisy_sig))
        # Update entry's signal_value temporarily with noisy version for prior calculation
        prior = max(0.0, min(1.0, noisy_sig * entry.decay_factor()))
        posterior = bayes_update(prior, likelihood)
        weights.append(posterior)
    return weights


def allocate_vram(
    total_budget_mb: int,
    reserve_mb: int,
    candidate_weights: List[float],
) -> List[int]:
    """
    Distribute (budget‑reserve) proportionally to the posterior weights.
    Guarantees integer MB allocations and that the sum does not exceed budget‑reserve.
    """
    allocatable = max(0, total_budget_mb - reserve_mb)
    if not candidate_weights:
        return []

    weight_sum = sum(candidate_weights)
    if weight_sum == 0.0:
        # Even split if all posteriors are zero
        equal_share = allocatable // len(candidate_weights)
        return [equal_share] * len(candidate_weights)

    raw_alloc = [w / weight_sum * allocatable for w in candidate_weights]
    int_alloc = [int(math.floor(a)) for a in raw_alloc]
    remainder = allocatable - sum(int_alloc)

    # Distribute the remaining MBs to the candidates with highest fractional parts
    fractions = [(i, raw_alloc[i] - int_alloc[i]) for i in range(len(candidate_weights))]
    fractions.sort(key=lambda x: x[1], reverse=True)
    for i, _ in fractions[:remainder]:
        int_alloc[i] += 1

    return int_alloc


def hybrid_step(
    pheromones: List[PheromoneEntry],
    current_usage_mb: int,
    total_budget_mb: int,
    reserve_mb: int,
    time_index: int,
    total_time_steps: int = 100,
) -> Tuple[List[int], List[float]]:
    """
    Executes a single hybrid scheduling step:
        * Computes posterior weights via pheromone diffusion and Bayesian update.
        * Allocates VRAM according to those weights.
    Returns both the integer allocations (MB) and the underlying posterior weights.
    """
    usage_frac = current_usage_mb / total_budget_mb if total_budget_mb else 0.0
    weights = compute_candidate_weights(
        pheromones, usage_frac, time_index, total_time_steps
    )
    allocations = allocate_vram(total_budget_mb, reserve_mb, weights)
    return allocations, weights


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration
    TOTAL_BUDGET_MB = 4096
    RESERVE_MB = 768
    CURRENT_USAGE_MB = 1800
    TIME_STEPS = 100

    # Create a few synthetic pheromone entries
    random.seed(42)
    pheromones = [
        PheromoneEntry(
            surface_key=f"layer_{i}",
            signal_kind="memory_pressure",
            signal_value=random.uniform(0.2, 0.9),
            half_life_seconds=random.randint(30, 120),
        )
        for i in range(5)
    ]

    # Simulate a single hybrid step at time index 42
    allocations, posteriors = hybrid_step(
        pheromones,
        current_usage_mb=CURRENT_USAGE_MB,
        total_budget_mb=TOTAL_BUDGET_MB,
        reserve_mb=RESERVE_MB,
        time_index=42,
        total_time_steps=TIME_STEPS,
    )

    print("Posterior weights:", ["{:.4f}".format(p) for p in posteriors])
    print("Allocated MB per candidate:", allocations)
    print("Total allocated:", sum(allocations), "MB (should be ≤", TOTAL_BUDGET_MB - RESERVE_MB, "MB)")