# DARWIN HAMMER — match 2013, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py (gen4)
# born: 2026-05-29T23:40:26Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A: `hybrid_hybrid_hybrid_hybrid_omni_chaotic_sprint_m621_s3.py`
    - Provides a Caputo fractional kernel for memory weighting,
      sinusoidally varying effective time constants, and stochastic daily
      resource allocation.

Parent B: `hybrid_hybrid_hybrid_bandit_hybrid_hybrid_rbf_su_m971_s0.py`
    - Supplies a temperature‑dependent Schoolfield developmental rate,
      a Gaussian radial‑basis‑function (RBF) surrogate, and a contextual
      multi‑armed bandit model.

Mathematical Bridge
-------------------
The bridge is the *fractional‑memory‑augmented developmental rate*.
For each day we compute the Schoolfield rate ρ(t) from the ambient
temperature.  A Caputo kernel with order `α∈(0,1)` weights the past
rates, yielding a memory‑augmented scalar `μ(t)`.  This scalar is used
as the shape parameter `ε` of the Gaussian RBF kernel, thereby letting
the surrogate inherit the long‑range temporal correlations of the
fractional calculus.  Resource allocation further mixes the daily
effective time constant with bandit propensities, so that actions with
higher confidence receive proportionally larger shares.

The resulting system couples fractional calculus, temperature‑driven
biology, kernel‑based surrogate modelling, and bandit‑driven decision
making into a single unified workflow.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def caputo_kernel(alpha: float, delta: int) -> float:
    """Caputo kernel for fractional order α and lag δ."""
    if delta < 0:
        raise ValueError("Delta must be non‑negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term


def fractional_memory_sum(alpha: float, values: List[float]) -> float:
    """Weighted sum of `values` using the Caputo kernel."""
    total = 0.0
    t = len(values) - 1
    for k, v in enumerate(values):
        delta = t - k
        total += caputo_kernel(alpha, delta) * v
    return total


def init_ltc_parameters(base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    """Initialize parameters for the sinusoidal effective time constant."""
    return {
        "base_tau": float(base_tau),
        "amplitude": float(amplitude),
        "gamma": 2 * math.pi / 7.0,  # weekly rhythm
    }


def effective_time_constant(day: int, params: dict) -> float:
    """Effective time constant τ(d) = base·[1 + amp·sin(γ·d)]."""
    base = params["base_tau"]
    amp = params["amplitude"]
    gamma = params["gamma"]
    return base * (1.0 + amp * math.sin(gamma * day))


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0  # default; will be overridden by memory‑augmented rate


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (low * high)


def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Hybrid Functions (the fused core)
# ----------------------------------------------------------------------
def compute_memory_augmented_rates(
    days: List[int],
    temps_celsius: List[float],
    alpha: float,
    ltc_params: dict,
) -> List[float]:
    """
    For each day:
        1. Compute the Schoolfield developmental rate ρ(t) from temperature.
        2. Apply the Caputo fractional memory operator to the sequence
           of rates up to the current day, producing μ(t).
        3. Scale μ(t) by the effective time constant τ(d) to embed the
           sinusoidal rhythm.
    Returns a list of memory‑augmented, time‑scaled rates.
    """
    if len(days) != len(temps_celsius):
        raise ValueError("days and temps_celsius must have the same length")
    raw_rates: List[float] = []
    mem_augmented: List[float] = []
    for d, temp_c in zip(days, temps_celsius):
        rho = developmental_rate(c_to_k(temp_c))
        raw_rates.append(rho)
        mu = fractional_memory_sum(alpha, raw_rates)
        tau = effective_time_constant(d, ltc_params)
        mem_augmented.append(mu * tau)
    return mem_augmented


def hybrid_rbf_predict(
    surrogate: RBFSurrogate,
    input_vec: Tuple[float, ...],
    epsilon: float,
) -> float:
    """
    Gaussian RBF prediction where the shape parameter ε is supplied by the
    memory‑augmented rate.  The surrogate's stored `epsilon` field is ignored
    in favor of the dynamic value.
    """
    if len(surrogate.centers) != len(surrogate.weights):
        raise ValueError("Number of centers must match number of weights")
    pred = 0.0
    for center, w in zip(surrogate.centers, surrogate.weights):
        r = euclidean(input_vec, center)
        pred += w * gaussian(r, epsilon)
    return pred


def hybrid_allocate_by_dates(
    days: List[int],
    groups: List[str],
    bandit_actions: Dict[str, BanditAction],
    ltc_params: dict,
    total_daily_budget: float = 100.0,
) -> Dict[int, Dict[str, float]]:
    """
    Daily allocation combines:
        • The sinusoidal effective time constant τ(d) (Parent A).
        • Random gate values (as in Parent A) to ensure stochasticity.
        • Bandit propensities to bias allocation toward high‑confidence actions.
    Returns a nested dict: day → group → allocated amount.
    """
    allocations: Dict[int, Dict[str, float]] = {}
    random.seed(0)  # reproducibility
    for d in days:
        tau = effective_time_constant(d, ltc_params)
        # Random gates for each group
        gates = {g: random.random() for g in groups}
        total_gate = sum(gates.values())
        # Propensity‑adjusted shares
        day_alloc: Dict[str, float] = {}
        for g in groups:
            prop = bandit_actions[g].propensity if g in bandit_actions else 1.0
            share = (gates[g] / total_gate) * prop
            day_alloc[g] = share * tau * total_daily_budget / sum(
                effective_time_constant(d, ltc_params) for _ in groups
            )
        allocations[d] = day_alloc
    return allocations


def update_bandit_actions(
    actions: Dict[str, BanditAction],
    rewards: Dict[str, float],
    alpha: float = 0.1,
) -> Dict[str, BanditAction]:
    """
    Simple Bayesian‑like update: increase propensity proportionally to the
    received reward, and shrink confidence bound.
    Returns a new dict with updated immutable BanditAction instances.
    """
    updated: Dict[str, BanditAction] = {}
    for aid, act in actions.items():
        r = rewards.get(aid, 0.0)
        new_propensity = max(0.0, act.propensity + alpha * (r - act.propensity))
        new_confidence = max(0.0, act.confidence_bound * (1.0 - alpha))
        updated[aid] = BanditAction(
            action_id=act.action_id,
            propensity=new_propensity,
            expected_reward=act.expected_reward + alpha * (r - act.expected_reward),
            confidence_bound=new_confidence,
            algorithm=act.algorithm,
        )
    return updated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a short temporal horizon
    days = list(range(10))
    groups = ["A", "B", "C"]

    # Synthetic temperature profile (°C)
    temps_c = [15 + 5 * math.sin(2 * math.pi * d / 7) for d in days]

    # Initialise parameters
    ltc_params = init_ltc_parameters(base_tau=1.2, amplitude=0.25)
    alpha_frac = 0.6  # fractional order for memory

    # Initialise bandit actions (one per group)
    bandit_actions = {
        g: BanditAction(
            action_id=g,
            propensity=1.0,
            expected_reward=0.0,
            confidence_bound=1.0,
            algorithm="HybridBandit",
        )
        for g in groups
    }

    # 1️⃣ Compute memory‑augmented, time‑scaled rates
    mem_rates = compute_memory_augmented_rates(days, temps_c, alpha_frac, ltc_params)
    print("Memory‑augmented rates per day:")
    for d, mr in zip(days, mem_rates):
        print(f" Day {d:2d}: {mr:.4f}")

    # 2️⃣ Build a dummy RBF surrogate
    rng = np.random.default_rng(42)
    centers = [tuple(rng.uniform(-1, 1, size=3)) for _ in range(5)]
    weights = list(rng.normal(0, 1, size=5))
    surrogate = RBFSurrogate(centers=centers, weights=weights)

    # Predict a reward for each day using the memory‑augmented rate as ε
    predictions = []
    for d, eps in zip(days, mem_rates):
        inp = tuple(rng.uniform(-1, 1, size=3))
        pred = hybrid_rbf_predict(surrogate, inp, epsilon=eps)
        predictions.append(pred)
    print("\nSurrogate predictions (one per day):")
    for d, p in zip(days, predictions):
        print(f" Day {d:2d}: {p:.4f}")

    # 3️⃣ Allocate resources using bandit propensities
    allocations = hybrid_allocate_by_dates(days, groups, bandit_actions, ltc_params)
    print("\nSample allocation for Day 0:")
    for g, amt in allocations[0].items():
        print(f" Group {g}: {amt:.2f}")

    # 4️⃣ Mock reward feedback (use predictions as proxy)
    mock_rewards = {g: predictions[0] * (1.0 + 0.1 * i) for i, g in enumerate(groups)}
    bandit_actions = update_bandit_actions(bandit_actions, mock_rewards)

    print("\nUpdated bandit propensities after mock reward:")
    for g, act in bandit_actions.items():
        print(f" Group {g}: propensity={act.propensity:.4f}, confidence={act.confidence_bound:.4f}")

    sys.exit(0)