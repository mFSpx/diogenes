# DARWIN HAMMER — match 169, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s3.py (gen1)
# born: 2026-05-29T23:27:19Z

"""hybrid_ssim_bandit_temperature.py
Fusion of:
- hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0 (SSIM similarity & workshare allocation)
- hybrid_bandit_router_poikilotherm_schoolf_m20_s3 (Multi‑armed bandit policy & Schoolfield temperature model)

Mathematical bridge:
The SSIM similarity score (dimensionless, 0‑1) quantifies how well a data vector matches a prototype.
The Schoolfield model provides a temperature‑dependent scaling factor ρ(T) (also dimensionless).
We fuse them by defining a **combined suitability score**:

    σ_i = SSIM_i × ρ(T)

where SSIM_i is the similarity of group *i* (or action *i*) and ρ(T) is the developmental rate at the current temperature.
σ_i drives both:
1. Allocation of work units among groups (deterministic share is scaled by ρ(T) and the stochastic LLM share is weighted by σ_i).
2. Propensity/expected‑reward estimation for bandit actions (propensity ∝ σ_i, expected_reward updated with observed rewards).

The three public functions below illustrate this hybrid operation."""
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – SSIM & workshare (adapted)
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round to six decimal places for readability."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Structural Similarity Index (SSIM) between two 1‑D vectors.
    Returns a value in [-1, 1]; typical use‑case expects [0, 1].
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Parent B – Schoolfield temperature model (adapted)
# ----------------------------------------------------------------------


R_CAL = 1.987  # cal mol⁻¹ K⁻¹
K25 = 298.15  # reference temperature (25 °C) in Kelvin


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0  # cal mol⁻¹
    t_low: float = 283.15  # K
    t_high: float = 307.15  # K
    delta_h_low: float = -45_000.0  # cal mol⁻¹
    delta_h_high: float = 65_000.0  # cal mol⁻¹
    r_cal: float = R_CAL


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(
    temp_k: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Full Schoolfield rate ρ(T) as a function of absolute temperature.
    Implements the classic enzyme‑temperature response curve.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be positive Kelvin.")
    if params.rho_25 < 0:
        raise ValueError("rho_25 must be non‑negative.")

    # Arrhenius term for activation energy
    act_exp = math.exp(
        -params.delta_h_activation / params.r_cal * (1.0 / temp_k - 1.0 / K25)
    )

    # Low‑temperature inactivation term
    low_exp = math.exp(
        params.delta_h_low / params.r_cal * (1.0 / params.t_low - 1.0 / temp_k)
    )

    # High‑temperature inactivation term
    high_exp = math.exp(
        params.delta_h_high / params.r_cal * (1.0 / params.t_high - 1.0 / temp_k)
    )

    denominator = 1.0 + low_exp + high_exp
    rate = params.rho_25 * (temp_k / K25) * act_exp / denominator
    return max(rate, 0.0)  # rate cannot be negative


# ----------------------------------------------------------------------
# Parent B – Bandit router (adapted)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float  # probability of selection (0‑1)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid_ssim_bandit_temperature"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# Global policy storage: action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()


def _empirical_reward(action_id: str) -> float:
    """Mean observed reward for an action (0 if never tried)."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Hybrid core – mathematical fusion
# ----------------------------------------------------------------------


def combined_suitability(
    ssim: float, temperature_c: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    σ = SSIM × ρ(T)

    Parameters
    ----------
    ssim : float
        Structural similarity (typically in [0, 1]).
    temperature_c : float
        Ambient temperature in Celsius.
    params : SchoolfieldParams
        Parameters of the Schoolfield model.

    Returns
    -------
    float
        Combined suitability score (dimensionless, ≥0).
    """
    temp_k = c_to_k(temperature_c)
    rate = developmental_rate(temp_k, params)
    return max(ssim, 0.0) * rate


def allocate_hybrid_workshare(
    *,
    total_units: float,
    temperature_c: float,
    prototype: np.ndarray,
    group_vectors: Dict[str, np.ndarray],
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
) -> Dict[str, float]:
    """
    Allocate work units among groups using a temperature‑scaled deterministic share
    and an SSIM‑weighted stochastic share.

    The deterministic portion is multiplied by ρ(T) to reflect temperature‑driven capacity.
    The stochastic (LLM) portion is distributed proportionally to σ_i = SSIM_i × ρ(T).

    Returns a dictionary mapping each group to its allocated units (rounded to 6 dp).
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive.")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100.")
    if not groups:
        raise ValueError("At least one group required.")

    # 1️⃣ Deterministic share (temperature‑scaled)
    base_det = total_units * deterministic_target_pct / 100.0
    rho_T = developmental_rate(c_to_k(temperature_c))
    deterministic_units = base_det * rho_T

    # 2️⃣ Stochastic LLM share
    llm_units = total_units - deterministic_units
    # Compute per‑group SSIM against the prototype
    ssim_scores = {
        g: compute_ssim(v, prototype) for g, v in group_vectors.items() if g in groups
    }

    # Compute σ_i = SSIM_i × ρ(T)
    sigma = {g: combined_suitability(ssim, temperature_c) for g, ssim in ssim_scores.items()}

    total_sigma = sum(sigma.values()) or 1.0  # avoid division by zero
    per_group_alloc = {
        g: deterministic_units / len(groups) + llm_units * sigma[g] / total_sigma
        for g in groups
    }

    # Round for clean output
    return {g: _pct(v) for g, v in per_group_alloc.items()}


def bandit_select_action(
    *,
    context_id: str,
    actions: List[str],
    temperature_c: float,
    prototype: np.ndarray,
    action_vectors: Dict[str, np.ndarray],
    epsilon: float = 0.1,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> BanditAction:
    """
    Choose an action using an epsilon‑greedy scheme where the exploitation score
    is the combined suitability σ_i = SSIM_i × ρ(T).

    Parameters
    ----------
    context_id : str
        Identifier for the current decision context (used for logging only).
    actions : List[str]
        Candidate action identifiers.
    temperature_c : float
        Ambient temperature in Celsius.
    prototype : np.ndarray
        Reference vector against which SSIM is computed.
    action_vectors : Dict[str, np.ndarray]
        Mapping from action_id to its feature vector.
    epsilon : float
        Exploration probability (0‑1).
    params : SchoolfieldParams
        Temperature model parameters.

    Returns
    -------
    BanditAction
        Chosen action enriched with propensity and expected reward.
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1.")
    if not actions:
        raise ValueError("At least one action required.")

    # Exploration branch
    if random.random() < epsilon:
        chosen_id = random.choice(actions)
        propensity = epsilon / len(actions)
        exp_reward = _empirical_reward(chosen_id)
        return BanditAction(
            action_id=chosen_id,
            propensity=_pct(propensity),
            expected_reward=_pct(exp_reward),
            confidence_bound=_pct(0.0),  # placeholder for future UCB
        )

    # Exploitation: compute σ_i for each action
    sigma_scores = {}
    for aid in actions:
        vec = action_vectors.get(aid)
        if vec is None:
            # Missing vector → treat as worst possible SSIM
            ssim = -1.0
        else:
            ssim = compute_ssim(vec, prototype)
        sigma_scores[aid] = combined_suitability(ssim, temperature_c, params)

    total_sigma = sum(sigma_scores.values()) or 1.0
    # Propensity proportional to σ_i
    propensities = {aid: sigma_scores[aid] / total_sigma for aid in actions}

    # Select the action with highest expected reward + propensity weighting
    best_aid = max(actions, key=lambda a: _empirical_reward(a) + propensities[a])

    return BanditAction(
        action_id=best_aid,
        propensity=_pct(propensities[best_aid]),
        expected_reward=_pct(_empirical_reward(best_aid)),
        confidence_bound=_pct(0.0),  # placeholder
    )


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy vectors (3‑dimensional)
    prototype_vec = np.array([0.2, 0.5, 0.3])
    group_vecs = {
        "codex": np.array([0.1, 0.4, 0.5]),
        "groq": np.array([0.3, 0.6, 0.2]),
        "cohere": np.array([0.25, 0.45, 0.3]),
        "local_models": np.array([0.2, 0.5, 0.3]),
    }

    # Allocation example
    alloc = allocate_hybrid_workshare(
        total_units=1000.0,
        temperature_c=22.0,
        prototype=prototype_vec,
        group_vectors=group_vecs,
    )
    print("Hybrid allocation:", alloc)

    # Bandit example
    action_vecs = {
        "act_A": np.array([0.15, 0.55, 0.30]),
        "act_B": np.array([0.35, 0.45, 0.20]),
        "act_C": np.array([0.25, 0.50, 0.25]),
    }
    chosen = bandit_select_action(
        context_id="ctx_001",
        actions=list(action_vecs.keys()),
        temperature_c=22.0,
        prototype=prototype_vec,
        action_vectors=action_vecs,
        epsilon=0.05,
    )
    print("Chosen bandit action:", chosen)

    # Simulate a reward and update policy
    upd = BanditUpdate(
        context_id="ctx_001",
        action_id=chosen.action_id,
        reward=random.uniform(0, 1),
        propensity=chosen.propensity,
    )
    update_policy([upd])
    print("Policy after update:", _POLICY)