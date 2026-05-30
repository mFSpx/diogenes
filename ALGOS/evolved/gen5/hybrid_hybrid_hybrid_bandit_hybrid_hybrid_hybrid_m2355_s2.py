# DARWIN HAMMER — match 2355, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s1.py (gen4)
# born: 2026-05-29T23:42:08Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBanditAI"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float
    fisher: float  # Fisher information observed for this update


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Gaussian temperature model (active inference)."""
    rho_25: float = 1.0               # baseline rate at 25°C
    t_opt: float = 298.15             # optimal temperature (K)
    sigma: float = 5.0                # spread of the Gaussian (K)
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = 0.0


@dataclass(frozen=True)
class Morphology:
    """Geometric description used as contextual features."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Global policy storage (action_id -> [total_reward, count, fisher_sum])
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear the global policy."""
    _POLICY.clear()


def _reward_estimate(action_id: str) -> float:
    total, n, _ = _POLICY.get(action_id, [0.0, 0.0, 0.0])
    return total / n if n else 0.0


def _fisher_average(action_id: str) -> float:
    _, n, fisher_sum = _POLICY.get(action_id, [0.0, 0.0, 0.0])
    return fisher_sum / n if n else 1.0  # avoid division by zero


def update_policy_with_fisher(updates: List[BanditUpdate]) -> None:
    """Incorporate rewards and observed Fisher information into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        stats[0] += float(u.reward)            # total reward
        stats[1] += 1.0                         # count
        stats[2] += float(u.fisher)             # accumulated Fisher


# ----------------------------------------------------------------------
# Temperature model (Active Inference) ------------------------------------------------
# ----------------------------------------------------------------------
def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """
    Gaussian temperature dependence:
        r(T) = rho_25 * exp( - (T - T_opt)^2 / (2 * sigma^2) )
    """
    exponent = -((temp_k - params.t_opt) ** 2) / (2.0 * params.sigma ** 2)
    return params.rho_25 * math.exp(exponent)


def normalized_activity(temp_c: float, low_c: float = 5.0) -> float:
    """Normalize activity to [0,1] using the Gaussian temperature model."""
    params = SchoolfieldParams()
    rate = developmental_rate(c_to_k(temp_c), params)
    # Simple linear scaling against a low‑temperature baseline
    baseline = developmental_rate(c_to_k(low_c), params)
    return max(0.0, (rate - baseline) / (1.0 - baseline))


# ----------------------------------------------------------------------
# Fisher information for the Gaussian temperature model
# ----------------------------------------------------------------------
def compute_fisher_information(params: SchoolfieldParams, temp_c: float) -> float:
    """
    For a Gaussian likelihood with fixed variance σ², the Fisher information
    with respect to the mean (here T_opt) is I = 1/σ².
    We return the scalar I multiplied by the normalized activity to reflect
    that higher activity yields more informative observations.
    """
    sigma_sq = params.sigma ** 2
    base_fisher = 1.0 / sigma_sq
    activity = normalized_activity(temp_c)
    return base_fisher * (0.5 + activity)  # ensure a minimum >0


# ----------------------------------------------------------------------
# Contextual dimensionality reduction (PCA) ---------------------------------
# ----------------------------------------------------------------------
def _pca_projection(matrix: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the top `n_components` principal components of `matrix` (samples x features)
    and return (projected_data, components).
    """
    # Center the data
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    # SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    projected = centered @ components.T
    return projected, components


def embed_morphology_batch(batch: List[Morphology], n_components: int = 2) -> np.ndarray:
    """Convert a batch of Morphology objects into a low‑dimensional embedding."""
    data = np.array([[m.length, m.width, m.height, m.mass] for m in batch], dtype=float)
    projected, _ = _pca_projection(data, n_components)
    return projected  # shape: (batch, n_components)


# ----------------------------------------------------------------------
# Bandit decision making with Fisher‑augmented confidence bounds
# ----------------------------------------------------------------------
def select_action(context: Morphology, possible_actions: List[str], timestep: int) -> BanditAction:
    """
    Choose the action with the highest Upper‑Confidence‑Bound (UCB):
        UCB = μ̂ + c * sqrt( (2 * log(t)) / (n * I) )
    where:
        μ̂    – empirical reward estimate,
        n    – number of times the action was selected,
        I    – average Fisher information for that action,
        c    – exploration coefficient (set to 1.0 here).
    """
    # Ensure policy entries exist
    for a in possible_actions:
        _POLICY.setdefault(a, [0.0, 0.0, 0.0])

    best_action = None
    best_score = -math.inf
    for a in possible_actions:
        mu = _reward_estimate(a)
        n = _POLICY[a][1]
        fisher = _fisher_average(a)
        ucb = mu + math.sqrt((2 * math.log(timestep)) / (n * fisher)) if n > 0 and fisher > 0 else math.inf
        if ucb > best_score:
            best_score = ucb
            best_action = a

    action_stats = _POLICY[best_action]
    return BanditAction(
        action_id=best_action,
        propensity=1.0,
        expected_reward=mu,
        confidence_bound=best_score - mu,
    )


def main():
    # Test the select_action function
    context = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)
    possible_actions = ['action1', 'action2', 'action3']
    timestep = 10
    action = select_action(context, possible_actions, timestep)
    print(action)


if __name__ == "__main__":
    main()