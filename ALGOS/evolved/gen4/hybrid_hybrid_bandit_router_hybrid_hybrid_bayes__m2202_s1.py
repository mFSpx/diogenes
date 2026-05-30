# DARWIN HAMMER — match 2202, survivor 1
# gen: 4
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s2.py (gen1)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s4.py (gen3)
# born: 2026-05-29T23:41:16Z

"""Hybrid Bandit‑Router & Deterministic Feature‑Bayes Model

Parents
-------
* **Bandit router (parent A)** – a lightweight multi‑armed bandit that
  computes an Upper‑Confidence‑Bound (UCB) score using a scale term
  `scale = √(∑ context_i²)` derived from a context vector.
* **Deterministic feature extractor & Bayesian updater (parent B)** – a
  reproducible pseudo‑random feature generator (`extract_master_vector`) and a
  simple conjugate‑prior Bayesian update for scalar rewards.

Mathematical Bridge
-------------------
Both parents expose a *gain* that multiplies a base quantity:

* In the bandit router the exploration term is `scale / √(1 + n_a)`,
  where `scale = √(∑ context_i²)`.
* In the feature‑Bayes side the posterior precision (inverse variance) grows
  with the number of observations; we treat the temperature‑dependent activity
  `A(T) ∈ [0,1]` as an additional precision‑like weight.

The hybrid algorithm therefore defines a **temperature‑aware scale**


S_T = A(T) * √(∑_i context_i²)


and plugs `S_T` into the UCB formula.  The same `A(T)` also weights the
Bayesian reward update, giving hotter (more biologically active) conditions
more influence on the posterior.

The module implements three core hybrid functions:
`temperature_activity`, `hybrid_select_action`, and `hybrid_update_policy`.
All other utilities are self‑contained and use only the allowed imports.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – deterministic feature extraction
# ----------------------------------------------------------------------


def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit integer hash based on SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_full_features(text: str) -> Dict[str, float]:
    """Return a reproducible pseudo‑random feature vector for *text*."""
    seed = _deterministic_hash(text) % (2**32)
    rnd = random.Random(seed)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Compact deterministic vector derived from the full feature set."""
    f = extract_full_features(text)
    return {
        "visceral_ratio": f["operator_visceral_ratio"],
        "tech_ratio": f["operator_tech_ratio"],
        "legal_osint_ratio": f["operator_legal_osint_ratio"],
        "forensic_shield_ratio": f["psyche_forensic_shield_ratio"],
        "poetic_entropy": f["psyche_poetic_entropy"],
        "dissociative_index": f["psyche_dissociative_index"],
        "bureaucratic_weaponization_index": f[
            "resilience_bureaucratic_weaponization_index"
        ],
        "resource_exhaustion_metric": f["resilience_resource_exhaustion_metric"],
        "swarm_orchestration_density": f[
            "resilience_swarm_orchestration_density"
        ],
        "corporate_grit_tension": f["rainmaker_corporate_grit_tension"],
        "countdown_density": f["rainmaker_countdown_density"],
        "asset_structuring_weight": f["rainmaker_asset_structuring_weight"],
        "agent_symmetry_ratio": f["telemetry_agent_symmetry_ratio"],
        "protocol_discipline": f["telemetry_protocol_discipline"],
        "manic_velocity": f["telemetry_manic_velocity"],
    }


# ----------------------------------------------------------------------
# Parent A – bandit router core (lightly adapted)
# ----------------------------------------------------------------------


@dataclass
class BanditPolicy:
    """Tracks counts, reward sums and Bayesian posterior for each action."""
    actions: List[str]
    # observation count per action
    n: Dict[str, int] = field(default_factory=dict)
    # sum of raw rewards per action
    reward_sum: Dict[str, float] = field(default_factory=dict)
    # posterior mean (μ) and precision (λ = 1/σ²)
    mu: Dict[str, float] = field(default_factory=dict)
    lam: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a in self.actions:
            self.n[a] = 0
            self.reward_sum[a] = 0.0
            # Prior: μ₀ = 0.0, λ₀ = 1.0 (unit precision)
            self.mu[a] = 0.0
            self.lam[a] = 1.0


# ----------------------------------------------------------------------
# Hybrid building blocks
# ----------------------------------------------------------------------


def temperature_activity(celsius: float,
                         t_opt: float = 30.0,
                         t_low: float = 10.0,
                         t_high: float = 40.0) -> float:
    """
    Normalized activity gate A(T) ∈ [0,1] based on a simple piecewise
    quadratic Schoolfield‑like curve.

    - Full activity (A≈1) between t_low and t_high.
    - Linear drop to zero outside the optimal band.
    """
    if celsius <= t_low:
        return max(0.0, (celsius - (t_low - 5)) / 5.0)  # ramp from 0 to 1
    if celsius >= t_high:
        return max(0.0, ((t_high + 5) - celsius) / 5.0)  # ramp down
    # Inside optimal band – smooth quadratic peak centered at t_opt
    width = (t_high - t_low) / 2.0
    return 1.0 - ((celsius - t_opt) / width) ** 2


def _context_norm(master_vec: Dict[str, float]) -> float:
    """Euclidean norm √(∑ v_i²) of the master feature vector."""
    arr = np.fromiter(master_vec.values(), dtype=float)
    return float(np.linalg.norm(arr))


def hybrid_select_action(policy: BanditPolicy,
                         temperature_c: float,
                         text: str,
                         exploration_coef: float = 1.0) -> str:
    """
    Temperature‑aware UCB action selection.

    For each action `a` compute

        S_T = A(T) * √(∑ context_i²)

        score_a = μ_a + exploration_coef * S_T / √(1 + n_a)

    The action with the highest score is returned.
    """
    activity = temperature_activity(temperature_c)
    master_vec = extract_master_vector(text)
    scale = activity * _context_norm(master_vec)

    best_action = None
    best_score = -math.inf

    for a in policy.actions:
        n_a = policy.n[a]
        mu_a = policy.mu[a]
        ucb = mu_a + exploration_coef * scale / math.sqrt(1.0 + n_a)
        if ucb > best_score:
            best_score = ucb
            best_action = a

    return best_action  # type: ignore[return-value]


def hybrid_update_policy(policy: BanditPolicy,
                         action: str,
                         reward: float,
                         temperature_c: float,
                         text: str,
                         reward_weight: float = 1.0) -> None:
    """
    Bayesian‑style update of the posterior for *action*.

    The temperature activity `A(T)` acts as an additional precision
    multiplier, making hot conditions count more strongly.

    Posterior update (conjugate normal‑with‑known‑variance):

        λ_new = λ_old + w
        μ_new = (λ_old·μ_old + w·r) / λ_new

    where `w = reward_weight * A(T)`.
    """
    activity = temperature_activity(temperature_c)
    weight = reward_weight * activity

    # Update raw statistics
    policy.n[action] += 1
    policy.reward_sum[action] += reward

    # Bayesian posterior update
    lam_old = policy.lam[action]
    mu_old = policy.mu[action]

    lam_new = lam_old + weight
    mu_new = (lam_old * mu_old + weight * reward) / lam_new

    policy.lam[action] = lam_new
    policy.mu[action] = mu_new


def hybrid_estimated_rewards(policy: BanditPolicy) -> Dict[str, float]:
    """
    Return the current posterior mean reward estimate for every action.
    """
    return {a: policy.mu[a] for a in policy.actions}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action space
    actions = ["alpha", "beta", "gamma"]
    policy = BanditPolicy(actions)

    # Simulated environment parameters
    temperature = 28.0  # Celsius, near optimal
    sample_text = "example request payload for feature extraction"

    # Run a few selection‑update cycles
    for step in range(10):
        chosen = hybrid_select_action(policy, temperature, sample_text)
        # Simulated reward: higher for "beta" to create a bias
        reward = {"alpha": 0.2, "beta": 0.9, "gamma": 0.4}[chosen] + random.random() * 0.1
        hybrid_update_policy(policy, chosen, reward, temperature, sample_text)

        print(
            f"Step {step:2d}: chose {chosen:5s}, reward={reward:.3f}, "
            f"estimates={hybrid_estimated_rewards(policy)}"
        )

    # Verify that the policy dictionary is still consistent
    assert all(a in policy.mu for a in actions), "Missing actions after updates"
    print("Smoke test completed successfully.")