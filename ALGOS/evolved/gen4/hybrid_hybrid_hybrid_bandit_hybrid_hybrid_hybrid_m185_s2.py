# DARWIN HAMMER — match 185, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s1.py (gen3)
# born: 2026-05-29T23:27:25Z

"""Hybrid Bandit-Temperature-Store Algorithm
Parents:
- hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s0.py (temperature‑dependent Schoolfield model)
- hybrid_hybrid_hybrid_bandit_hybrid_capybara_opti_m41_s1.py (contextual bandit + honeybee virtual store)

Mathematical bridge:
The Schoolfield temperature function ρ(T) is used as a *temperature‑dependent learning‑rate factor* λ_T.
The honeybee “store” equation provides a *memory‑based scaling* σ_S = 1 / (1 + exp(‑S_k)) where S_k is the stored value for a key.
The effective learning‑rate for the LinUCB‑style bandit becomes
    η = η_0 · λ_T · σ_S
where η_0 is a base learning‑rate. This fuses the temperature activity curve of Parent A with the store dynamics of Parent B.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (identical to parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Parent A – Schoolfield temperature model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K
    t_high: float = 307.15           # K
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈ 8.314 J mol⁻¹ K⁻¹)


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent rate.

    ρ(T) = (ρ_25 * (T/298.15) *
            exp[ -ΔH_a / R * (1/T - 1/298.15) ]) /
           (1 + exp[ ΔH_l / R * (1/T_l - 1/T) ] +
                exp[ ΔH_h / R * (1/T - 1/T_h) ])

    Returns a dimensionless factor (≈1 at 25 °C) that can be used to
    scale learning‑rates or propensities.
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")

    # Numerator: Arrhenius‑type activation
    num = (params.rho_25 *
           (temp_k / 298.15) *
           math.exp(-(params.delta_h_activation / (params.r_cal * 4.184)) *
                    (1.0 / temp_k - 1.0 / 298.15)))

    # Denominator: low‑ and high‑temperature deactivation terms
    low_term = math.exp((params.delta_h_low /
                         (params.r_cal * 4.184)) *
                        (1.0 / params.t_low - 1.0 / temp_k))
    high_term = math.exp((params.delta_h_high /
                          (params.r_cal * 4.184)) *
                         (1.0 / temp_k - 1.0 / params.t_high))

    den = 1.0 + low_term + high_term
    return num / den


# ----------------------------------------------------------------------
# Parent B – Virtual store (honeybee) and LinUCB bandit
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}          # action -> [cumulative_reward, count]
_STORE: Dict[str, float] = {}                # key -> stored value (virtual VRAM)


def reset_policy() -> None:
    """Clear learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def honeybee_store_update(key: str, value: float, decay: float = 0.01) -> None:
    """
    Honeybee store dynamics.
    The store behaves like an exponential moving average:
        S ← (1‑decay)·S + decay·value
    """
    old = _STORE.get(key, 0.0)
    _STORE[key] = (1.0 - decay) * old + decay * value


def honeybee_scaling(key: str) -> float:
    """
    Convert the stored value into a scaling factor σ_S ∈ (0,1).
    σ_S = 1 / (1 + exp(‑S))
    """
    s = _STORE.get(key, 0.0)
    return 1.0 / (1.0 + math.exp(-s))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_select_action(context: Dict[str, float],
                         actions: List[str],
                         algorithm: str = "linucb",
                         epsilon: float = 0.1,
                         seed: int | str | None = 7,
                         base_lr: float = 0.5) -> BanditAction:
    """
    Select an action using a LinUCB‑style score that is scaled by:
        η = base_lr × λ_T × σ_S
    where λ_T is the temperature factor from the Schoolfield model
    and σ_S is the honeybee store scaling for the chosen action.
    """
    if not actions:
        raise ValueError("No actions provided")
    random.seed(seed)

    # Extract temperature (Celsius) from context, default 25 °C
    temp_c = context.get("temp_c", 25.0)
    temp_k = c_to_k(temp_c)
    lambda_T = developmental_rate(temp_k)          # temperature factor

    # Prepare a random LinUCB theta vector
    theta = np.random.normal(0, 1, len(context))
    x = np.array(list(context.values()))
    raw_score = np.dot(theta, x)

    # Choose the action with the highest raw score
    best_idx = int(np.argmax([raw_score + random.random() for _ in actions]))
    action_id = actions[best_idx]

    # Compute store‑based scaling for this action
    sigma_S = honeybee_scaling(action_id)

    # Effective learning‑rate (used here to modulate propensity & confidence)
    eta = base_lr * lambda_T * sigma_S

    propensity = min(max(eta * random.random(), 0.0), 1.0)
    confidence_bound = min(max(eta * random.random(), 0.0), 1.0)

    return BanditAction(
        action_id=action_id,
        propensity=propensity,
        expected_reward=_reward(action_id),
        confidence_bound=confidence_bound,
        algorithm=algorithm,
    )


def hybrid_update_policy(updates: List[BanditUpdate]) -> None:
    """
    Update the reward policy and simultaneously refresh the honeybee store.
    The store receives the *reward* as its new observation, allowing the
    scaling factor σ_S to reflect recent performance.
    """
    for u in updates:
        # Update policy statistics
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

        # Update the virtual store with the received reward
        honeybee_store_update(u.action_id, u.reward)


def hybrid_expected_reward(action_id: str) -> float:
    """
    Return the temperature‑adjusted expected reward for an action.
    The raw reward is multiplied by the current temperature factor λ_T
    (using a default temperature of 25 °C) to illustrate the fusion.
    """
    raw = _reward(action_id)
    lambda_T = developmental_rate(c_to_k(25.0))  # default temperature
    return raw * lambda_T


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any previous state
    reset_policy()

    # Define a simple context containing temperature and two dummy features
    ctx = {"temp_c": 30.0, "feature_a": 0.7, "feature_b": 0.2}

    # Define possible actions
    act_list = ["click", "scroll", "hover"]

    # Select an action using the hybrid selector
    selected = hybrid_select_action(context=ctx, actions=act_list, seed=42)
    print("Selected action:", asdict(selected))

    # Simulate a reward (e.g., 1 for click, 0 otherwise)
    reward_val = 1.0 if selected.action_id == "click" else 0.0

    # Create a BanditUpdate and apply it
    upd = BanditUpdate(
        context_id="session_001",
        action_id=selected.action_id,
        reward=reward_val,
        propensity=selected.propensity,
    )
    hybrid_update_policy([upd])

    # Show updated expected reward
    exp_r = hybrid_expected_reward(selected.action_id)
    print(f"Updated expected reward for '{selected.action_id}': {exp_r:.4f}")

    # Verify store contents (debug output)
    print("Store snapshot:", {k: round(v, 4) for k, v in _STORE.items()})