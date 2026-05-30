# DARWIN HAMMER — match 2323, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_pheromone_inf_m495_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s3.py (gen3)
# born: 2026-05-29T23:41:52Z

"""Hybrid Bandit‑State‑Space Duality with Pheromone‑Infotaxis

Parents
-------
* **Parent A** – Bandit‑Router with Pheromone‑Infotaxis (bandit action
  propensities, entropy‑based pheromone bonus, variational free‑energy
  inspired update).
* **Parent B** – Poikilotherm State‑Space Duality (temperature‑dependent
  developmental rate `r(t)` scaling a linear state‑space model and its
  semiseparable matrix representation).

Mathematical Bridge
-------------------
The developmental rate `r(t)=developmental_rate(T(t))` is used as a *scalar
gain* that simultaneously

1. **Scales the bandit action propensity** – a higher temperature (larger
   `r`) amplifies the influence of the expected reward and the pheromone
   entropy bonus on the soft‑max propensity.
2. **Scales the state‑transition matrix** `A_t = r(t)·A₀` of a linear
   dynamical system `x_{t+1}=A_t x_t + B u_t`.  Thus the same physiological
   factor that modulates decision confidence also drives the underlying
   latent dynamics.

The pheromone field provides an entropy‑based bonus  
`b_i = -η·p_i·log(p_i)` (with `p_i` a normalized pheromone concentration) that
is added to the expected reward before the soft‑max.  The resulting
propensity `π_i` enters the control input `u_t` of the state‑space model,
closing the loop between belief‑driven action selection and temperature‑
modulated dynamics.

The hybrid algorithm therefore consists of:

* `temperature_dependent_state_transition` – builds `A_t = r·A₀`.
* `compute_propensities` – combines expected reward, pheromone bonus and
  developmental rate into a soft‑max propensity.
* `hybrid_ssm_step` – advances the latent state with the selected action.
* `update_policy` – variational‑free‑energy‑like update of the bandit policy.

The three functions below illustrate the complete hybrid operation."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A: Bandit core (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    expected_reward: float
    confidence_bound: float = 0.0  # placeholder, not used in this hybrid
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# Global policy storage: action_id -> [total_reward, count, propensity]
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()

def _average_reward(action_id: str) -> float:
    total, count, _ = _POLICY.get(action_id, [0.0, 0.0, 0.0])
    return total / count if count > 0 else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    """Variational‑free‑energy inspired policy update."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        stats[0] += float(u.reward)          # cumulative reward
        stats[1] += 1.0                       # count
        # Simple exponential moving average for propensity (learning rate α=0.1)
        α = 0.1
        old_prop = stats[2]
        stats[2] = (1 - α) * old_prop + α * u.propensity

def get_policy_snapshot() -> Dict[str, Tuple[float, float, float]]:
    """Return a copy of the policy for inspection."""
    return {k: tuple(v) for k, v in _POLICY.items()}

# ----------------------------------------------------------------------
# Parent‑B: Poikilotherm developmental rate & state‑space duality
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (arbitrary units)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15            # K  (10 °C)
    t_high: float = 307.15           # K  (34 °C)
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987             # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield‑Rollinson temperature‑dependent developmental rate.

    Returns a dimensionless factor r(T) such that r(298.15 K)=1 when
    `rho_25` = 1.
    """
    if temp_k <= 0.0:
        raise ValueError("Temperature must be > 0 K")
    # Convert calories to Joules for consistency
    R = params.r_cal * 4.184  # J mol⁻¹ K⁻¹
    term1 = math.exp(
        -params.delta_h_activation / (R * temp_k)
        + params.delta_h_activation / (R * 298.15)
    )
    term2 = 1.0 + math.exp(
        params.delta_h_low / R *
        (1.0 / params.t_low - 1.0 / temp_k)
    )
    term3 = 1.0 + math.exp(
        -params.delta_h_high / R *
        (1.0 / params.t_high - 1.0 / temp_k)
    )
    return params.rho_25 * term1 / (term2 * term3)


def temperature_dependent_state_transition(A0: np.ndarray,
                                            temp_k: float) -> np.ndarray:
    """
    Scale the base state‑transition matrix A0 by the developmental rate.
    """
    r = developmental_rate(temp_k)
    return r * A0


def hybrid_ssm_step(x: np.ndarray,
                    A: np.ndarray,
                    B: np.ndarray,
                    u: np.ndarray) -> np.ndarray:
    """
    One step of the linear state‑space model:
        x_{t+1} = A x_t + B u_t
    """
    return A @ x + B @ u


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def compute_pheromone_bonus(pheromone_map: Dict[str, float],
                            eta: float = 0.5) -> Dict[str, float]:
    """
    Entropy‑based bonus for each action:
        b_i = -η * p_i * log(p_i)
    where p_i are normalized pheromone concentrations.
    """
    total = sum(pheromone_map.values())
    if total == 0.0:
        # No pheromone → zero bonus
        return {k: 0.0 for k in pheromone_map}
    probs = {k: v / total for k, v in pheromone_map.items()}
    bonus = {k: -eta * p * math.log(p + 1e-12) for k, p in probs.items()}
    return bonus


def compute_propensities(actions: List[BanditAction],
                         pheromone_map: Dict[str, float],
                         temperature_k: float,
                         beta: float = 5.0) -> Dict[str, float]:
    """
    Combine expected reward, pheromone entropy bonus and temperature‑scaled
    developmental rate into soft‑max propensities.

    π_i ∝ exp( β·r_i·r(T) + β·b_i )
    where
        r_i = expected reward of action i,
        r(T) = developmental_rate,
        b_i = pheromone entropy bonus.
    """
    r_temp = developmental_rate(temperature_k)
    bonus = compute_pheromone_bonus(pheromone_map)
    logits = {}
    for act in actions:
        r_i = act.expected_reward
        b_i = bonus.get(act.action_id, 0.0)
        logits[act.action_id] = beta * (r_i * r_temp + b_i)
    # Soft‑max
    max_logit = max(logits.values()) if logits else 0.0
    exp_vals = {k: math.exp(v - max_logit) for k, v in logits.items()}
    sum_exp = sum(exp_vals.values()) + 1e-12
    propensities = {k: v / sum_exp for k, v in exp_vals.items()}
    return propensities


def hybrid_step(state: np.ndarray,
                A0: np.ndarray,
                B: np.ndarray,
                actions: List[BanditAction],
                pheromone_map: Dict[str, float],
                temperature_k: float) -> Tuple[np.ndarray, List[BanditUpdate]]:
    """
    Execute a single hybrid iteration:
      1. Compute temperature‑scaled propensities.
      2. Sample an action according to the propensities.
      3. Simulate a stochastic reward.
      4. Update the bandit policy.
      5. Advance the latent state with the selected action as control input.
    Returns the new state and a list of BanditUpdate objects (length 1).
    """
    # 1. Propensities
    propensities = compute_propensities(actions, pheromone_map, temperature_k)

    # 2. Sample action
    ids, probs = zip(*propensities.items())
    chosen_id = random.choices(ids, weights=probs, k=1)[0]
    chosen_action = next(a for a in actions if a.action_id == chosen_id)

    # 3. Simulated reward (Gaussian around expected reward)
    reward_noise = random.gauss(0.0, 0.1)
    reward = max(0.0, chosen_action.expected_reward + reward_noise)

    # 4. Policy update (store the propensity before soft‑max normalisation)
    update = BanditUpdate(
        context_id="hybrid_step",
        action_id=chosen_id,
        reward=reward,
        propensity=propensities[chosen_id],
    )
    update_policy([update])

    # 5. State‑space advance
    A_t = temperature_dependent_state_transition(A0, temperature_k)
    # Control vector u is the (scalar) propensity of the chosen action
    u = np.array([propensities[chosen_id]])
    new_state = hybrid_ssm_step(state, A_t, B, u)

    return new_state, [update]


# ----------------------------------------------------------------------
# Convenience helpers for building a simple problem instance
# ----------------------------------------------------------------------
def build_dummy_actions(n: int = 3) -> List[BanditAction]:
    """Create `n` actions with random expected rewards."""
    actions = []
    for i in range(n):
        action_id = f"a{i}"
        expected_reward = random.uniform(0.0, 1.0)
        actions.append(BanditAction(action_id=action_id,
                                    expected_reward=expected_reward))
    return actions


def build_pheromone_map(actions: List[BanditAction]) -> Dict[str, float]:
    """Assign a random pheromone concentration to each action."""
    return {a.action_id: random.uniform(0.1, 1.0) for a in actions}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Initialise policy and system matrices
    reset_policy()
    actions = build_dummy_actions(4)
    pheromone = build_pheromone_map(actions)

    # Simple 1‑D latent state (scalar) with base transition A0 = 0.9, B = 0.1
    A0 = np.array([[0.9]])
    B = np.array([[0.1]])
    state = np.array([[0.0]])  # start at zero

    # Run a short trajectory with a slowly varying temperature profile
    temps = np.linspace(285.0, 303.0, 10)  # K

    print("Step | Temp(K) | Action | Reward | Propensity | State")
    for step, T in enumerate(temps, 1):
        state, updates = hybrid_step(state, A0, B, actions, pheromone, T)
        upd = updates[0]
        print(f"{step:4d} | {T:7.2f} | {upd.action_id:6s} | "
              f"{upd.reward:6.3f} | {upd.propensity:10.4f} | {state.ravel()[0]:.4f}")

    # Final policy snapshot
    print("\nFinal policy (total_reward, count, propensity):")
    for aid, vals in get_policy_snapshot().items():
        print(f"{aid}: {vals}")