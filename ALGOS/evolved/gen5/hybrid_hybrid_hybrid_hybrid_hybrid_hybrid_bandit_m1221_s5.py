# DARWIN HAMMER — match 1221, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0.py (gen4)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1.py (gen4)
# born: 2026-05-29T23:34:30Z

"""Hybrid Bandit-Temperature-Gini Algorithm

Parents:
- hybrid_hybrid_hybrid_bandit_hybrid_hybrid_doomsd_m988_s0.py
- hybrid_hybrid_bandit_router_hybrid_hybrid_bandit_m534_s1.py

Mathematical Bridge:
Both parents expose a multi‑armed bandit core that maintains per‑action reward
statistics.  The second parent augments this core with a temperature‑dependent
developmental rate (Schoolfield model).  The first parent provides a dispersion
measure (implicitly via Gini‑style reasoning) that can modulate exploration.
The fusion therefore consists of:

1. Computing a temperature scaling factor  τ = developmental_rate(T)  from the
   Schoolfield model (parent B).
2. Adjusting each action’s empirical mean reward  r̂  by τ, i.e. r̃ = τ·r̂.
3. Computing a Gini coefficient G over the vector of adjusted rewards
   to quantify inequality across actions (parent A inspiration).
4. Using G to adapt the confidence bound in the Upper‑Confidence‑Bound (UCB)
   selection rule: UCB = r̃ + α·(1+G)/√(n+1).

The resulting selector simultaneously respects temperature‑dependent reward
dynamics and the inequality‑driven exploration pressure, yielding a unified
hybrid algorithm.

"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures shared by both parents
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
# Schoolfield temperature model (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0                # baseline rate at 25 °C
    delta_h_activation: float = 12_000.0   # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15              # low‑temperature cutoff (K)
    t_high: float = 307.15             # high‑temperature cutoff (K)
    delta_h_low: float = -45_000.0     # low‑temperature enthalpy (J mol⁻¹)
    delta_h_high: float = 65_000.0     # high‑temperature enthalpy (J mol⁻¹)
    r_cal: float = 1.987               # gas constant (cal mol⁻¹ K⁻¹)

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    Returns a dimensionless scaling factor τ ∈ (0, ∞).
    """
    if temp_k <= 0:
        raise ValueError("temperature must be positive Kelvin")
    # Numerator – Arrhenius term
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / 298.15 - 1 / temp_k)
    )
    # Denominator – low and high temperature deactivation
    low_term = math.exp(
        (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / temp_k)
    )
    high_term = math.exp(
        (params.delta_h_high / params.r_cal) * (1 / params.t_high - 1 / temp_k)
    )
    denominator = 1 + low_term + high_term
    return numerator / denominator

# ----------------------------------------------------------------------
# Bandit policy store (shared)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}  # action_id → [cumulative_reward, count]

def reset_policy() -> None:
    """Erase all learned statistics."""
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of BanditUpdate objects into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _raw_reward(action_id: str) -> float:
    """Empirical mean reward for a given action (unscaled)."""
    total, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return total / cnt if cnt > 0 else 0.0

def _count(action_id: str) -> float:
    """Number of times an action has been selected."""
    return _POLICY.get(action_id, [0.0, 0.0])[1]

# ----------------------------------------------------------------------
# Gini coefficient (inspired by Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a 1‑D array.
    G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mean(x))
    """
    if values.size == 0:
        return 0.0
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    diff_sum = np.abs(values[:, None] - values[None, :]).sum()
    n = values.size
    return diff_sum / (2 * n * n * mean)

# ----------------------------------------------------------------------
# Hybrid selection mechanism
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "ucb_temp_gini",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a temperature‑scaled UCB rule augmented with a
    Gini‑based exploration factor.

    Parameters
    ----------
    context : dict
        Must contain a key ``'temperature_c'`` (float) representing ambient
        temperature in Celsius.  Additional numeric entries are ignored for the
        selection rule but may be used by callers.
    actions : list[str]
        Candidate action identifiers.
    algorithm : str
        Currently only ``'ucb_temp_gini'`` or ``'epsilon_greedy'`` are supported.
    epsilon : float
        Exploration probability for the epsilon‑greedy fallback.
    seed : int | str | None
        Random seed for reproducibility.

    Returns
    -------
    BanditAction
        The selected action together with its propensity, scaled expected reward,
        confidence bound and the algorithm name.
    """
    if not actions:
        raise ValueError("actions list cannot be empty")
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1️⃣ Temperature scaling
    # ------------------------------------------------------------------
    temp_c = context.get("temperature_c", 25.0)  # default to 25 °C if absent
    tau = developmental_rate(c_to_k(temp_c))

    # ------------------------------------------------------------------
    # 2️⃣ Raw statistics
    # ------------------------------------------------------------------
    raw_means = np.array([_raw_reward(a) for a in actions])
    counts = np.array([_count(a) for a in actions])

    # Apply temperature scaling to means
    scaled_means = tau * raw_means

    # ------------------------------------------------------------------
    # 3️⃣ Gini coefficient over scaled means (global, same for all actions)
    # ------------------------------------------------------------------
    gini = gini_coefficient(scaled_means)

    # ------------------------------------------------------------------
    # 4️⃣ Selection rule
    # ------------------------------------------------------------------
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        # Upper‑Confidence‑Bound with temperature and Gini influence
        # α is a tunable exploration constant; we embed Gini inside it.
        alpha = 0.1 * (1.0 + gini)  # larger inequality → larger exploration term
        ucb_values = scaled_means + alpha / np.sqrt(1.0 + counts)
        # Resolve ties deterministically via rng
        max_ucb = np.max(ucb_values)
        candidates = [a for a, val in zip(actions, ucb_values) if np.isclose(val, max_ucb)]
        chosen = rng.choice(candidates)

    # Propensity is uniform across the action set (could be refined)
    propensity = 1.0 / len(actions)
    expected = _raw_reward(chosen) * tau
    confidence = alpha / math.sqrt(1.0 + _count(chosen))

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence,
        algorithm=algorithm,
    )

# ----------------------------------------------------------------------
# Helper: temperature‑aware policy update
# ----------------------------------------------------------------------
def temperature_aware_update(
    context_id: str,
    action_id: str,
    raw_reward: float,
    context: Dict[str, float],
) -> BanditUpdate:
    """
    Wrap a raw reward with the current temperature scaling before creating a
    BanditUpdate.  The scaling is applied symmetrically to both reward and
    propensity to keep the estimator unbiased under the assumed model.
    """
    temp_c = context.get("temperature_c", 25.0)
    tau = developmental_rate(c_to_k(temp_c))
    scaled_reward = raw_reward * tau
    # Propensity is uniform for simplicity; more sophisticated schemes can be added.
    propensity = 1.0
    return BanditUpdate(
        context_id=context_id,
        action_id=action_id,
        reward=scaled_reward,
        propensity=propensity,
    )

# ----------------------------------------------------------------------
# Demonstration function: run a single bandit episode
# ----------------------------------------------------------------------
def run_episode(
    actions: List[str],
    temperature_c: float,
    rng_seed: int | str | None = 42,
) -> Tuple[BanditAction, BanditUpdate]:
    """
    Simulate one interaction:
    1. Build a context containing the temperature.
    2. Select an action via ``hybrid_select_action``.
    3. Generate a synthetic raw reward (here a noisy sinusoid) and create a
       temperature‑aware update.
    4. Apply the update to the global policy.
    Returns the chosen action and the update object.
    """
    context = {"temperature_c": temperature_c}
    chosen = hybrid_select_action(context, actions, seed=rng_seed)

    # Synthetic reward: sinusoidal function of a hidden true value per action
    rng = random.Random(rng_seed)
    true_means = {a: math.sin(hash(a) % 360 * math.pi / 180) for a in actions}
    noise = rng.gauss(0, 0.1)
    raw_reward = true_means[chosen.action_id] + noise

    upd = temperature_aware_update(
        context_id="episode_1",
        action_id=chosen.action_id,
        raw_reward=raw_reward,
        context=context,
    )
    update_policy([upd])
    return chosen, upd

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    reset_policy()
    action_set = [f"arm_{i}" for i in range(5)]

    # Run a few episodes with varying temperatures to exercise the hybrid logic
    for i, temp in enumerate([15.0, 25.0, 35.0, 20.0, 30.0]):
        chosen_action, update = run_episode(action_set, temperature_c=temp, rng_seed=i)
        print(
            f"Episode {i+1:02d} | Temp={temp:5.1f}°C | Chosen={chosen_action.action_id:7s} "
            f"| ScaledReward={chosen_action.expected_reward: .3f} | "
            f"Confidence={chosen_action.confidence_bound: .3f}"
        )
    # Final policy snapshot
    print("\nFinal policy statistics:")
    for aid, (cum, cnt) in _POLICY.items():
        print(f"  {aid}: total={cum:.3f}, count={cnt:.0f}, mean={cum/cnt if cnt else 0:.3f}")