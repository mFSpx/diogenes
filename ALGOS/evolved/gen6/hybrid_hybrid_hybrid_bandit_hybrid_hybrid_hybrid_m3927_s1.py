# DARWIN HAMMER — match 3927, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2011_s3.py (gen5)
# born: 2026-05-29T23:52:32Z

"""Hybrid Bandit‑Router + Developmental‑Rate Epistemic Algorithm
================================================================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit that selects actions
  (`select_action`) and maintains a simple reward policy (`_POLICY`).  
* **Parent B** – a temperature‑dependent developmental‑rate model
  (`developmental_rate`) together with epistemic confidence modulation
  (`epistemic_modulated_step_size`).

**Mathematical bridge**

The reward estimate of each bandit arm is scaled by the *developmental rate*
computed at a given temperature `temp_k`.  This makes the perceived value of
an action temperature‑aware, i.e.  


scaled_reward(a) = raw_reward(a) * developmental_rate(temp_k)


Furthermore, the learning‑rate (or step size) used when updating the policy is
modulated by an epistemic confidence flag (`FACT`, `PROBABLE`, …) via  


μ_eff = μ * developmental_rate(temp_k) * confidence(flag)


A lightweight Count‑Min Sketch (`CountMinSketch`) provides a differentially‑
private, compact estimator for the number of times each action has been
selected; its estimate is also temperature‑scaled when used for reward
adjustment.  The three public functions below demonstrate the hybrid operation.

"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
import numpy as np
from datetime import date as dt

# ----------------------------------------------------------------------
# Parent‑A structures (bandit)
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

_POLICY: dict[str, list[float]] = {}          # action_id -> [cumulative_reward, count]

def reset_policy() -> None:
    """Clear the global policy."""
    _POLICY.clear()

def _raw_reward(action: str) -> float:
    total, n = _POLICY.get(action, (0.0, 0.0))
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Parent‑B structures (temperature & epistemic)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
_EPISTEMIC_CONFIDENCE = {
    "FACT": 1.0,
    "PROBABLE": 0.9,
    "POSSIBLE": 0.8,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.5,
}

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Temperature‑scaled developmental rate (Schoolfield model)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def epistemic_modulated_step_size(mu: float, temp_k: float, flag: str) -> float:
    """Effective learning‑rate after epistemic confidence modulation."""
    confidence = _EPISTEMIC_CONFIDENCE[flag]
    rate = developmental_rate(temp_k)
    return mu * rate * confidence

# ----------------------------------------------------------------------
# Simple Count‑Min Sketch for privacy‑preserving count estimation
# ----------------------------------------------------------------------
class CountMinSketch:
    """Very small Count‑Min Sketch with deterministic hash functions."""
    def __init__(self, width: int = 256, depth: int = 4, seed: int = 7):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.int64)
        rng = random.Random(seed)
        self.seeds = [rng.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: str, i: int) -> int:
        return (hash(item) ^ self.seeds[i]) % self.width

    def add(self, item: str, increment: int = 1) -> None:
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item: str) -> int:
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_select_action(
    context: dict[str, float],
    actions: list[str],
    temp_k: float,
    flag: str,
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a bandit rule whose reward estimates are
    temperature‑scaled by ``developmental_rate`` and epistemically modulated.

    Returns a ``BanditAction`` containing the selected ``action_id`` and the
    scaled expected reward.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # epsilon‑greedy fallback
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    else:
        # Compute temperature‑scaled upper confidence bound for each arm
        scale = math.sqrt(sum(float(v) * float(v) for v in context.values())) if context else 1.0
        rate = developmental_rate(temp_k)
        def ucb(a: str) -> float:
            raw = _raw_reward(a)
            count = _POLICY.get(a, [0.0, 0.0])[1]
            confidence = rate * scale / math.sqrt(1 + count)
            return raw * rate + confidence
        chosen = max(actions, key=ucb)

    # Propensity is the probability of selection under the current policy
    propensity = 1.0 / len(actions) if algorithm == "epsilon_greedy" else 0.0
    expected = _raw_reward(chosen) * developmental_rate(temp_k)
    confidence_bound = epistemic_modulated_step_size(mu=0.5, temp_k=temp_k, flag=flag)

    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=expected,
        confidence_bound=confidence_bound,
        algorithm=algorithm,
    )

def hybrid_update_policy(
    updates: list[BanditUpdate],
    cms: CountMinSketch,
    temp_k: float,
) -> None:
    """
    Update the global reward policy and the Count‑Min Sketch.
    Rewards are first temperature‑scaled before aggregation.
    """
    rate = developmental_rate(temp_k)
    for u in updates:
        # temperature‑scaled reward contribution
        scaled_reward = u.reward * rate
        # update policy dict
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += scaled_reward
        s[1] += 1.0
        # update CMS with the (scaled) count of selections
        cms.add(u.action_id, increment=1)

def hybrid_estimated_value(
    action_id: str,
    cms: CountMinSketch,
    temp_k: float,
    flag: str,
) -> float:
    """
    Produce a privacy‑preserving, temperature‑aware estimate of an arm's value.
    The count from the CMS is multiplied by the developmental rate and
    epistemic confidence.
    """
    count_est = cms.estimate(action_id)
    rate = developmental_rate(temp_k)
    confidence = _EPISTEMIC_CONFIDENCE[flag]
    # If the arm has never been observed, fall back to the raw policy reward.
    if count_est == 0:
        base = _raw_reward(action_id)
    else:
        base = count_est / max(1, _POLICY.get(action_id, [0.0, 1.0])[1])
    return base * rate * confidence

def hybrid_temperature_epistemic_state_space(
    A: np.ndarray,
    temp_k: float,
    mu: float,
    flag: str,
    section: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Combine the temperature‑scaled matrix transformation from Parent B
    with an epistemic‑modulated step size, then apply a sheaf‑consistency
    measure on the provided ``section``.
    """
    # Scale the matrix by developmental rate
    scaled_A = temperature_scaled_transition(A, temp_k)
    # Apply epistemic‑modulated step size
    step = epistemic_modulated_step_size(mu, temp_k, flag)
    transformed = scaled_A @ (section * step)
    # Consistency measure (sum of absolute weighted values)
    consistency = sheaf_consistency_measure(transformed, weights)
    # Return a vector that appends the consistency scalar for downstream use
    return np.append(transformed.flatten(), consistency)

# ----------------------------------------------------------------------
# Helper utilities from Parent B (re‑implemented for completeness)
# ----------------------------------------------------------------------
def temperature_scaled_transition(A: np.ndarray, temp_k: float) -> np.ndarray:
    """Scale a matrix by the developmental rate at ``temp_k``."""
    rate = developmental_rate(temp_k)
    return rate * A

def sheaf_consistency_measure(section: np.ndarray, weights: np.ndarray) -> float:
    """L1‑type consistency measure weighted by ``weights``."""
    return float(np.sum(np.abs(section * weights)))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    rng = random.Random(42)

    # Define a tiny context and action set
    context = {"feature1": 0.7, "feature2": -0.2}
    actions = ["alloc_A", "alloc_B", "alloc_C"]
    temp_k = 298.15  # ~25 °C
    flag = "PROBABLE"

    # Initialise CMS
    cms = CountMinSketch(width=128, depth=3, seed=123)

    # Select an action using the hybrid selector
    chosen = hybrid_select_action(
        context=context,
        actions=actions,
        temp_k=temp_k,
        flag=flag,
        algorithm="linucb",
        seed=13,
    )
    print("Chosen action:", chosen)

    # Simulate a reward and perform a policy update
    reward = rng.uniform(0, 1)
    update = BanditUpdate(
        context_id="ctx_1",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    hybrid_update_policy([update], cms, temp_k)

    # Estimate value of the chosen arm
    est = hybrid_estimated_value(chosen.action_id, cms, temp_k, flag)
    print(f"Estimated value for {chosen.action_id!r}:", est)

    # Demonstrate the state‑space transformation
    A = np.eye(3)
    section = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 0.5, 0.2])
    transformed = hybrid_temperature_epistemic_state_space(
        A=A,
        temp_k=temp_k,
        mu=0.05,
        flag=flag,
        section=section,
        weights=weights,
    )
    print("Transformed state space (last element = consistency):", transformed)