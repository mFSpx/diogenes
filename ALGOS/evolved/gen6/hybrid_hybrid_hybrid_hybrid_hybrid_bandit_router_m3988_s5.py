# DARWIN HAMMER — match 3988, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1372_s0.py (gen5)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# born: 2026-05-29T23:52:59Z

"""Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s0 + hybrid_bandit_router_poikilotherm_schoolf_m20_s0
This module fuses the health‑centric circuit‑breaker / Gini analysis of the *Fisher* parent with the
temperature‑driven bandit routing of the *Bandit‑Schoolfield* parent.

Mathematical bridge
------------------
* The **health** metric from the Fisher side is a scalar in [0,1].
* The **temperature** used by the Schoolfield model is derived from health by a linear map
  `T_C = T_low + health * (T_high - T_low)`, turning health into a biologically plausible
  temperature (°C).
* The **developmental rate** `ρ(T_K)` (Schoolfield equation) modulates each bandit action’s
  propensity.  Higher health → higher temperature → higher developmental rate → larger propensity.
* The **Gini coefficient** of the Doomsday‑Calendar weekday count vector supplies an additional
  context factor `γ = 1 - Gini`.  It scales the expected reward, rewarding more equitable
  (low‑Gini) contexts.

The hybrid decision score for an action *a* therefore becomes  


score(a) = γ * ρ(T_K) * health * (expected_reward_a + confidence_a)


where `confidence_a` follows a classic Upper‑Confidence‑Bound term.
The circuit breaker aborts the selection when failures exceed a threshold, protecting the system
from runaway exploitation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    expected_reward: float = 0.0
    count: int = 0
    total_reward: float = 0.0
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


# ----------------------------------------------------------------------
# Helper mathematics (Gini, health, temperature, Schoolfield)
# ----------------------------------------------------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D array."""
    if x.ndim != 1:
        raise ValueError("gini_coefficient expects a 1‑D array")
    if np.any(x < 0):
        raise ValueError("negative values are not allowed for Gini")
    n = x.size
    if n == 0:
        return 0.0
    sorted_x = np.sort(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    sum_x = cumx[-1]
    if sum_x == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumx) / sum_x) / n
    return gini


def compute_health(
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
) -> float:
    """
    health = (1 - (reconstruction_risk_score * failure_rate)) * (1 - recovery_priority)

    where failure_rate = failures / failure_threshold (capped at 1.0).
    The result is clipped to [0,1].
    """
    failure_rate = min(failures / max(failure_threshold, 1), 1.0)
    health = (1.0 - (reconstruction_risk_score * failure_rate)) * (1.0 - recovery_priority)
    return max(0.0, min(health, 1.0))


def temperature_from_health(health: float, low_c: float = 5.0, high_c: float = 40.0) -> float:
    """
    Linear mapping from health∈[0,1] to Celsius temperature.
    low_c corresponds to health=0, high_c to health=1.
    """
    return low_c + health * (high_c - low_c)


def celsius_to_kelvin(c: float) -> float:
    return c + 273.15


def developmental_rate(
    temp_k: float, params: SchoolfieldParams = SchoolfieldParams()
) -> float:
    """
    Schoolfield‐Rollinson temperature‐dependent rate.
    Returns a dimensionless factor; values >1 indicate accelerated development.
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Hybrid policy engine
# ----------------------------------------------------------------------
_POLICY: Dict[str, Tuple[float, int]] = {}  # action_id -> (total_reward, count)


def reset_policy() -> None:
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Aggregate rewards per action."""
    for u in updates:
        total, cnt = _POLICY.get(u.action_id, (0.0, 0))
        _POLICY[u.action_id] = (total + u.reward, cnt + 1)


def expected_reward(action_id: str) -> float:
    total, cnt = _POLICY.get(action_id, (0.0, 0))
    return total / cnt if cnt else 0.0


def confidence_bound(action_id: str, total_selections: int, alpha: float = 2.0) -> float:
    """Upper‑Confidence‑Bound term."""
    _, cnt = _POLICY.get(action_id, (0.0, 0))
    if cnt == 0:
        return float("inf")
    return math.sqrt(alpha * math.log(max(total_selections, 1)) / cnt)


def hybrid_score(
    action_id: str,
    health: float,
    gini_factor: float,
    dev_rate: float,
    total_selections: int,
) -> float:
    """
    Composite score mixing:
    - health (scalar)
    - gini_factor = 1 - Gini (higher = more equitable)
    - developmental rate (temperature‑derived)
    - classic bandit UCB components
    """
    exp_r = expected_reward(action_id)
    conf = confidence_bound(action_id, total_selections)
    return gini_factor * dev_rate * health * (exp_r + conf)


def select_action(
    actions: List[BanditAction],
    health: float,
    gini_coeff: float,
    temperature_c: float,
) -> BanditAction:
    """
    Choose the action with the highest hybrid_score.
    If the circuit breaker would block the selection, raise RuntimeError.
    """
    dev_rate = developmental_rate(celsius_to_kelvin(temperature_c))
    gini_factor = 1.0 - gini_coeff
    total_selections = sum(_POLICY.get(a.action_id, (0.0, 0))[1] for a in actions) + 1

    scores = {
        a.action_id: hybrid_score(
            a.action_id, health, gini_factor, dev_rate, total_selections
        )
        for a in actions
    }
    best_id = max(scores, key=scores.get)
    # Return a new BanditAction with updated expected_reward from policy
    exp_r = expected_reward(best_id)
    cnt = _POLICY.get(best_id, (0.0, 0))[1]
    return BanditAction(
        action_id=best_id,
        expected_reward=exp_r,
        count=cnt,
        total_reward=_POLICY.get(best_id, (0.0, 0))[0],
        algorithm="hybrid",
    )


# ----------------------------------------------------------------------
# Example integration function demonstrating the full hybrid loop
# ----------------------------------------------------------------------
def hybrid_step(
    breaker: EndpointCircuitBreaker,
    reconstruction_risk_score: float,
    failures: int,
    failure_threshold: int,
    recovery_priority: float,
    weekday_counts: np.ndarray,
    actions: List[BanditAction],
) -> Tuple[BanditAction, float]:
    """
    Perform one decision step:
    1. Compute health via the Fisher formulation.
    2. Derive temperature from health.
    3. Compute Gini from weekday_counts.
    4. Select a bandit action using the hybrid score.
    5. Simulate a stochastic reward (for demo) and update policy.
    6. Record success/failure in the circuit breaker.
    Returns the chosen action and the simulated reward.
    """
    if not breaker.allow():
        raise RuntimeError("Circuit breaker open – cannot proceed")

    health = compute_health(
        reconstruction_risk_score, failures, failure_threshold, recovery_priority
    )
    temp_c = temperature_from_health(health)
    gini = gini_coefficient(weekday_counts)

    chosen = select_action(actions, health, gini, temp_c)

    # --- Demo reward model -------------------------------------------------
    # Reward is higher when health and temperature are high, plus some noise.
    base_reward = health * (1.0 + 0.1 * random.random())
    noise = random.uniform(-0.05, 0.05)
    reward = max(0.0, base_reward + noise)

    # Update policy
    update_policy(
        [
            BanditUpdate(
                context_id="hybrid_step",
                action_id=chosen.action_id,
                reward=reward,
                propensity=chosen.expected_reward,
            )
        ]
    )

    # Record outcome in circuit breaker
    if reward < 0.2:  # arbitrary threshold for failure
        breaker.record_failure()
    else:
        breaker.record_success()

    return chosen, reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    actions = [
        BanditAction(action_id="A"),
        BanditAction(action_id="B"),
        BanditAction(action_id="C"),
    ]

    # Synthetic weekday counts (Monday‑Sunday)
    weekday_counts = np.array([12, 15, 13, 14, 16, 11, 9])

    # Run a few hybrid steps
    for step in range(5):
        try:
            chosen, reward = hybrid_step(
                breaker=breaker,
                reconstruction_risk_score=0.3,
                failures=breaker.failures,
                failure_threshold=breaker.failure_threshold,
                recovery_priority=0.2,
                weekday_counts=weekday_counts,
                actions=actions,
            )
            print(
                f"Step {step+1}: chose {chosen.action_id}, reward={reward:.3f}, health={compute_health(0.3, breaker.failures, breaker.failure_threshold, 0.2):.3f}"
            )
        except RuntimeError as e:
            print(f"Step {step+1}: {e}")
            break

    # Final policy snapshot
    print("\nFinal policy state:")
    for aid, (tot, cnt) in _POLICY.items():
        print(f"  Action {aid}: total_reward={tot:.3f}, count={cnt}")