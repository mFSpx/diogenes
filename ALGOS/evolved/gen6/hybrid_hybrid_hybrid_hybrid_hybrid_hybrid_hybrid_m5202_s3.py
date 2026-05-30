# DARWIN HAMMER — match 5202, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_endpoi_m2625_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_worksh_m150_s0.py (gen4)
# born: 2026-05-30T00:00:36Z

"""HybridGA_Bandit
This module fuses two distinct parents:

* **Parent A** – a 2‑D Euclidean geometric‑algebra core (GA2DMultivector) that provides
  multivector arithmetic, norms and Euclidean distances between vector parts.
* **Parent B** – a temperature‑aware multi‑armed bandit that uses a Schoolfield
  developmental‑rate model to modulate a reward function and a work‑share allocator
  that biases actions according to external features (e.g. day of week).

**Mathematical bridge**

The bridge is the *reward* supplied to the bandit.  
For a candidate action we embed its parameters into a GA2DMultivector **a** and we
define a *target* multivector **τ** (the desired solution).  
The geometric distance  


d(a,τ) = ‖ (a.e1, a.e2) – (τ.e1, τ.e2) ‖


measures how far the action is from the target in the 2‑D vector sub‑space.
We convert a physical temperature **T** (Kelvin) into a *developmental rate* **R(T)**
with the Schoolfield equation (Parent B).  
The final scalar reward fed to the bandit is


R = exp( – α·d(a,τ) ) · R(T)


where **α** is a tunable scaling constant.  This couples the geometric algebra
distance (Parent A) with the temperature‑dependent reward (Parent B), allowing the
bandit to explore actions that are both geometrically promising and thermodynamically
favourable.  The work‑share allocator then redistributes the exploitation budget
according to deterministic pseudo‑features (e.g. day of week).

The module therefore implements:
1. GA2DMultivector arithmetic (Parent A).
2. Schoolfield developmental‑rate model (Parent B).
3. A temperature‑aware UCB‑style bandit that uses the fused reward.
4. A simple work‑share allocation based on external features.
"""

import math
import random
import sys
import pathlib
import datetime
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Iterable, Callable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra core
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GA2DMultivector:
    """2‑D Euclidean Clifford (geometric) algebra multivector.

    Components are ordered as (scalar, e1, e2, e12).
    """
    s: float = 0.0   # scalar
    e1: float = 0.0  # vector e1
    e2: float = 0.0  # vector e2
    e12: float = 0.0 # bivector e12

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        return GA2DMultivector(
            self.s + other.s,
            self.e1 + other.e1,
            self.e2 + other.e2,
            self.e12 + other.e12,
        )

    def __sub__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        return GA2DMultivector(
            self.s - other.s,
            self.e1 - other.e1,
            self.e2 - other.e2,
            self.e12 - other.e12,
        )

    def __mul__(self, other: "GA2DMultivector") -> "GA2DMultivector":
        """Geometric product (Cl(2,0))."""
        a0, a1, a2, a12 = self.s, self.e1, self.e2, self.e12
        b0, b1, b2, b12 = other.s, other.e1, other.e2, other.e12

        s = a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12
        e1 = a0 * b1 + a1 * b0 + a2 * b12 - a12 * b2
        e2 = a0 * b2 + a2 * b0 - a1 * b12 + a12 * b1
        e12 = a0 * b12 + a12 * b0 + a1 * b2 - a2 * b1
        return GA2DMultivector(s, e1, e2, e12)

    def reverse(self) -> "GA2DMultivector":
        """Reversion changes sign of bivector part."""
        return GA2DMultivector(self.s, self.e1, self.e2, -self.e12)

    # ------------------------------------------------------------------
    # Metric utilities
    # ------------------------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the vector part (e1, e2)."""
        return math.hypot(self.e1, self.e2)

    def distance_to(self, other: "GA2DMultivector") -> float:
        """Euclidean distance between the vector parts of two multivectors."""
        return math.hypot(self.e1 - other.e1, self.e2 - other.e2)

# ----------------------------------------------------------------------
# Parent B – Schoolfield temperature model and bandit core
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15        # low temperature bound (K)
    t_high: float = 307.15       # high temperature bound (K)
    delta_h_low: float = -45_000.0  # low‑temp enthalpy (J mol⁻¹)
    delta_h_high: float = 65_000.0  # high‑temp enthalpy (J mol⁻¹)
    r_cal: float = 1.987         # gas constant (cal mol⁻¹ K⁻¹)


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, params: SchoolfieldParams) -> float:
    """Schoolfield developmental‑rate model.

    Returns a dimensionless rate factor; the formula follows the classic
    enzyme‑temperature relationship.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be positive Kelvin")
    # Convert gas constant to J mol⁻¹ K⁻¹ for consistency
    r = params.r_cal * 4.184
    # Helper for Arrhenius term
    def arrhenius(delta_h: float) -> float:
        return math.exp(-delta_h / (r * temp_k))
    # Main Schoolfield expression
    numerator = params.rho_25 * arrhenius(params.delta_h_activation)
    denominator = 1.0 + (arrhenius(params.delta_h_low) / arrhenius(params.delta_h_activation)) * \
                  math.exp(params.delta_h_low / r * (1.0 / params.t_low - 1.0 / temp_k)) + \
                  (arrhenius(params.delta_h_high) / arrhenius(params.delta_h_activation)) * \
                  math.exp(params.delta_h_high / r * (1.0 / params.t_high - 1.0 / temp_k))
    return numerator / denominator


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    multivector: GA2DMultivector
    count: int = 0
    total_reward: float = 0.0
    propensity: float = 1.0  # probability of being selected (raw, not normalized)

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0.0


def ucb_score(action: BanditAction, total_counts: int, alpha: float = 2.0) -> float:
    """Upper‑confidence‑bound score used for exploration–exploitation."""
    if action.count == 0:
        return float('inf')
    exploitation = action.avg_reward
    exploration = alpha * math.sqrt(math.log(total_counts) / action.count)
    return exploitation + exploration


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def fused_reward(
    action_mv: GA2DMultivector,
    target_mv: GA2DMultivector,
    temperature_c: float,
    sf_params: SchoolfieldParams,
    distance_scale: float = 1.0,
) -> float:
    """Compute the temperature‑aware reward linking GA distance with the Schoolfield model.

    R = exp( -α·d ) · developmental_rate(T)
    """
    d = action_mv.distance_to(target_mv)
    temp_k = c_to_k(temperature_c)
    rate = developmental_rate(temp_k, sf_params)
    return math.exp(-distance_scale * d) * rate


def select_action_ucb(actions: List[BanditAction], total_counts: int, temperature_c: float,
                     target_mv: GA2DMultivector, sf_params: SchoolfieldParams,
                     distance_scale: float = 1.0) -> BanditAction:
    """Select an action using a UCB score that incorporates the fused reward."""
    # Pre‑compute the fused reward for each arm (used as a proxy for expected reward)
    rewards = [
        fused_reward(a.multivector, target_mv, temperature_c, sf_params, distance_scale)
        for a in actions
    ]

    # Build temporary actions with the fused reward as expected_reward
    temp_actions = [
        replace(a, total_reward=a.total_reward + r, count=a.count + 1)
        for a, r in zip(actions, rewards)
    ]

    # Compute UCB scores on the temporary actions
    scores = [ucb_score(a, total_counts + 1) for a in temp_actions]
    best_idx = int(np.argmax(scores))
    return actions[best_idx]


def update_action(action: BanditAction, reward: float) -> BanditAction:
    """Return a new BanditAction with updated statistics."""
    return replace(
        action,
        count=action.count + 1,
        total_reward=action.total_reward + reward,
        propensity=action.propensity  # unchanged; could be re‑scaled later
    )


def allocate_workshare(models: List[str],
                       feature_dict: Dict[str, float],
                       day_of_week: int) -> List[Tuple[str, float]]:
    """Allocate a normalized share of total effort among models.

    The allocation uses a deterministic pseudo‑feature derived from the day of week
    to bias the distribution.  The output list sums to 1.0.
    """
    base = np.array([feature_dict.get(m, 0.1) for m in models], dtype=float)
    # Inject a simple sinusoidal bias based on the day (0=Mon … 6=Sun)
    bias = 0.5 * (1 + math.sin(2 * math.pi * day_of_week / 7))
    biased = base * (1 + bias)
    total = biased.sum()
    if total == 0:
        return [(m, 1.0 / len(models)) for m in models]
    return [(m, v / total) for m, v in zip(models, biased)]


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------


def _demo():
    # Define a small set of candidate actions (random multivectors)
    random.seed(0)
    actions = []
    for i in range(5):
        mv = GA2DMultivector(
            s=random.uniform(-1, 1),
            e1=random.uniform(-5, 5),
            e2=random.uniform(-5, 5),
            e12=random.uniform(-0.5, 0.5)
        )
        actions.append(BanditAction(action_id=f"a{i}", multivector=mv))

    # Target multivector (the "desired" solution)
    target = GA2DMultivector(s=0.0, e1=0.0, e2=0.0, e12=0.0)

    # Temperature (°C) and Schoolfield parameters
    temperature_c = 25.0
    sf_params = SchoolfieldParams()

    # Run a few bandit iterations
    total_counts = 0
    for step in range(10):
        chosen = select_action_ucb(
            actions,
            total_counts,
            temperature_c,
            target,
            sf_params,
            distance_scale=0.2
        )
        reward = fused_reward(chosen.multivector, target, temperature_c, sf_params, distance_scale=0.2)
        # Update the chosen action in the list (immutability -> replace)
        actions = [
            update_action(a, reward) if a.action_id == chosen.action_id else a
            for a in actions
        ]
        total_counts += 1
        print(f"Step {step+1:2d}: chose {chosen.action_id}, reward={reward:.4f}")

    # Work‑share demonstration
    models = ["model_x", "model_y", "model_z"]
    feature_vals = {"model_x": 0.8, "model_y": 0.5, "model_z": 0.3}
    today = datetime.datetime.utcnow().weekday()  # 0 = Monday
    allocation = allocate_workshare(models, feature_vals, today)
    print("\nWork‑share allocation:")
    for m, share in allocation:
        print(f"  {m}: {share:.3f}")


if __name__ == "__main__":
    _demo()