# DARWIN HAMMER — match 4624, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1683_s2.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_variational_free_ene_m56_s0.py (gen2)
# born: 2026-05-29T23:56:59Z

"""Hybrid Strike-Bandit Free Energy Algorithm

Parents:
- hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1683_s2.py (physics‑based strike integration,
  hashing utilities, endpoint circuit breaker)
- hybrid_hybrid_bandit_router_variational_free_ene_m56_s0.py (multi‑armed bandit router
  combined with a Variational Free Energy (VFE) framework)

Mathematical Bridge:
The bridge is the *expected reward* of a bandit action, which is derived from the
physics simulation of a strike.  Each BanditAction proposes a peak force (encoded
in its `propensity`).  The `integrate_strike` routine computes the resulting
velocity, distance and peak velocity.  This physical outcome is turned into a
scalar reward (here the travelled distance).  The VFE formalism supplies a
regularisation term – the KL divergence between the current posterior over
action propensities (`q`) and a fixed Gaussian prior (`p`).  The free‑energy
objective  

    F = -E_q[reward] + KL(q‖p)

is minimised by updating the bandit policy: actions that generate larger
distances (higher reward) receive higher propensities, while the KL term prevents
over‑confidence.  The three core functions below demonstrate this fused workflow. 
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A (physics)
# ----------------------------------------------------------------------
Node = object  # placeholder for hashable node type
Graph = Dict[Node, set[Node]]

@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float

def compute_dhash(values: List[float]) -> int:
    """Directional hash – 1 bit per adjacent comparison."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits

def compute_phash(values: List[float]) -> int:
    """Population hash – 1 bit per value relative to the mean (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer bit‑patterns."""
    return (a ^ b).bit_count()

def integrate_strike(
    force_series: List[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> StrikeState:
    """Simple 1‑D projectile under linear drag."""
    if dt <= 0:
        raise ValueError("dt must be greater than zero")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)

def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse with given peak amplitude."""
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]

# ----------------------------------------------------------------------
# Core structures from Parent B (bandit + VFE)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # raw (unnormalised) preference
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def kl_gaussian(
    mu_q: float,
    sigma_q: float,
    mu_p: float = 0.0,
    sigma_p: float = 1.0,
) -> float:
    """KL divergence KL(q‖p) for univariate Gaussians."""
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    return 0.5 * (
        (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
        + math.log(var_p / var_q)
    )

def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax."""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

# ----------------------------------------------------------------------
# Hybrid Functions (the new fused topology)
# ----------------------------------------------------------------------
def hybrid_select_action(
    actions: List[BanditAction],
    temperature: float = 1.0,
) -> BanditAction:
    """
    Choose an action using a Boltzmann distribution over propensities.
    The temperature controls exploration; lower values make the selection
    more deterministic.
    """
    if not actions:
        raise ValueError("No actions to select from")
    logits = [a.propensity / max(temperature, 1e-8) for a in actions]
    probs = softmax(logits)
    chosen_id = random.choices([a.action_id for a in actions], weights=probs, k=1)[0]
    for a in actions:
        if a.action_id == chosen_id:
            return a
    # Fallback (should never happen)
    return actions[0]

def hybrid_evaluate_strike(
    action: BanditAction,
    dt: float = 0.01,
    steps: int = 20,
    m_head: float = 0.5,
) -> Tuple[float, StrikeState]:
    """
    Run the physics simulation using the action's propensity as the peak force.
    Returns a scalar reward (distance travelled) and the full StrikeState.
    """
    # Scale propensity to a plausible force magnitude
    peak_force = max(0.0, action.propensity) * 10.0  # arbitrary scaling
    force_series = pulse_force(peak_force, steps)
    strike = integrate_strike(force_series, dt, m_head)
    reward = strike.distance  # larger distance → higher reward
    return reward, strike

def hybrid_update_policy(
    actions: List[BanditAction],
    selected: BanditAction,
    reward: float,
    prior_mu: float = 0.0,
    prior_sigma: float = 5.0,
    beta: float = 0.1,
) -> List[BanditAction]:
    """
    Update the policy using a Variational Free Energy objective:
        F = -E_q[reward] + beta * KL(q‖p)
    The posterior q is approximated by a Gaussian whose mean is the
    propensity of the selected action and a small variance.
    Propensities are nudged in the direction of the reward while the KL term
    penalises large deviations from the prior.
    """
    # Record the raw reward in the global policy store
    update = BanditUpdate(
        context_id="global",
        action_id=selected.action_id,
        reward=reward,
        propensity=selected.propensity,
    )
    update_policy([update])

    # Posterior approximation (mean = selected propensity, sigma = 1.0)
    mu_q = selected.propensity
    sigma_q = 1.0

    # KL regularisation
    kl = kl_gaussian(mu_q, sigma_q, prior_mu, prior_sigma)

    # Free‑energy gradient step (simple proportional update)
    lr = 0.05  # learning rate
    delta = lr * (reward - beta * kl)

    # Produce a new list of actions with updated propensities
    new_actions = []
    for a in actions:
        if a.action_id == selected.action_id:
            new_propensity = max(0.0, a.propensity + delta)
        else:
            # Slight decay for non‑selected actions to encourage exploration
            new_propensity = max(0.0, a.propensity * (1.0 - lr * 0.1))
        new_actions.append(
            BanditAction(
                action_id=a.action_id,
                propensity=new_propensity,
                algorithm=a.algorithm,
            )
        )
    return new_actions

# ----------------------------------------------------------------------
# Demonstration / Smoke Test
# ----------------------------------------------------------------------
def _demo() -> None:
    # Initialise three candidate actions with different initial propensities
    actions = [
        BanditAction(action_id="A", propensity=1.0),
        BanditAction(action_id="B", propensity=2.0),
        BanditAction(action_id="C", propensity=0.5),
    ]

    reset_policy()
    print("Starting hybrid optimisation (10 iterations)")
    for epoch in range(10):
        # 1️⃣ Select
        chosen = hybrid_select_action(actions, temperature=0.5)

        # 2️⃣ Evaluate via physics
        reward, strike = hybrid_evaluate_strike(chosen)

        # 3️⃣ Update policy with VFE regularisation
        actions = hybrid_update_policy(actions, chosen, reward)

        # Logging
        dh = compute_dhash([strike.velocity, strike.distance, strike.peak_velocity])
        ph = compute_phash([strike.velocity, strike.distance, strike.peak_velocity])
        print(
            f"Iter {epoch+1:02d} | Chosen {chosen.action_id} | "
            f"Reward {reward:.4f} | Propensities {[a.propensity for a in actions]} | "
            f"Dhash {dh:#06x} | Phash {ph:#06x}"
        )

if __name__ == "__main__":
    _demo()