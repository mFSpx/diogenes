# DARWIN HAMMER — match 5591, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s5.py (gen3)
# parent_b: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s1.py (gen4)
# born: 2026-05-30T00:03:12Z

"""HybridModelBanditPool
Combines:
- Parent A: ModelPool with RAM ceiling, linear “rectified flow” allocation and morphology‑derived priority f(endpoint, model) = p(m)*(1 – r/R_max).
- Parent B: Multi‑armed bandit with fold‑change detection, pheromone infotaxis and hybrid store factor.

Mathematical bridge
------------------
For each (endpoint, model) pair we treat the pair as a *bandit action*.
The morphology‑derived priority f acts as a *pheromone* concentration,
while the bandit reward (derived from historic selections) supplies the
expected payoff.  The hybrid score used for allocation is

    S = f(endpoint, model) · ( reward(action) + store_factor )

where
    store_factor = log_count_ratio * count(action)

and log_count_ratio is a global exploration/exploitation knob.
The target RAM allocation for an endpoint is proportional to its S
relative to the sum of all S, and the actual allocation follows the
linear interpolant

    a(α) = (1‑α)·a₀ + α·a₁ ,   α∈[0,1]

Thus the system fuses the linear flow schedule of Parent A with the
adaptive bandit decision logic of Parent B into a single unified
resource‑allocation engine.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Model and Endpoint definitions
# ----------------------------------------------------------------------
class Model:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str = "standard"):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class Endpoint:
    """Morphology‑aware endpoint."""
    def __init__(self, identifier: str, righting_time: float):
        """
        righting_time: raw time to self‑right (seconds).  Larger values imply
        lower health, therefore priority is normalized as p = 1 / (1+righting_time).
        """
        self.id = identifier
        self.righting_time = righting_time

    @property
    def priority(self) -> float:
        """Normalized priority p(m) ∈ [0,1]."""
        return 1.0 / (1.0 + self.righting_time)

# ----------------------------------------------------------------------
# Parent B – Bandit data structures and helpers
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str          # convention: f"{endpoint.id}|{model.name}"
    propensity: float      # not used in this hybrid but kept for compatibility
    expected_reward: float # placeholder, overridden by dynamic reward
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}   # action_id -> [total_reward, count]

def reset_policy() -> None:
    """Reset the bandit policy."""
    _POLICY.clear()

def _reward(action_id: str) -> float:
    """Mean reward observed for an action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action_id: str) -> float:
    """Number of times an action has been selected."""
    return _POLICY.get(action_id, [0.0, 0.0])[1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    """Store factor = log_count_ratio * count."""
    return log_count_ratio * count

def hybrid_select_action(actions: List[BanditAction], log_count_ratio: float) -> str:
    """
    Choose the action with maximal (store_factor + reward).
    Returns the selected action_id.
    """
    best_action_id = None
    best_value = -math.inf
    for act in actions:
        cnt = _count(act.action_id)
        value = _hybrid_store_factor(act.action_id, cnt, log_count_ratio) + _reward(act.action_id)
        if value > best_value:
            best_value = value
            best_action_id = act.action_id
    return best_action_id

def update_policy(action_id: str, reward: float) -> None:
    """Record a new reward for the given action."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, n + 1]

# ----------------------------------------------------------------------
# Hybrid core utilities
# ----------------------------------------------------------------------
def linear_interpolant(a0: Dict[str, float],
                       a1: Dict[str, float],
                       alpha: float) -> Dict[str, float]:
    """
    Linear interpolation between two allocation dictionaries.
    Missing keys are treated as zero.
    """
    keys = set(a0) | set(a1)
    result = {}
    for k in keys:
        v0 = a0.get(k, 0.0)
        v1 = a1.get(k, 0.0)
        result[k] = (1.0 - alpha) * v0 + alpha * v1
    return result

def hybrid_priority(endpoint: Endpoint, model: Model, ram_ceiling_mb: int) -> float:
    """
    Compute the morphology‑derived priority field f(endpoint, model).
    """
    r = model.ram_mb
    R = ram_ceiling_mb
    return endpoint.priority * (1.0 - r / R)

def compute_hybrid_score(endpoint: Endpoint,
                         model: Model,
                         ram_ceiling_mb: int,
                         log_count_ratio: float) -> float:
    """
    Hybrid score S = f * (reward + store_factor)
    where reward and store_factor are bandit quantities for the
    action identified by f"{endpoint.id}|{model.name}".
    """
    action_id = f"{endpoint.id}|{model.name}"
    f_val = hybrid_priority(endpoint, model, ram_ceiling_mb)
    reward = _reward(action_id)
    count = _count(action_id)
    store = _hybrid_store_factor(action_id, count, log_count_ratio)
    return f_val * (reward + store + 1e-12)   # tiny epsilon to avoid zero

def allocate_resources(endpoints: List[Endpoint],
                       models: List[Model],
                       current_alloc: Dict[str, float],
                       ram_ceiling_mb: int,
                       log_count_ratio: float,
                       alpha: float) -> Tuple[Dict[str, float], BanditAction]:
    """
    Produce a new RAM allocation following the hybrid schedule.
    Returns (new_allocation, selected_action) where selected_action
    is the BanditAction that received the highest hybrid score.
    """
    # Build all possible (endpoint, model) actions
    actions: List[BanditAction] = []
    scores: Dict[str, float] = {}
    for ep in endpoints:
        for md in models:
            aid = f"{ep.id}|{md.name}"
            score = compute_hybrid_score(ep, md, ram_ceiling_mb, log_count_ratio)
            scores[aid] = score
            actions.append(BanditAction(
                action_id=aid,
                propensity=0.0,
                expected_reward=_reward(aid),
                confidence_bound=0.0,
                algorithm="Hybrid"
            ))

    # Select action via bandit logic (store_factor + reward)
    selected_id = hybrid_select_action(actions, log_count_ratio)

    # Derive target allocation proportional to hybrid scores
    total_score = sum(scores.values()) + 1e-12
    target_alloc: Dict[str, float] = {}
    for aid, sc in scores.items():
        proportion = sc / total_score
        target_alloc[aid] = proportion * ram_ceiling_mb

    # Interpolate between current and target
    new_alloc = linear_interpolant(current_alloc, target_alloc, alpha)

    # Update policy with a synthetic reward equal to the selected score
    update_policy(selected_id, scores[selected_id])

    # Return the chosen BanditAction object
    selected_action = next(act for act in actions if act.action_id == selected_id)
    return new_alloc, selected_action

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any previous state
    reset_policy()

    # Define a tiny RAM ceiling for quick testing
    RAM_CEILING = 6000

    # Create endpoints with differing righting times
    eps = [
        Endpoint("ep_A", righting_time=0.5),   # higher priority
        Endpoint("ep_B", righting_time=2.0)    # lower priority
    ]

    # Create models of varying size
    mods = [
        Model("model_X", ram_mb=1200),
        Model("model_Y", ram_mb=2500)
    ]

    # Start with zero allocation
    current_allocation: Dict[str, float] = {}

    # Hybrid hyper‑parameter
    LOG_COUNT_RATIO = 0.1
    ALPHA = 0.5   # halfway between current and target

    # Perform a single allocation step
    new_alloc, chosen = allocate_resources(
        endpoints=eps,
        models=mods,
        current_alloc=current_allocation,
        ram_ceiling_mb=RAM_CEILING,
        log_count_ratio=LOG_COUNT_RATIO,
        alpha=ALPHA
    )

    # Print results – these prints are allowed in the smoke test
    print("New RAM allocation (MB per action):")
    for aid, mb in new_alloc.items():
        print(f"  {aid}: {mb:.2f}")

    print("\nChosen action by hybrid bandit:")
    print(f"  Action ID: {chosen.action_id}")
    print(f"  Expected reward (pre‑update): {chosen.expected_reward:.4f}")

    # Verify that total allocation does not exceed ceiling (within small epsilon)
    total_used = sum(new_alloc.values())
    assert total_used <= RAM_CEILING * 1.001, "Allocation exceeds RAM ceiling!"

    print("\nSmoke test completed successfully.")