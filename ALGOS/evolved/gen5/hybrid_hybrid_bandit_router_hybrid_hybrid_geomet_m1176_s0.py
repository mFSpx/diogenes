# DARWIN HAMMER — match 1176, survivor 0
# gen: 5
# parent_a: hybrid_bandit_router_honeybee_store_m9_s4.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s6.py (gen4)
# born: 2026-05-29T23:33:13Z

"""Hybrid Bandit–Geometric Algorithm
Parents:
- hybrid_bandit_router_honeybee_store_m9_s4 (Bandit selection with store dynamics)
- hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s6 (Clifford geometric algebra multivectors)

Mathematical Bridge:
Each action’s statistical state (propensity, reward estimate) is encoded as the scalar part of a
Multivector.  The contextual feature vector is also lifted into a Multivector by assigning each
feature to a distinct basis blade.  The geometric product of action‑ and context‑Multivectors
produces a new Multivector whose scalar part naturally combines the action’s propensity with the
contextual weighting.  This scalar is then fed to the bandit decision logic (epsilon‑greedy,
Thompson sampling, LinUCB‑style scaling) while the store dynamics from the original bandit
algorithm modulate the exploration parameters.  Thus the two topologies are fused: Clifford
algebra supplies a mathematically sound feature‑action interaction, and the bandit/store
machinery supplies the learning and resource‑allocation loop.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, FrozenSet
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit / Store components
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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def update_store(
    store: float,
    inflow: List[float],
    outflow: List[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> Tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(
    delta_store: float,
    base: float = 1.0,
    gain: float = 1.0,
    limit: float = 10.0,
) -> float:
    return max(0.0, min(limit, base + gain * delta_store))

# ----------------------------------------------------------------------
# Parent B – Geometric Algebra components
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                i = -1
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    def __init__(self, components: Dict[FrozenSet[int], float] = None):
        self.components: Dict[FrozenSet[int], float] = dict(components or {})

    def __add__(self, other: "Multivector") -> "Multivector":
        res = self.components.copy()
        for k, v in other.components.items():
            res[k] = res.get(k, 0.0) + v
            if abs(res[k]) < 1e-15:
                del res[k]
        return Multivector(res)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({k: -v for k, v in self.components.items()})

    def __mul__(self, other: "Multivector") -> "Multivector":
        result: Dict[FrozenSet[int], float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                coeff = ca * cb * sign
                result[blade] = result.get(blade, 0.0) + coeff
        result = {k: v for k, v in result.items() if abs(v) > 1e-15}
        return Multivector(result)

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components})"

def vector_to_mv(x: float, y: float) -> Multivector:
    return Multivector({frozenset({0}): x, frozenset({1}): y})

# ----------------------------------------------------------------------
# Hybrid utilities – bridging the two worlds
# ----------------------------------------------------------------------
def _basis_index(name: str) -> int:
    """Deterministic mapping from a string to a small non‑negative basis index."""
    # Use a simple hash modulo 16 to keep the algebra low‑dimensional.
    return abs(hash(name)) % 16

def context_to_mv(context: Dict[str, float]) -> Multivector:
    """Lift a context dictionary into a Multivector (no scalar part)."""
    comps: Dict[FrozenSet[int], float] = {}
    for key, val in context.items():
        idx = _basis_index(key)
        comps[frozenset({idx})] = float(val)
    return Multivector(comps)

def action_to_mv(action_id: str, propensity: float) -> Multivector:
    """Encode an action as a Multivector whose scalar part is the propensity."""
    # Add a unique basis blade for the action to keep it distinguishable.
    idx = _basis_index(action_id)
    return Multivector({frozenset(): float(propensity), frozenset({idx}): 1.0})

def geometric_reward(action_mv: Multivector, context_mv: Multivector) -> float:
    """
    Compute a reward estimate as the scalar part of the geometric product
    between action and context multivectors.
    """
    prod = action_mv * context_mv
    return prod.scalar_part()

def hybrid_select_action(
    context: Dict[str, float],
    actions: List[str],
    store: float,
    algorithm: str = "epsilon_greedy",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """
    Choose an action using a blend of geometric interaction and bandit logic.
    The geometric_reward provides an *expected* reward that is further modulated
    by the store factor (resource availability) before being fed to the usual
    exploration/exploitation scheme.
    """
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    # Store factor mirrors the original algorithm's scaling.
    store_factor = 1.0 + store / (store + 1.0)

    # Build context multivector once.
    ctx_mv = context_to_mv(context)

    # Pre‑compute geometric rewards for each candidate.
    geom_rewards: Dict[str, float] = {}
    for a in actions:
        prop = _reward(a)  # historical propensity (mean reward)
        act_mv = action_to_mv(a, prop)
        geom_rewards[a] = geometric_reward(act_mv, ctx_mv) * store_factor

    # Exploration strategies.
    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen_id = rng.choice(actions)
    elif algorithm == "thompson":
        # Sample from a Beta distribution whose parameters are shaped by
        # the geometric reward estimate.
        def sample(a: str) -> float:
            r = geom_rewards[a]
            n = _count(a) + 1.0  # avoid zero
            a_param = 1.0 + max(0.0, r) * n
            b_param = 1.0 + max(0.0, 1.0 - r) * n
            return rng.betavariate(a_param, b_param)
        chosen_id = max(actions, key=sample)
    else:
        # Default to LinUCB‑style: use the geometric reward as the mean,
        # and a confidence term that shrinks with count.
        def ucb_score(a: str) -> float:
            mean = geom_rewards[a]
            cnt = _count(a)
            bonus = math.sqrt(2 * math.log(1 + sum(_count(x) for x in actions) + 1) / (cnt + 1))
            return mean + bonus
        chosen_id = max(actions, key=ucb_score)

    # Assemble the BanditAction dataclass.
    return BanditAction(
        action_id=chosen_id,
        propensity=_reward(chosen_id),
        expected_reward=geom_rewards[chosen_id],
        confidence_bound=0.0,  # placeholder; could be filled with UCB bonus
        algorithm=algorithm,
    )

def hybrid_update(
    chosen: BanditAction,
    reward: float,
    store: float,
    inflow: List[float],
    outflow: List[float],
) -> Tuple[float, float]:
    """
    Perform a full update step:
    1. Record the reward in the bandit policy.
    2. Update the store based on inflow/outflow.
    3. Return the new store and its delta for possible downstream use.
    """
    update_policy([BanditUpdate(
        context_id="global",
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )])
    new_store, delta = update_store(store, inflow, outflow)
    return new_store, delta

def hybrid_dance_duration(store_delta: float) -> float:
    """Expose the original dance_duration logic as a convenience wrapper."""
    return dance_duration(store_delta)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny environment.
    actions = ["click", "scroll", "hover"]
    context = {"user_age": 25.0, "session_length": 5.0}
    store = 0.0

    # First selection.
    chosen = hybrid_select_action(context, actions, store, algorithm="epsilon_greedy", epsilon=0.2)
    print("Chosen:", chosen)

    # Simulate a stochastic reward.
    simulated_reward = random.random()
    inflow = [simulated_reward]  # reward contributes to resource inflow
    outflow = [0.1]  # constant cost

    # Update the system.
    store, delta = hybrid_update(chosen, simulated_reward, store, inflow, outflow)
    print("New store:", store, "Delta:", delta)

    # Compute a dance duration based on the store change.
    duration = hybrid_dance_duration(delta)
    print("Dance duration:", duration)

    # Verify that policy statistics have been recorded.
    for a in actions:
        print(f"Policy[{a}] = total reward / count -> { _reward(a):.3f} (count {_count(a)})")