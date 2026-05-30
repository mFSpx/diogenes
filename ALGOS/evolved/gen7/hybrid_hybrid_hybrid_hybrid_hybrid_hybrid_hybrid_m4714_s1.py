# DARWIN HAMMER — match 4714, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1264_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_regret_engine_m136_s2.py (gen4)
# born: 2026-05-29T23:57:37Z

"""Hybrid algorithm merging geometric algebra (Parent B) with regret‑aware store dynamics (Parent A).

Mathematical bridge:
- Actions are encoded as basis blades of a Clifford algebra; the geometric product
  combines them into a multivector whose scalar (grade‑0) part represents the
  aggregated deterministic signal, while higher‑grade parts capture interaction
  (regret) structure.
- The distribution of absolute multivector coefficients is summarised by the
  Gini coefficient (Parent B).  This serves as a surrogate for the RLCT‑based
  information‑loss estimate from Parent A.
- The Gini coefficient modulates the deterministic target percentage used in
  the StoreState update (Parent A).  Thus the algebraic interaction of actions
  directly influences the bandit‑style work‑share allocation.

The module provides three core functions demonstrating this fusion:
1. `geometric_action_product` – builds a multivector from actions and returns its
   geometric product.
2. `regret_aware_update` – extracts scalar and distribution information from the
   multivector and updates a `StoreState`.
3. `allocate_workshare` – uses the updated store state to compute deterministic
   and probabilistic allocations for each action.
"""

import sys
import math
import random
import pathlib
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Iterable, FrozenSet

# ----------------------------------------------------------------------
# Parent A structures (store dynamics)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply deterministic inflow/outflow and return new level and delta."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded modulation derived from the last delta."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

# ----------------------------------------------------------------------
# Parent B structures (geometric algebra & regret)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic action with value, cost and optional risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                # duplicate index cancels (e_i ^ e_i = 0)
                lst.pop(j)
                lst.pop(j)  # second element shifts to j after first pop
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a: List[int], blade_b: List[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a: Dict[FrozenSet[int], float],
                     b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Full Clifford product `ab` for two multivectors."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coef_a in a.items():
        for blade_b, coef_b in b.items():
            blade, sign = _multiply_blades(list(blade_a), list(blade_b))
            result[blade] = result.get(blade, 0.0) + sign * coef_a * coef_b
    return result

def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return max(0.0, min(1.0, gini))

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def geometric_action_product(actions: List[MathAction]) -> Dict[FrozenSet[int], float]:
    """
    Encode each action as a grade‑1 blade with coefficient (value - cost)
    and return the geometric product of all encoded actions.
    """
    # Map each action to a unique basis index
    multivector: Dict[FrozenSet[int], float] = {}
    for idx, act in enumerate(actions):
        blade = frozenset({idx})
        coeff = act.expected_value - act.cost
        multivector[blade] = coeff
    # Iteratively multiply all blades
    product = {frozenset(): 1.0}  # scalar identity
    for blade, coeff in multivector.items():
        product = geometric_product(product, {blade: coeff})
    return product

def regret_aware_update(state: StoreState,
                        multivector: Dict[FrozenSet[int], float]) -> Tuple[float, float]:
    """
    Use the scalar part of the multivector as deterministic inflow.
    Use the Gini coefficient of absolute non‑scalar coefficients as a
    proxy for information loss; this scales the outflow.
    """
    # Scalar (grade‑0) part
    scalar = multivector.get(frozenset(), 0.0)

    # Gather magnitudes of higher‑grade components
    higher = [abs(v) for k, v in multivector.items() if k]
    gini = gini_coefficient(higher)

    # Deterministic inflow is the scalar, outflow is modulated by gini
    inflow = [scalar]
    outflow = [state.dance * gini]  # larger gini → larger outflow (more regret)

    level, delta = state.update(inflow, outflow)
    return level, delta

def allocate_workshare(state: StoreState,
                       actions: List[MathAction],
                       multivector: Dict[FrozenSet[int], float]) -> List[Tuple[MathAction, float]]:
    """
    Allocate a workshare percentage to each action.
    Deterministic share = state.dance * (1 - gini).
    Remaining share is distributed proportionally to the absolute coefficients
    of the corresponding grade‑1 blades.
    """
    # Compute Gini over all non‑scalar coefficients
    coeffs = [abs(v) for k, v in multivector.items() if k]
    gini = gini_coefficient(coeffs)

    deterministic_share = state.dance * (1.0 - gini)
    remaining_share = max(0.0, 1.0 - deterministic_share)

    # Extract grade‑1 coefficients (one per action)
    grade1 = {}
    for idx, act in enumerate(actions):
        blade = frozenset({idx})
        grade1[act.id] = abs(multivector.get(blade, 0.0))

    total = sum(grade1.values()) or 1.0
    allocations: List[Tuple[MathAction, float]] = []
    for act in actions:
        prob = (grade1[act.id] / total) * remaining_share
        allocations.append((act, deterministic_share + prob))
    return allocations

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small set of actions
    actions = [
        MathAction(id="A", expected_value=5.0, cost=1.0),
        MathAction(id="B", expected_value=3.0, cost=0.5),
        MathAction(id="C", expected_value=4.0, cost=2.0),
    ]

    # Build geometric product
    mv = geometric_action_product(actions)

    # Initialise store state
    state = StoreState(level=0.0, alpha=0.8, beta=0.5, dt=1.0,
                       base=1.0, gain=2.0, limit=5.0)

    # Perform regret‑aware update
    level, delta = regret_aware_update(state, mv)
    print(f"Store level after update: {level:.3f}, delta: {delta:.3f}")

    # Allocate workshare
    allocations = allocate_workshare(state, actions, mv)
    for act, share in allocations:
        print(f"Action {act.id}: allocated share = {share:.3%}")

    # Simple sanity check: shares sum to ≤ 1
    total_share = sum(s for _, s in allocations)
    print(f"Total allocated share: {total_share:.3%}")