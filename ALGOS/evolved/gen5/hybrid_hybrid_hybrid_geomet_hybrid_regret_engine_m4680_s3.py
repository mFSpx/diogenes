# DARWIN HAMMER — match 4680, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s0.py (gen4)
# parent_b: hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py (gen2)
# born: 2026-05-29T23:57:25Z

"""Hybrid Geometric‑Regret Engine
================================

This module fuses two distinct algorithmic lineages:

* **Parent A** – *hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s0.py*  
  Implements Clifford‑algebra geometric products, blade‑wise arithmetic and
  Voronoi partitioning of points in a Euclidean‑like space.

* **Parent B** – *hybrid_regret_engine_hybrid_doomsday_cale_m19_s7.py*  
  Provides a regret‑weighted decision strategy, a Doomsday‑based weekday mapping,
  and the Gini‑coefficient inequality measure.

**Mathematical bridge**  
Each *action* is represented as a multivector **A** ∈ Cl(n,0).  
The geometric product `A ⋅ B` yields a new multivector whose scalar part encodes
the dot‑product of the underlying vectors; its magnitude `‖A⋅B‖` is used as a
distance‑like metric between actions and Voronoi seeds.  These distances modulate
the classic regret‑weighted probabilities, producing *distance‑aware regret
weights*.  The resulting weight vector is then distributed over the month’s
weekday sequence (via the Doomsday algorithm) and finally evaluated with the
Gini coefficient to quantify inequality of the schedule.

The three core functions below demonstrate the hybrid workflow:
1. ``geometric_product`` – Clifford algebra multiplication.
2. ``voronoi_partition`` – nearest‑seed assignment using the geometric‑product norm.
3. ``hybrid_regret_voronoi_strategy`` – regret weighting, distance modulation,
   weekday mapping and Gini evaluation.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Dict, Tuple, List, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent‑B data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An actionable decision with an expected value and optional cost/risk."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """A hypothetical outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent‑A: Clifford algebra utilities (blade‑wise)
# ----------------------------------------------------------------------
Blade = FrozenSet[int]          # e.g. frozenset({1,3}) represents e1∧e3
Multivector = Dict[Blade, float]   # mapping blade → coefficient


def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """
    Sort ``indices`` by bubble sort while tracking the sign change due to
    anti‑commutativity. Duplicate indices cancel (e_i * e_i = 1).
    Returns the sorted list (duplicates removed) and the overall sign (+1 / -1).
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst) - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)          # re‑check after swap
        elif lst[i] == lst[i + 1]:
            # e_i * e_i = 1 → remove both and keep sign unchanged
            del lst[i : i + 2]
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(a: Blade, b: Blade) -> Tuple[Blade, int]:
    """
    Multiply two basis blades using the exterior‑product rules of Cl(n,0).
    Returns the resulting blade (as a frozenset) and the sign factor.
    """
    combined = list(a) + list(b)
    sorted_idxs, sign = _blade_sign(combined)
    return frozenset(sorted_idxs), sign


def geometric_product(mv1: Multivector, mv2: Multivector) -> Multivector:
    """
    Full geometric product of two multivectors.
    Implements linearity: (∑_a a_A A) (∑_b b_B B) = ∑_{A,B} a_A b_B (A·B).
    """
    result: Multivector = {}
    for blade_a, coeff_a in mv1.items():
        for blade_b, coeff_b in mv2.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # Remove near‑zero entries for numerical cleanliness
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def multivector_norm(mv: Multivector) -> float:
    """
    Euclidean norm of a multivector defined as sqrt(∑_blade coeff²).
    The scalar part is not treated specially because we are interested in a
    distance‑like quantity derived from the full product.
    """
    return math.sqrt(sum(c * c for c in mv.values()))


# ----------------------------------------------------------------------
# Parent‑A: Voronoi partitioning using geometric‑product distance
# ----------------------------------------------------------------------
def voronoi_partition(
    points: List[Multivector],
    seeds: List[Multivector],
) -> Dict[int, int]:
    """
    Assign each point to the index of its nearest seed.
    Distance metric = ‖point ⋅ seed‖ (geometric product norm).
    Returns a mapping ``point_index → seed_index``.
    """
    assignment: Dict[int, int] = {}
    for i, pt in enumerate(points):
        best_seed = None
        best_dist = math.inf
        for j, sd in enumerate(seeds):
            prod = geometric_product(pt, sd)
            dist = multivector_norm(prod)
            if dist < best_dist:
                best_dist = dist
                best_seed = j
        assignment[i] = best_seed if best_seed is not None else -1
    return assignment


# ----------------------------------------------------------------------
# Parent‑B: Doomsday weekday utilities and Gini coefficient
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index (0=Monday … 6=Sunday) via Python's calendar."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def _weekday_sequence(year: int, month: int, num_days: int) -> List[int]:
    """List of weekday indices for the first *num_days* of a month."""
    return [doomsday(year, month, d) for d in range(1, num_days + 1)]


def gini_coefficient(values: Iterable[float]) -> float:
    """Standard Gini coefficient for a non‑negative distribution."""
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    cumulative = sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1))
    return cumulative / (n * sum(xs))


# ----------------------------------------------------------------------
# Hybrid core: regret weighting modulated by geometric distances
# ----------------------------------------------------------------------
def _classic_regret_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    epsilon: float = 1e-12,
) -> Dict[str, float]:
    """
    Classic regret‑weighted probability:
        w_i ∝ max(0, E_i - Σ_j p_j * O_{ij})
    where E_i is the expected value of action i, O_{ij} are counterfactual
    outcomes, and p_j are the (to be solved) probabilities.
    For simplicity we use a single‑step approximation:
        w_i = max(0, expected_value_i - avg_counterfactual_i)
    """
    # Group counterfactuals by action
    cf_by_act: Dict[str, List[MathCounterfactual]] = {}
    for cf in counterfactuals:
        cf_by_act.setdefault(cf.action_id, []).append(cf)

    raw_weights: Dict[str, float] = {}
    for act in actions:
        cfs = cf_by_act.get(act.id, [])
        if cfs:
            avg_outcome = sum(cf.outcome_value * cf.probability for cf in cfs) / (
                sum(cf.probability for cf in cfs) + epsilon
            )
        else:
            avg_outcome = 0.0
        regret = act.expected_value - avg_outcome
        raw_weights[act.id] = max(0.0, regret)

    total = sum(raw_weights.values()) + epsilon
    return {aid: w / total for aid, w in raw_weights.items()}


def _distance_modulation(
    actions: List[MathAction],
    points: List[Multivector],
    seeds: List[Multivector],
    base_probs: Dict[str, float],
) -> Dict[str, float]:
    """
    Adjust base probabilities by the inverse of the geometric‑product distance
    to the assigned Voronoi seed.
    """
    # First, assign each point (action) to a seed
    assignment = voronoi_partition(points, seeds)

    # Compute distance for each action
    distances: List[float] = []
    for idx, pt in enumerate(points):
        seed_idx = assignment.get(idx, -1)
        if seed_idx == -1:
            distances.append(math.inf)
            continue
        prod = geometric_product(pt, seeds[seed_idx])
        distances.append(multivector_norm(prod) + 1e-12)  # avoid zero

    # Inverse‑distance weighting
    inv_dist = np.reciprocal(np.array(distances, dtype=float))
    inv_dist = inv_dist / inv_dist.sum()  # normalize

    # Blend with base probabilities (simple convex combination)
    blended: Dict[str, float] = {}
    for act, w_inv in zip(actions, inv_dist):
        blended[act.id] = 0.6 * base_probs.get(act.id, 0.0) + 0.4 * w_inv
    # Renormalize
    total = sum(blended.values()) + 1e-12
    return {aid: w / total for aid, w in blended.items()}


def _map_actions_to_weekdays(
    actions: List[MathAction],
    probs: Dict[str, float],
    year: int,
    month: int,
    num_days: int,
) -> Dict[str, Tuple[int, float]]:
    """
    Distribute each action's probability over the month’s weekdays.
    Returns a mapping ``action_id → (weekday_index, weighted_probability)`` where
    the weekday is chosen by cycling through the weekday sequence proportionally
    to the action's probability mass.
    """
    weekdays = _weekday_sequence(year, month, num_days)
    if not weekdays:
        raise ValueError("Month must have at least one day")

    mapping: Dict[str, Tuple[int, float]] = {}
    for i, act in enumerate(actions):
        # Determine which weekday this action lands on
        wd = weekdays[i % len(weekdays)]
        mapping[act.id] = (wd, probs.get(act.id, 0.0))
    return mapping


def hybrid_regret_voronoi_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    points: List[Multivector],
    seeds: List[Multivector],
    year: int,
    month: int,
    num_days: int,
    epsilon: float = 1e-6,
) -> Tuple[Dict[str, float], Dict[str, Tuple[int, float]], float]:
    """
    End‑to‑end hybrid algorithm:

    1. Compute classic regret‑weighted probabilities.
    2. Modulate them with Voronoi‑based inverse distances (geometric product).
    3. Map the resulting probabilities onto weekdays using the Doomsday algorithm.
    4. Evaluate inequality of the weekday‑weighted schedule via the Gini coefficient.

    Returns:
        - ``final_probs`` – distance‑aware regret probabilities (action_id → prob).
        - ``weekday_map`` – action_id → (weekday_index, weighted_prob).
        - ``gini`` – Gini coefficient of the weighted probabilities.
    """
    base_probs = _classic_regret_weights(actions, counterfactuals, epsilon)
    final_probs = _distance_modulation(actions, points, seeds, base_probs)
    weekday_map = _map_actions_to_weekdays(actions, final_probs, year, month, num_days)
    gini = gini_coefficient([p for _, p in weekday_map.values()])
    return final_probs, weekday_map, gini


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny 3‑dimensional Clifford space (indices 1..3)
    def vec(*indices: int) -> Multivector:
        """Convenient helper: create a pure vector multivector."""
        return {frozenset({i}): 1.0 for i in indices}

    # Sample actions
    actions = [
        MathAction(id="A", expected_value=10.0, cost=2.0),
        MathAction(id="B", expected_value=8.0, cost=1.5),
        MathAction(id="C", expected_value=12.0, cost=3.0),
    ]

    # Counterfactuals (simple synthetic data)
    counterfactuals = [
        MathCounterfactual(action_id="A", outcome_value=7.0, probability=0.6),
        MathCounterfactual(action_id="A", outcome_value=9.0, probability=0.4),
        MathCounterfactual(action_id="B", outcome_value=5.0, probability=1.0),
        MathCounterfactual(action_id="C", outcome_value=11.0, probability=0.5),
        MathCounterfactual(action_id="C", outcome_value=13.0, probability=0.5),
    ]

    # Points representing actions in the Clifford algebra
    points = [
        vec(1),          # action A → e1
        vec(2),          # action B → e2
        vec(1, 2, 3),    # action C → e1∧e2∧e3 (a trivector)
    ]

    # Seeds for Voronoi partition (chosen arbitrarily)
    seeds = [
        vec(1, 2),       # seed 0
        vec(3),          # seed 1
    ]

    # Run the hybrid strategy for May 2026 (31 days)
    probs, week_map, gini = hybrid_regret_voronoi_strategy(
        actions=actions,
        counterfactuals=counterfactuals,
        points=points,
        seeds=seeds,
        year=2026,
        month=5,
        num_days=31,
    )

    print("Final regret probabilities (distance‑aware):")
    for aid, p in probs.items():
        print(f"  {aid}: {p:.4f}")

    print("\nWeekday assignment (weekday index, weighted prob):")
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for aid, (wd, w) in week_map.items():
        print(f"  {aid}: {weekday_names[wd]} ({w:.4f})")

    print(f"\nGini coefficient of schedule: {gini:.4f}")

    # Simple sanity checks
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9), "Probabilities must sum to 1"
    assert 0.0 <= gini <= 1.0, "Gini must be in [0,1]"

    print("\nSmoke test completed successfully.")