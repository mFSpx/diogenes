# DARWIN HAMMER — match 183, survivor 1
# gen: 3
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s3.py (gen1)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_krampus_brain_m6_s3.py (gen2)
# born: 2026-05-29T23:26:00Z

"""Hybrid Bandit‑Voronoi‑Geometric Algebra Module.

This module fuses two previously independent algorithms:

* **Parent A – Bandit Router**: a contextual multi‑armed bandit that stores
  cumulative rewards per action and provides empirical means, propensities and
  confidence bounds.

* **Parent B – Voronoi + Geometric Algebra**: a planar Voronoi partitioner that
  assigns points to the nearest seed and a lightweight implementation of
  multivectors (geometric algebra) with blade multiplication.

**Mathematical bridge**

Each Bandit action is represented as a 2‑D point  
`p = (expected_reward, confidence_bound)`.  
These points become Voronoi seeds; arbitrary context points (e.g. sensor
measurements) are assigned to the nearest action‑seed.  The resulting
partition gives a natural “region of influence’’ for every action.

From the partition we build a **policy multivector**:

* For every action *a* we create a blade `e_i` where *i* is the index of the
  action in the action list.
* The coefficient of that blade is the cumulative reward of *a* (the
  empirical mean multiplied by the number of pulls).  
* The geometric product of all action blades yields a single multivector that
  encodes the joint policy state.

The three exported functions demonstrate this hybrid workflow:
`bandit_to_point`, `assign_contexts_to_actions` and `policy_multivector`.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit core (adapted)
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


# Global policy storage: action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Clear all stored reward statistics."""
    _POLICY.clear()


def _reward(a: str) -> float:
    """Empirical mean reward for action *a* (0 if never tried)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """In‑place update of the global policy with a batch of observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Parent B – Voronoi + Geometric Algebra (adapted)
# ----------------------------------------------------------------------


Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the seed nearest to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of *points* to the nearest *seeds*."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def _blade_sign(indices):
    """Return a sorted list of distinct blade indices and the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate indices cancel (e_i * e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Geometric product of two blades (represented as frozensets of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple multivector for an n‑dimensional geometric algebra."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        """Extract the k‑grade part."""
        return Multivector({blade: coef for blade, coef in self.components.items() if len(blade) == k}, self.n)

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (distributive over addition)."""
        result: Dict[frozenset, float] = {}
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coef_a * coef_b
        return Multivector(result, self.n)

    __rmul__ = __mul__


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def bandit_to_point(action: BanditAction) -> Point:
    """
    Map a BanditAction to a 2‑D point for Voronoi seeding.

    The x‑coordinate is the empirical mean reward,
    the y‑coordinate is the confidence bound.
    """
    mean = _reward(action.action_id)
    return (mean, action.confidence_bound)


def assign_contexts_to_actions(
    contexts: List[Point],
    actions: List[BanditAction],
) -> Dict[str, List[Point]]:
    """
    Perform a Voronoi partition of *contexts* using the actions' (reward,
    confidence) points as seeds.  Returns a mapping from ``action_id`` to the
    list of contexts that fall in its region.
    """
    seeds = [bandit_to_point(a) for a in actions]
    regions = assign(contexts, seeds)
    return {actions[i].action_id: regions[i] for i in range(len(actions))}


def policy_multivector(actions: List[BanditAction]) -> Multivector:
    """
    Build a multivector that encodes the current policy.

    For each action *a_i* we create a blade ``e_i`` with coefficient equal to
    the cumulative reward (total reward, not the mean).  The geometric product
    of all such blades yields a single multivector representing the joint
    policy state.
    """
    # Gather cumulative rewards; if an action has never been tried its reward is 0.
    blades = []
    for idx, act in enumerate(actions):
        total, count = _POLICY.get(act.action_id, [0.0, 0.0])
        coeff = total  # total reward as coefficient
        if coeff != 0.0:
            blade = frozenset({idx})
            blades.append(Multivector({blade: coeff}, n=len(actions)))
        else:
            # Even a zero‑coefficient blade must be present for dimensionality
            blade = frozenset({idx})
            blades.append(Multivector({blade: 0.0}, n=len(actions)))

    # Geometric product of all action blades
    result = Multivector({frozenset(): 1.0}, n=len(actions))  # start with scalar 1
    for mv in blades:
        result = result * mv
    return result


def hybrid_step(
    contexts: List[Point],
    actions: List[BanditAction],
    updates: List[BanditUpdate],
) -> Tuple[Dict[str, List[Point]], Multivector]:
    """
    Execute a full hybrid iteration:

    1. Incorporate the incoming Bandit updates.
    2. Partition the contexts among actions via Voronoi.
    3. Produce the policy multivector.

    Returns the Voronoi regions and the policy multivector.
    """
    update_policy(updates)
    regions = assign_contexts_to_actions(contexts, actions)
    mv = policy_multivector(actions)
    return regions, mv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed reproducibility
    random.seed(0)

    # Define three dummy actions
    actions = [
        BanditAction(
            action_id="A",
            propensity=0.33,
            expected_reward=0.5,
            confidence_bound=0.1,
            algorithm="bandit",
        ),
        BanditAction(
            action_id="B",
            propensity=0.33,
            expected_reward=0.2,
            confidence_bound=0.2,
            algorithm="bandit",
        ),
        BanditAction(
            action_id="C",
            propensity=0.34,
            expected_reward=0.8,
            confidence_bound=0.05,
            algorithm="bandit",
        ),
    ]

    # Simulated updates (random rewards)
    updates = [
        BanditUpdate(context_id="c1", action_id="A", reward=1.0, propensity=0.33),
        BanditUpdate(context_id="c2", action_id="B", reward=0.0, propensity=0.33),
        BanditUpdate(context_id="c3", action_id="C", reward=1.0, propensity=0.34),
        BanditUpdate(context_id="c4", action_id="A", reward=0.5, propensity=0.33),
    ]

    # Random 2‑D context points
    contexts = [(random.uniform(0, 1), random.uniform(0, 0.3)) for _ in range(20)]

    # Run hybrid step
    regions, mv = hybrid_step(contexts, actions, updates)

    # Display results
    print("Voronoi regions (action_id -> number of contexts):")
    for aid, pts in regions.items():
        print(f"  {aid}: {len(pts)} points")

    print("\nPolicy multivector:")
    print(mv)

    # Simple sanity checks (should not raise)
    assert isinstance(regions, dict)
    assert isinstance(mv, Multivector)
    print("\nSmoke test completed successfully.")