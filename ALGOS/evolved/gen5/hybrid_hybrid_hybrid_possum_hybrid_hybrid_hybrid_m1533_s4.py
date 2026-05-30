# DARWIN HAMMER — match 1533, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1027_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s1.py (gen4)
# born: 2026-05-29T23:37:19Z

"""Hybrid Morphology‑Geometric Free‑Energy Algorithm
-------------------------------------------------
Parent A: hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1027_s0.py  
Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m296_s1.py  

Mathematical Bridge
~~~~~~~~~~~~~~~~~~~
The morphology‑driven recovery priority of an ``Entity`` (product of
sphericity, flatness and a righting‑time index) is interpreted as the
scalar (grade‑0) component of a geometric‑algebra ``Multivector``.
That scalar scales the Hoeffding bound used in the parent‑B decision
logic, thus allowing dynamic endpoint selection that reacts to the
physical shape of the data point.  The hybrid functions below construct
the multivector, compute the scaled bound and expose a split‑decision
criterion that fuses both parent topologies in a single mathematically
coherent system.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    mass: float = 0.0

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximum edge length."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """How flat an object is (larger => flatter)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Entity, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Simplified righting‑time model:
        τ = k * (m.mass * neck_lever) ** b
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    return k * (m.mass * neck_lever) ** b

# ----------------------------------------------------------------------
# Parent B – Geometric‑Algebra utilities
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel equal basis vectors (e_i ∧ e_i = 0)
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Simple exterior‑algebra multivector limited to scalar and basis blades."""
    def __init__(self, components: dict, n: int):
        # keep only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int):
        """Extract grade‑k part."""
        return Multivector(
            {bl: self.components[bl] for bl in self.components if len(bl) == k},
            self.n,
        )

    def __mul__(self, other):
        if isinstance(other, Multivector):
            result = {}
            for k, v in self.components.items():
                for k2, v2 in other.components.items():
                    new_k, sign = _multiply_blades(k, k2)
                    result[new_k] = result.get(new_k, 0.0) + sign * v * v2
            return Multivector(result, self.n)
        elif isinstance(other, (int, float)):
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)
        else:
            raise TypeError("Unsupported operand type for * with Multivector")

    __rmul__ = __mul__

def hoeffding_bound(r: float, delta: float, n: int, multivector: Multivector) -> float:
    """
    Classic Hoeffding bound, scaled by the scalar part of ``multivector``.
    """
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    scalar = multivector.components.get(frozenset(), 1.0)
    scaled_r = r * scalar
    return math.sqrt((scaled_r * scaled_r * math.log(1.0 / delta)) / (2.0 * n))

# ----------------------------------------------------------------------
# Hybrid layer – marrying morphology with Hoeffding logic
# ----------------------------------------------------------------------
def entity_to_multivector(entity: Entity) -> Multivector:
    """
    Encode an Entity as a multivector:
        - scalar (grade‑0)   : 1.0 (base weight)
        - e1 (grade‑1)       : length
        - e2 (grade‑1)       : width
        - e3 (grade‑1)       : height
        - e4 (grade‑1)       : mass
    Basis blades are represented by frozensets of integer indices.
    """
    comps = {
        frozenset(): 1.0,                     # scalar
        frozenset({1}): entity.length,
        frozenset({2}): entity.width,
        frozenset({3}): entity.height,
        frozenset({4}): entity.mass,
    }
    # ``n`` is the dimensionality of the underlying space (4 here)
    return Multivector(comps, n=4)

def hybrid_priority(entity: Entity) -> float:
    """
    Morphology‑driven priority (dimensionless) used as the ``r`` argument
    for the Hoeffding bound.
    """
    sph = sphericity_index(entity.length, entity.width, entity.height)
    flt = flatness_index(entity.length, entity.width, entity.height)
    rgt = righting_time_index(entity)
    # Geometric mean of the three indices gives a balanced priority
    return (sph * flt * rgt) ** (1.0 / 3.0)

def hybrid_hoeffding_decision(entity: Entity,
                              day_of_week: int,
                              delta: float = 0.05,
                              min_samples: int = 5) -> bool:
    """
    Decide whether to split / create a new endpoint for ``entity``.
    The decision threshold adapts to the day of the week (simulating
    the dynamic endpoint selection of parent‑A) and to the morphology‑driven
    priority via a scaled Hoeffding bound.
    """
    # 1️⃣ Morphology priority as the Hoeffding range parameter
    r = hybrid_priority(entity)

    # 2️⃣ Number of observations – a simple function of priority and day
    n = max(min_samples, int(r * (day_of_week + 1) * 10))

    # 3️⃣ Build the multivector representation
    mv = entity_to_multivector(entity)

    # 4️⃣ Compute the bound
    bound = hoeffding_bound(r, delta, n, mv)

    # 5️⃣ Heuristic split rule: split if bound exceeds a day‑dependent cutoff
    cutoff = 0.1 + 0.02 * (day_of_week % 7)   # grows slightly through the week
    return bound > cutoff

# ----------------------------------------------------------------------
# Demonstration utilities (three distinct hybrid functions)
# ----------------------------------------------------------------------
def hybrid_recovery_score(entity: Entity, today: date) -> float:
    """
    Returns a composite score blending morphology priority with the
    Hoeffding bound.  Higher scores indicate a stronger need for recovery
    or endpoint creation.
    """
    day_idx = today.weekday()                     # Monday == 0
    priority = hybrid_priority(entity)
    bound = hoeffding_bound(priority, 0.05,
                            int(priority * 20) + 1,
                            entity_to_multivector(entity))
    # Weighted sum – weights chosen to balance the two contributions
    return 0.6 * priority + 0.4 * bound + 0.01 * day_idx

def batch_hybrid_decision(entities: list[Entity]) -> dict[str, bool]:
    """
    Apply ``hybrid_hoeffding_decision`` to a batch, using the current
    weekday for all entries.  Returns a mapping from entity id to decision.
    """
    today = date.today()
    day_idx = today.weekday()
    decisions = {}
    for e in entities:
        decisions[e.id] = hybrid_hoeffding_decision(e, day_idx)
    return decisions

def simulate_endpoint_selection(entities: list[Entity],
                                delta: float = 0.05) -> list[Entity]:
    """
    From a list of candidates, keep only those that do *not* trigger a split.
    This mimics the dynamic endpoint selection of the original parent‑A
    algorithm, now governed by the hybrid Hoeffding bound.
    """
    today = date.today()
    selected = [e for e in entities
                if not hybrid_hoeffding_decision(e, today.weekday(), delta)]
    return selected

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few synthetic entities
    ents = [
        Entity(id="A1", lat=0.0, lon=0.0, category="type1",
               length=2.0, width=2.0, height=1.0, mass=5.0),
        Entity(id="B2", lat=1.0, lon=1.0, category="type2",
               length=1.0, width=1.0, height=0.5, mass=2.0),
        Entity(id="C3", lat=-1.0, lon=2.0, category="type3",
               length=3.0, width=2.5, height=2.0, mass=8.0),
    ]

    # 1️⃣ Hybrid recovery scores
    for e in ents:
        score = hybrid_recovery_score(e, date.today())
        print(f"Entity {e.id} recovery score: {score:.4f}")

    # 2️⃣ Batch decision dictionary
    decisions = batch_hybrid_decision(ents)
    print("\nBatch split decisions:")
    for eid, dec in decisions.items():
        print(f"  {eid}: {'split' if dec else 'keep'}")

    # 3️⃣ Simulated endpoint selection
    kept = simulate_endpoint_selection(ents)
    kept_ids = [e.id for e in kept]
    print("\nEntities kept after endpoint selection:", kept_ids)

    # Verify that the multivector arithmetic works without error
    mv = entity_to_multivector(ents[0])
    mv2 = entity_to_multivector(ents[1])
    product = mv * mv2
    print("\nSample multivector product (scalar part):",
          product.components.get(frozenset(), None))