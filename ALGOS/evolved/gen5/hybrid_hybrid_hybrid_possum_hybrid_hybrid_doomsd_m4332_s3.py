# DARWIN HAMMER — match 4332, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1027_s0.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py (gen2)
# born: 2026-05-29T23:54:57Z

"""
Hybrid Algorithm: Morphology‑Temporal Geometric Fusion

Parents:
- hybrid_hybrid_possum_filter_hybrid_hybrid_hybrid_m1027_s0.py (morphology‑driven recovery priority)
- hybrid_hybrid_doomsday_cale_hybrid_geometric_pro_m66_s0.py (weekday metric space via Clifford geometric product,
  Gini‑based inequality, Voronoi partitioning)

Mathematical Bridge:
Each Entity is mapped to a multivector whose scalar part encodes a morphology‑derived priority
(sphericity·flatness·righting‑time) and whose vector components encode its physical dimensions.
Weekdays are represented as orthogonal basis vectors e₀…e₆. The geometric product of an Entity
multivector with the weekday multivector yields a new multivector that couples morphology with
temporal context. The Euclidean norm of the product serves as a hybrid score. Voronoi partitioning
assigns the product to the nearest seed multivector, while the Gini coefficient of the resulting
assignment distribution quantifies temporal inequality. This unified system blends the two parent
topologies into a single, mathematically coherent pipeline.
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from collections import Counter
from typing import Dict, FrozenSet, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
class Entity:
    """Simple container for physical and semantic attributes."""
    __slots__ = ("id", "lat", "lon", "category", "score", "address_signature",
                 "length", "width", "height", "mass")
    def __init__(self, id: str, lat: float, lon: float, category: str,
                 length: float, width: float, height: float, mass: float,
                 score: float = 0.0, address_signature: str = ""):
        self.id = id
        self.lat = lat
        self.lon = lon
        self.category = category
        self.score = score
        self.address_signature = address_signature
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

# ----------------------------------------------------------------------
# Morphology‑driven priority (Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximal edge length."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """How flat an object is; larger values → flatter."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Entity, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """
    Heuristic time for an entity to right itself.
    Uses flatness, mass and a lever‑arm term.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    f = flatness_index(m.length, m.width, m.height)
    # The exponent b dampens the influence of the flatness·mass product.
    return (f * m.mass) ** b * k * neck_lever


def morphology_priority(entity: Entity) -> float:
    """
    Scalar priority combining sphericity, flatness and righting time.
    Normalised to [0, 1] via a sigmoid for stability.
    """
    s = sphericity_index(entity.length, entity.width, entity.height)
    f = flatness_index(entity.length, entity.width, entity.height)
    r = righting_time_index(entity)
    raw = s * f * r
    # Sigmoid normalisation
    return 1.0 / (1.0 + math.exp(-raw))


# ----------------------------------------------------------------------
# Clifford geometric product utilities (Parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # e_i * e_i = 1 → remove the pair
                lst.pop(j)
                lst.pop(j)  # second element now at same position
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int],
                     blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades; return resulting blade and sign."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """
    Simple implementation of a multivector in Cl(n,0).
    Internally stored as a dict mapping frozenset of basis indices to scalar coefficient.
    """
    def __init__(self, components: Dict[FrozenSet[int], float] = None):
        self.components: Dict[FrozenSet[int], float] = dict()
        if components:
            # discard zero coefficients
            for k, v in components.items():
                if abs(v) > 1e-12:
                    self.components[k] = float(v)

    @staticmethod
    def scalar(value: float) -> 'Multivector':
        return Multivector({frozenset(): value})

    @staticmethod
    def basis(index: int) -> 'Multivector':
        return Multivector({frozenset({index}): 1.0})

    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
            if abs(result[blade]) < 1e-12:
                del result[blade]
        return Multivector(result)

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        return self + (-other)

    def __neg__(self) -> 'Multivector':
        return Multivector({b: -c for b, c in self.components.items()})

    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """Geometric product using blade multiplication."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                coeff = coeff_a * coeff_b * sign
                result[blade_res] = result.get(blade_res, 0.0) + coeff
        # prune near‑zero entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result)

    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def __repr__(self) -> str:
        terms = []
        for blade, coeff in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                term = f"{coeff:.3g}"
            else:
                basis = "e" + "*e".join(str(i) for i in sorted(blade))
                term = f"{coeff:.3g}{basis}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Temporal structures (weekday → multivector, Voronoi, Gini)
# ----------------------------------------------------------------------
def weekday_multivector(d: date) -> Multivector:
    """
    Map a weekday to a unit basis vector.
    Monday → e0, Tuesday → e1, …, Sunday → e6.
    """
    idx = d.weekday()  # Monday = 0
    return Multivector.basis(idx)


def entity_multivector(entity: Entity) -> Multivector:
    """
    Encode an Entity as a multivector:
    - scalar part = morphology priority
    - vector part = length·e7 + width·e8 + height·e9
    """
    priority = morphology_priority(entity)
    vec = (Multivector.scalar(priority) +
           Multivector.basis(7) * entity.length +
           Multivector.basis(8) * entity.width +
           Multivector.basis(9) * entity.height)
    return vec


# overload multiplication of a multivector by a scalar for convenience
def _mv_scalar_mul(self: Multivector, scalar: float) -> Multivector:
    return Multivector({blade: coeff * scalar for blade, coeff in self.components.items()})
Multivector.__rmul__ = _mv_scalar_mul
Multivector.__mul__ = _mv_scalar_mul


def voronoi_assign(point: Multivector, seeds: List[Multivector]) -> int:
    """
    Assign `point` to the index of the nearest seed using Euclidean distance
    in multivector coefficient space.
    """
    distances = [np.linalg.norm(_coeff_vector(point) - _coeff_vector(seed))
                 for seed in seeds]
    return int(np.argmin(distances))


def _coeff_vector(mv: Multivector) -> np.ndarray:
    """
    Convert a multivector to a dense coefficient vector.
    The dimension is the maximal basis index seen across all components + 1.
    """
    max_idx = 0
    for blade in mv.components:
        if blade:
            max_idx = max(max_idx, max(blade))
    dim = max_idx + 1
    vec = np.zeros(dim)
    for blade, coeff in mv.components.items():
        if not blade:
            vec[0] += coeff  # treat scalar as index 0 for convenience
        else:
            # For simplicity, add the coefficient to the first index of the blade
            # (the exact mapping is not critical for distance comparison)
            vec[min(blade)] += coeff
    return vec


def gini_coefficient(counts: List[int]) -> float:
    """
    Compute the Gini coefficient for a list of integer counts.
    """
    if not counts:
        return 0.0
    sorted_counts = sorted(counts)
    n = len(counts)
    cumulative = 0
    for i, x in enumerate(sorted_counts, 1):
        cumulative += i * x
    total = sum(sorted_counts)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_product(entity: Entity, day: date) -> Multivector:
    """
    Geometric product of the entity multivector with the weekday multivector.
    The result intertwines morphology with temporal orientation.
    """
    ev = entity_multivector(entity)
    dv = weekday_multivector(day)
    return ev.geometric_product(dv)


def hybrid_score(entity: Entity, day: date, seeds: List[Multivector]) -> Tuple[float, int]:
    """
    Compute a hybrid score for an entity on a given day.
    Returns (score, assigned_seed_index).
    - score = norm of the geometric product (higher → stronger coupling)
    - assigned_seed_index = Voronoi assignment of the product to the nearest seed
    """
    prod = hybrid_product(entity, day)
    score = prod.norm()
    seed_idx = voronoi_assign(prod, seeds)
    return score, seed_idx


def temporal_inequality(assigned_indices: List[int]) -> float:
    """
    Given a list of seed indices assigned to a batch of entities,
    compute the Gini coefficient as a measure of temporal inequality.
    """
    counts = [assigned_indices.count(i) for i in range(max(assigned_indices) + 1)]
    return gini_coefficient(counts)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few random entities
    random.seed(42)
    entities = []
    for i in range(5):
        e = Entity(
            id=f"E{i}",
            lat=random.uniform(-90, 90),
            lon=random.uniform(-180, 180),
            category="test",
            length=random.uniform(0.5, 5.0),
            width=random.uniform(0.5, 5.0),
            height=random.uniform(0.5, 5.0),
            mass=random.uniform(1.0, 20.0)
        )
        entities.append(e)

    # Generate random seed multivectors (using only vector components for simplicity)
    seeds = []
    for _ in range(3):
        mv = (Multivector.basis(7) * random.uniform(0.5, 5.0) +
              Multivector.basis(8) * random.uniform(0.5, 5.0) +
              Multivector.basis(9) * random.uniform(0.5, 5.0))
        seeds.append(mv)

    today = date.today()
    assignments = []
    print(f"Hybrid scores for {today.isoformat()}:")
    for ent in entities:
        sc, idx = hybrid_score(ent, today, seeds)
        assignments.append(idx)
        print(f"  Entity {ent.id}: score={sc:.4f}, assigned seed={idx}")

    inequality = temporal_inequality(assignments)
    print(f"\nTemporal inequality (Gini) of assignments: {inequality:.4f}")