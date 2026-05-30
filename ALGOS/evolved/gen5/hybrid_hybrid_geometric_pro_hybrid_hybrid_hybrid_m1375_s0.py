# DARWIN HAMMER — match 1375, survivor 0
# gen: 5
# parent_a: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s3.py (gen4)
# born: 2026-05-29T23:35:41Z

import math
import numpy as np
import random
import sys
import pathlib

# ---------------------------------------------------------------------------
# Parent Algorithm A: hybrid_geometric_product_voronoi_partition_m4_s1.py
# ---------------------------------------------------------------------------

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coeffi
    """

    def __init__(self, components):
        self.components = components

    def __add__(self, other):
        new_components = self.components.copy()
        for blade, coeff in other.components.items():
            if blade in new_components:
                new_components[blade] += coeff
            else:
                new_components[blade] = coeff
        return Multivector(new_components)

    def __rmul__(self, scalar):
        return Multivector({blade: scalar * coeff for blade, coeff in self.components.items()})


# ---------------------------------------------------------------------------
# Parent Algorithm B: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hdc_se_m172_s3.py
# ---------------------------------------------------------------------------

import math
import random
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Iterable

import numpy as np

# Hyperdimensional primitives
Vector = List[int]  # bipolar hypervector (elements are +1 or -1)


def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    """Generate a random bipolar hypervector of length *dim*."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    """Deterministic hypervector for an arbitrary symbol using SHA‑256 as seed."""
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    """Component‑wise binding (multiplication) of two hypervectors."""
    if len(a) != len(b):
        raise ValueError("vector dimensions must match")
    return [x * y for x, y in zip(a, b)]


# ---------------------------------------------------------------------------
# Hybrid Algorithm
# ---------------------------------------------------------------------------

def geometric_product_hypervector(multivector: Multivector, hypervector: Vector) -> float:
    """Compute the geometric product between a multivector and a hypervector."""
    result = 0
    for blade, coeff in multivector.components.items():
        result += coeff * np.dot(hypervector, list(blade))
    return result


def hypervector_distance(multivector: Multivector, hypervector: Vector) -> float:
    """Compute the Euclidean distance between a multivector and a hypervector."""
    return np.linalg.norm(geometric_product_hypervector(multivector, hypervector))


def hybrid_encoding(t: datetime, seed: int | str | None = None) -> Multivector:
    """Encode a timestamp using a deterministic hash-seeded random generator."""
    hypervector = symbol_vector(str(t), dim=10000)
    multivector = Multivector({frozenset(i): 1 for i in range(10000)})
    return multivector.__rmul__(hypervector_distance(multivector, hypervector))


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = datetime.now(timezone.utc)
    multivector = hybrid_encoding(t)
    print(multivector.components)
    hypervector = random_vector(dim=10000)
    distance = hypervector_distance(multivector, hypervector)
    print(distance)