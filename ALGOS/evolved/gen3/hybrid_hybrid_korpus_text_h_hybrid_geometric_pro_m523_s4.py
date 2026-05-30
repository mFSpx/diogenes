# DARWIN HAMMER — match 523, survivor 4
# gen: 3
# parent_a: hybrid_korpus_text_hybrid_krampus_brain_m43_s0.py (gen2)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s0.py (gen1)
# born: 2026-05-29T23:29:20Z

"""
Hybrid Text‑Geometric Voronoi Algorithm
======================================

This module fuses the core of **PARENT ALGORITHM A** (min‑hashing of text) with the
core of **PARENT ALGORITHM B** (Clifford geometric product inside Voronoi
partitions).

Mathematical bridge
-------------------
* The min‑hash routine produces a *k*‑dimensional integer signature.
* Each integer of the signature is deterministically mapped to a 2‑D point.
* A set of seed points (taken from the signature) defines a Voronoi diagram that
  partitions the point cloud.
* For every Voronoi region we aggregate a **multivector** built from textual
  statistics (entropy, master‑vector keys, …).  The aggregation is a simple
  coefficient sum over the blades.
* Finally the multivectors of the regions are combined by the **Clifford
  geometric product** defined in the original geometric‑product module.

The result is a single multivector that encodes the whole text while respecting
the spatial relationships imposed by the Voronoi partition of its min‑hash
signature.

Only the standard library, ``numpy`` and ``math`` are used.
"""

import math
import random
import sys
from pathlib import Path
from collections import deque
import re
import numpy as np

# ----------------------------------------------------------------------
# 1.  Text – min‑hash utilities (from parent A)
# ----------------------------------------------------------------------
def minhash_signature(text: str, k: int = 64) -> np.ndarray:
    """Return a k‑length min‑hash signature for *text*."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h & 0xFFFFFFFFFFFFFFFF)
    return signature


def entropy_for_text(text: str) -> float:
    """Shannon‑like entropy proxy used as a scalar feature."""
    txt = (text or "")[:10000]
    return float(len(set(txt))) / len(txt) if txt else 0.0


def extract_master_vector(text: str) -> dict[str, float]:
    """Create a deterministic dict of 24 scalar features from *text*."""
    if not text.strip():
        return {}
    # deterministic pseudo‑features based on hash of the text and a small key list
    keys = [
        "visceral_ratio", "tech_ratio", "legal_osint_ratio", "ledger_density",
        "recursion_score", "directive_ratio", "target_density",
        "forensic_shield_ratio", "poetic_entropy", "dissociative_index",
        "wrath_velocity", "bureaucratic_weaponization_index",
        "resource_exhaustion_metric", "swarm_orchestration_density",
        "logic_crucifixion_index", "conspiracy_grounding_ratio",
        "chaotic_good_tax", "corporate_grit_tension", "countdown_density",
        "asset_structuring_weight", "pitch_formatting_ratio",
        "agent_symmetry_ratio", "protocol_discip", "synthetic_motive"
    ]
    base = hash(text)
    return {k: ((base ^ hash(k)) % 1000) / 1000.0 for k in keys}


# ----------------------------------------------------------------------
# 2.  Voronoi utilities (from parent B)
# ----------------------------------------------------------------------
Point = tuple[float, float]


def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: list[Point]) -> int:
    """Index of the nearest seed (ties broken by lower index)."""
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: (_euclidean(point, seeds[i]), i))


def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    """Assign each *point* to the region of its nearest *seed*."""
    regions: dict[int, list[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


# ----------------------------------------------------------------------
# 3.  Clifford geometric product utilities (from parent B)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vectors cancel (e² = 0)
                lst.pop(j)
                lst.pop(j)
                n -= 2
                sign = sign  # unchanged
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Simple multivector for an n‑dimensional Clifford algebra."""

    def __init__(self, components: dict[frozenset[int], float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match")
        new = self.components.copy()
        for b, c in other.components.items():
            new[b] = new.get(b, 0.0) + c
            if abs(new[b]) < 1e-12:
                del new[b]
        return Multivector(new, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match")
        result: dict[frozenset[int], float] = {}
        for ba, ca in self.components.items():
            for bb, cb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                coef = ca * cb * sign
                result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                term = f"{coef:.3g}"
            else:
                basis = "".join(f"e{i}" for i in sorted(blade))
                term = f"{coef:.3g}{basis}"
            terms.append(term)
        return " + ".join(terms)


# ----------------------------------------------------------------------
# 4.  Bridge helpers – mapping signature → points → regions → multivectors
# ----------------------------------------------------------------------
def signature_to_points(sig: np.ndarray) -> list[Point]:
    """
    Deterministically map each integer of *sig* to a 2‑D point.
    The mapping folds the integer into a unit square.
    """
    points: list[Point] = []
    for v in sig:
        # simple hash‑mix to obtain reproducible coordinates
        x = ((v >> 16) & 0xFFFF) / 0xFFFF
        y = (v & 0xFFFF) / 0xFFFF
        points.append((x, y))
    return points


def text_to_multivector(text: str, algebra_dim: int = 5) -> Multivector:
    """
    Encode *text* as a multivector.
    Each key of the master vector is turned into a blade whose basis indices
    are derived from the hash of the key modulo *algebra_dim*.
    The scalar coefficient is the corresponding feature value.
    """
    master = extract_master_vector(text)
    comps: dict[frozenset[int], float] = {}
    for key, val in master.items():
        # map the key to a set of basis indices (size 1‑3)
        base = hash(key)
        indices = frozenset({(base >> (2 * i)) % algebra_dim for i in range(3)})
        comps[indices] = comps.get(indices, 0.0) + val
    # add a scalar part proportional to entropy
    comps[frozenset()] = entropy_for_text(text)
    return Multivector(comps, algebra_dim)


def hybrid_text_geometric_product(
    text: str,
    k: int = 64,
    seed_count: int = 8,
    algebra_dim: int = 5,
) -> Multivector:
    """
    Full hybrid pipeline:

    1. Compute a min‑hash signature of *text*.
    2. Map the signature to 2‑D points.
    3. Choose *seed_count* seeds (first points) and build a Voronoi diagram.
    4. For each region compute a region‑multivector by scaling the text
       multivector with the region size.
    5. Reduce all region multivectors with the Clifford geometric product.

    The final multivector is returned.
    """
    # 1. min‑hash
    sig = minhash_signature(text, k)

    # 2. points
    pts = signature_to_points(sig)

    if seed_count > len(pts):
        raise ValueError("seed_count exceeds number of signature points")
    seeds = pts[:seed_count]

    # 3. Voronoi partition
    regions = assign(pts, seeds)

    # 4. Region multivectors
    region_mvs: list[Multivector] = []
    base_mv = text_to_multivector(text, algebra_dim)

    for idx, pts_in_region in regions.items():
        weight = len(pts_in_region) / len(pts)  # relative region weight
        # scale each component by the weight
        scaled_components = {
            blade: coef * weight for blade, coef in base_mv.components.items()
        }
        region_mvs.append(Multivector(scaled_components, algebra_dim))

    # 5. Geometric product fold
    result = region_mvs[0] if region_mvs else Multivector({}, algebra_dim)
    for mv in region_mvs[1:]:
        result = result * mv
    return result


# ----------------------------------------------------------------------
# 6.  Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "In the quiet glade, the algorithm whispered, "
        "binding text to geometry, forging a new algebraic bridge."
    )
    final_mv = hybrid_text_geometric_product(sample, k=32, seed_count=4, algebra_dim=4)
    print("Hybrid multivector result:")
    print(final_mv)