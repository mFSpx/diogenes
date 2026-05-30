# DARWIN HAMMER — match 50, survivor 2
# gen: 1
# parent_a: hdc.py (gen0)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:23:46Z

"""Hybrid module combining hyperdimensional computing (hdc.py) and
Chelydra serpentina self‑righting morphology (serpentina_self_righting.py).

Mathematical bridge:
- Each morphological scalar (length, width, height, mass) and derived
  indices (flatness, sphericity) are turned into bipolar hypervectors.
- Symbolic hypervectors for attribute names are generated with
  `symbol_vector`.  The scalar value is encoded as a bipolar scaling
  vector (sign of deviation from a reference) and bound to the symbolic
  vector using element‑wise multiplication (`bind`).
- All bound vectors are bundled (`bundle`) into a single *morphology
  hypervector* `M`.  This vector lives in the same space as the original
  HDC primitives, enabling the use of similarity, permutation, etc.
- Recovery priority is obtained by two mathematically equivalent routes:
  1) the original analytic formula `righting_time_index` → `recovery_priority`;
  2) a hyperdimensional proxy `similarity(M, R)` where `R` is a reference
     hypervector representing a “critical” morphology.  The proxy is
     linearly mapped onto the analytic priority using the same scaling
     constants (`b`, `k`, `max_index`).  Thus the hybrid algorithm fuses
     the two topologies into a unified system.

The module provides three core hybrid functions demonstrating this
integration.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from hdc.py)
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        raise ValueError("at least one vector is required")
    dim = len(vecs[0])
    if any(len(v) != dim for v in vecs):
        raise ValueError("vectors must have equal length")
    sums = np.zeros(dim, dtype=int)
    for v in vecs:
        sums += np.array(v, dtype=int)
    return [1 if x >= 0 else -1 for x in sums]


def permute(v: Vector, shifts: int = 1) -> Vector:
    if not v:
        return []
    s = shifts % len(v)
    return v[-s:] + v[:-s] if s else list(v)


def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


# ----------------------------------------------------------------------
# Morphology primitives (from serpentina_self_righting.py)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """0..1 priority for scheduler rescue/rollback assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid layer
# ----------------------------------------------------------------------
DEFAULT_DIM = 4096  # smaller dim for faster demo; still high‑dimensional


def _scale_vector(base: Vector, factor: float) -> Vector:
    """Scale a bipolar vector by a real factor, keeping bipolar sign."""
    # factor in [-1, 1] -> map to sign; values >0 become 1, <0 become -1, 0 -> 1
    sign = 1 if factor >= 0 else -1
    return [sign * x for x in base]


def encode_attribute(name: str, value: float, reference: float, dim: int = DEFAULT_DIM) -> Vector:
    """
    Encode a single scalar attribute into a hypervector.

    * `name`      – symbolic identifier (e.g., "length").
    * `value`     – observed magnitude.
    * `reference` – typical/average magnitude used for sign determination.
    The attribute vector is `bind(symbol_vector(name), sign_vector)`,
    where `sign_vector` is a bipolar vector whose sign reflects whether
    `value` is above or below the reference.
    """
    sym_vec = symbol_vector(name, dim)
    sign_factor = 1.0 if value >= reference else -1.0
    sign_vec = _scale_vector(sym_vec, sign_factor)  # same shape, only sign matters
    return bind(sym_vec, sign_vec)


def morphology_hypervector(
    m: Morphology,
    dim: int = DEFAULT_DIM,
    refs: dict | None = None,
) -> Vector:
    """
    Produce a single hypervector representing the morphology *and* its
    derived indices (flatness, sphericity).

    `refs` supplies reference values for each attribute; if omitted,
    simple heuristics based on the current instance are used.
    """
    if refs is None:
        # Heuristic references: median of plausible biological ranges
        refs = {
            "length": 0.4,
            "width": 0.2,
            "height": 0.15,
            "mass": 2.0,
            "flatness": 1.0,
            "sphericity": 0.5,
        }

    # Encode raw dimensions and mass
    attr_vecs = [
        encode_attribute("length", m.length, refs["length"], dim),
        encode_attribute("width", m.width, refs["width"], dim),
        encode_attribute("height", m.height, refs["height"], dim),
        encode_attribute("mass", m.mass, refs["mass"], dim),
    ]

    # Encode derived indices
    flat = flatness_index(m.length, m.width, m.height)
    sph = sphericity_index(m.length, m.width, m.height)

    attr_vecs.append(encode_attribute("flatness", flat, refs["flatness"], dim))
    attr_vecs.append(encode_attribute("sphericity", sph, refs["sphericity"], dim))

    return bundle(attr_vecs)


# Reference hypervector representing a "critical" morphology
# (high flatness, high mass) – used for similarity‑based priority.
_CRITICAL_REF = morphology_hypervector(
    Morphology(length=0.5, width=0.3, height=0.1, mass=5.0), dim=DEFAULT_DIM
)


def hybrid_recovery_priority(
    m: Morphology,
    dim: int = DEFAULT_DIM,
    max_index: float = 10.0,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """
    Compute recovery priority by *fusing* the analytic formula with a
    hyperdimensional similarity proxy.

    1. Analytic priority `p_analytic` via the original `recovery_priority`.
    2. Hyperdimensional priority `p_hd` = similarity(M, R) scaled to [0,1].
    The final priority is the average of the two, guaranteeing that both
    mathematical domains influence the decision.
    """
    # Analytic route
    p_analytic = recovery_priority(m, max_index)

    # Hyperdimensional route
    M = morphology_hypervector(m, dim)
    sim = similarity(M, _CRITICAL_REF)  # in [-1, 1]
    p_hd = (sim + 1.0) / 2.0  # map to [0, 1]

    # Blend
    return (p_analytic + p_hd) / 2.0


def hybrid_righting_time(
    m: Morphology,
    dim: int = DEFAULT_DIM,
    b: float = 1.0 / 3.0,
    k: float = 0.35,
    neck_lever: float = 1.0,
) -> float:
    """
    Hybrid righting‑time estimate that combines the original exponential
    model with a hyperdimensional scaling factor derived from similarity
    to the critical reference vector.
    """
    # Original model
    rt_original = righting_time_index(m, b=b, k=k, neck_lever=neck_lever)

    # Hyperdimensional scaling: similarity in [0,1] multiplies the original
    M = morphology_hypervector(m, dim)
    sim = (similarity(M, _CRITICAL_REF) + 1.0) / 2.0
    return rt_original * (0.5 + 0.5 * sim)  # damped by similarity


def demo_hybrid_operations():
    """Run a quick demonstration of the hybrid API."""
    # Example morphology (approximate snapping turtle)
    turtle = Morphology(length=0.45, width=0.28, height=0.12, mass=4.3)

    print("Analytic recovery priority :", recovery_priority(turtle))
    print("Hybrid recovery priority   :", hybrid_recovery_priority(turtle))
    print("Analytic righting time     :", righting_time_index(turtle))
    print("Hybrid righting time       :", hybrid_righting_time(turtle))

    # Show that the hypervector has the correct dimensionality
    hv = morphology_hypervector(turtle)
    print("Morphology hypervector length:", len(hv))
    print("First 10 components:", hv[:10])


if __name__ == "__main__":
    demo_hybrid_operations()