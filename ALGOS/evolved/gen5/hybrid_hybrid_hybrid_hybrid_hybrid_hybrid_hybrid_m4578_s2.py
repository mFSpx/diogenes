# DARWIN HAMMER — match 4578, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m915_s0.py (gen4)
# born: 2026-05-29T23:56:40Z

"""Hybrid Sheaf‑Multivector Certainty (HSMC)

This module merges the two parent algorithms:

* **HybridSheafCertainty** (parent A) – a sheaf‑theoretic sketch where each
  node/edge carries a `CertaintyFlag`.  The coboundary operator is
  `δ(s) = w_u·R_u·s_u – w_v·R_v·s_v` with `w = confidence/10000`.

* **Multivector Geometry** (parent B) – a Clifford‑algebra multivector
  implementation with blade multiplication and grade extraction.  In the
  original endpoint‑circuit‑breaker code a health score weighted curvature
  before feeding it to multivector operations.

The **mathematical bridge** is a scalar weight that can be applied both to
sheaf sections and to multivector products.  We interpret the certainty
weight `w` and the health score `h∈[0,1]` as multiplicative factors and
define a **certainty‑health‑weighted coboundary**:


Δ_uv = (w_u·h)·R_u·s_u  –  (w_v·h)·R_v·s_v


`Δ_uv` is itself a multivector, so its norm can be used as a unified
measure of information loss (sheaf side) together with geometric
information (multivector side).  The functions below demonstrate this
fusion."""


import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty helpers (adapted)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Deterministic epistemic‑certainty container."""
    label: str
    confidence_bps: int                     # 0 .. 10000 basis points
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

    @property
    def weight(self) -> float:
        """Scalar weight w = confidence / 10000."""
        return self.confidence_bps / 10000.0


# ----------------------------------------------------------------------
# Parent B – Multivector (Clifford algebra) utilities
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort indices by bubble sort, returning the sign of the permutation."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate index → blade squares to zero; remove both
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades (as frozensets of indices)."""
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


class Multivector:
    """Element of the Clifford algebra Cl(n,0) as a sparse dict of blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # filter zero coefficients for compactness
        self.components: Dict[FrozenSet[int], float] = {
            frozenset(k): float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result, self.n)

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        result = dict(self.components)
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) - coeff
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> 'Multivector':
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Multivector by a scalar.")
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Geometric product (blade‑wise multiplication)
    # ------------------------------------------------------------------
    def geometric_product(self, other: 'Multivector') -> 'Multivector':
        """Geometric product ⨂ of two multivectors."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                new_blade, sign = _multiply_blades(blade_a, blade_b)
                new_coeff = coeff_a * coeff_b * sign
                result[new_blade] = result.get(new_blade, 0.0) + new_coeff
        return Multivector(result, self.n)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def grade(self, k: int) -> 'Multivector':
        """Return a multivector containing only blades of grade k."""
        return Multivector({b: c for b, c in self.components.items() if len(b) == k}, self.n)

    def norm(self) -> float:
        """Euclidean norm √(∑ coeff²)."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def __repr__(self) -> str:
        if not self.components:
            return f"Multivector(0, n={self.n})"
        terms = []
        for blade, coeff in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "e" + "^".join(str(i) for i in sorted(blade))
            else:
                basis = "1"
            terms.append(f"{coeff:.3g}{basis}")
        return " + ".join(terms)


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
@dataclass
class EndpointCircuitBreaker:
    """Simple health‑score container (0.0 – 1.0)."""
    health_score: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.health_score <= 1.0):
            raise ValueError("health_score must be in [0, 1]")


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def weighted_coboundary(
    node_sections: Dict[Any, Multivector],
    edge_restrictions: Dict[Tuple[Any, Any], Tuple[float, float]],
    node_flags: Dict[Any, CertaintyFlag],
    edge_flags: Dict[Tuple[Any, Any], CertaintyFlag],
    breaker: EndpointCircuitBreaker,
) -> Dict[Tuple[Any, Any], Multivector]:
    """
    Compute the certainty‑ and health‑weighted coboundary for each edge.

    For edge (u, v) with restriction scalars (r_u, r_v):
        Δ_uv = (w_u·h)·r_u·s_u  –  (w_v·h)·r_v·s_v

    Returns a mapping edge → resulting Multivector.
    """
    h = breaker.health_score
    result: Dict[Tuple[Any, Any], Multivector] = {}
    for (u, v), (r_u, r_v) in edge_restrictions.items():
        s_u = node_sections[u]
        s_v = node_sections[v]
        w_u = node_flags[u].weight
        w_v = node_flags[v].weight

        term_u = s_u * (w_u * h * r_u)
        term_v = s_v * (w_v * h * r_v)

        delta = term_u - term_v
        result[(u, v)] = delta
    return result


def multivector_product_with_health(
    a: Multivector,
    b: Multivector,
    breaker: EndpointCircuitBreaker,
) -> Multivector:
    """
    Compute the geometric product a ⨂ b and scale the result by the
    circuit‑breaker health score h.
    """
    prod = a.geometric_product(b)
    return prod * breaker.health_score


def hybrid_metric(
    node_sections: Dict[Any, Multivector],
    edge_restrictions: Dict[Tuple[Any, Any], Tuple[float, float]],
    node_flags: Dict[Any, CertaintyFlag],
    edge_flags: Dict[Tuple[Any, Any], CertaintyFlag],
    breaker: EndpointCircuitBreaker,
) -> float:
    """
    Unified metric combining:

    * The L2 norm of all weighted coboundary multivectors.
    * The norm of a health‑weighted geometric product of the first two nodes
      (if at least two nodes exist).

    The final metric is a weighted sum:
        M = α·Σ‖Δ_uv‖ + β·‖h·(s_i ⨂ s_j)‖
    with α=0.6, β=0.4 (arbitrary but fixed for demonstration).
    """
    α, β = 0.6, 0.4

    coboundaries = weighted_coboundary(
        node_sections, edge_restrictions, node_flags, edge_flags, breaker
    )
    coboundary_norm_sum = sum(mv.norm() for mv in coboundaries.values())

    # product of first two distinct nodes (if possible)
    nodes = list(node_sections.keys())
    if len(nodes) >= 2:
        prod_mv = multivector_product_with_health(
            node_sections[nodes[0]],
            node_sections[nodes[1]],
            breaker,
        )
        prod_norm = prod_mv.norm()
    else:
        prod_norm = 0.0

    return α * coboundary_norm_sum + β * prod_norm


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def random_multivector(dim: int, max_grade: int = 3) -> Multivector:
    """Create a random sparse multivector in Cl(dim,0) up to `max_grade`."""
    components: Dict[FrozenSet[int], float] = {}
    for grade in range(max_grade + 1):
        # pick a random number of blades for this grade
        count = random.randint(0, max(1, dim // 2))
        for _ in range(count):
            blade = frozenset(random.sample(range(dim), grade))
            coeff = random.uniform(-5.0, 5.0)
            if abs(coeff) > 1e-6:
                components[blade] = coeff
    return Multivector(components, dim)


def demo_hybrid_workflow() -> None:
    """Run a tiny example that exercises the hybrid functions."""
    # Graph with two nodes and one edge
    nodes = ("A", "B")
    edge = ("A", "B")

    # Random multivector sections per node
    node_sections = {
        n: random_multivector(dim=4) for n in nodes
    }

    # Simple scalar restrictions (could be matrices; using scalars here)
    edge_restrictions = {
        edge: (1.0, 1.0)
    }

    # Certainty flags with random confidence
    node_flags = {
        n: CertaintyFlag(
            label=random.choice(EPISTEMIC_FLAGS),
            confidence_bps=random.randint(0, 10000),
            authority_class="demo",
            rationale="random demo",
        )
        for n in nodes
    }

    edge_flags = {
        edge: CertaintyFlag(
            label=random.choice(EPISTEMIC_FLAGS),
            confidence_bps=random.randint(0, 10000),
            authority_class="demo",
            rationale="edge demo",
        )
    }

    # Random health score
    breaker = EndpointCircuitBreaker(health_score=random.random())

    # Compute hybrid metric
    metric = hybrid_metric(
        node_sections,
        edge_restrictions,
        node_flags,
        edge_flags,
        breaker,
    )
    print("Hybrid metric value:", metric)


if __name__ == "__main__":
    demo_hybrid_workflow()