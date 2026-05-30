# DARWIN HAMMER — match 104, survivor 4
# gen: 4
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py (gen3)
# born: 2026-05-29T23:27:06Z

"""
Hybrid Voronoi‑Circuit‑Breaker with Clifford Geometric‑Product Resource Allocation

Parents:
- hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py (Voronoi partition + EndpointCircuitBreaker)
- hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py (Clifford geometric product for multivector updates)

Mathematical Bridge:
The Voronoi diagram assigns each request point to the nearest site.  
Each site owns an EndpointCircuitBreaker (to model availability) and a
resource multivector **R** ∈ 𝔾(p) (Clifford algebra of dimension *p*).  
When a request arrives, the site’s circuit breaker decides if the
endpoint is usable; if so the request’s demand multivector **D**
is combined with the site’s resource via the **geometric product**
  R ← R ⊙ D
which simultaneously updates scalar, vector and higher‑grade components.
Thus spatial partitioning, reliability gating, and algebraic resource
allocation are fused into a single unified system.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient


# ----------------------------------------------------------------------
# Parent A – Voronoi helpers
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


# ----------------------------------------------------------------------
# Parent A – Circuit‑breaker
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = _now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = _now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def _now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Parent A – Morphology (simple geometric descriptor)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent B – Clifford geometric product utilities
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
                # Duplicate index → blade squares to scalar (removes pair)
                lst.pop(j)
                lst.pop(j)  # second element now at same position
                sign *= 1  # removal does not flip sign
                break
    return lst, sign


def _multiply_blades(blade_a: Blade, blade_b: Blade) -> Tuple[Blade, int]:
    """
    Multiply two basis blades using the exterior algebra sign rule.
    Returns (resulting_blade, sign).
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """
    Compute the Clifford geometric product of two multivectors.
    The result is a new multivector.
    """
    result: Multivector = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # Remove near‑zero entries for cleanliness
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def vector_to_multivector(vec: np.ndarray) -> Multivector:
    """
    Convert a 1‑D NumPy array into a grade‑1 multivector.
    Basis e_i corresponds to index i (starting at 0).
    """
    return {frozenset({i}): float(v) for i, v in enumerate(vec) if abs(v) > 1e-12}


def scalar_to_multivector(s: float) -> Multivector:
    """Represent a scalar as the empty blade."""
    return {frozenset(): float(s)}


def multivector_to_array(mv: Multivector, dim: int) -> np.ndarray:
    """
    Extract the vector (grade‑1) part of a multivector into a NumPy array.
    Non‑vector components are ignored.
    """
    arr = np.zeros(dim)
    for blade, coeff in mv.items():
        if len(blade) == 1:
            idx = next(iter(blade))
            if idx < dim:
                arr[idx] = coeff
    return arr


# ----------------------------------------------------------------------
# Hybrid Core – Resource allocation via Voronoi + CircuitBreaker + Geometric product
# ----------------------------------------------------------------------
def allocate_resources(
    regions: Dict[int, List[Point]],
    site_resources: Dict[int, Multivector],
    site_breakers: Dict[int, EndpointCircuitBreaker],
    demand_generator: callable,
    dim: int,
) -> Dict[int, Multivector]:
    """
    For each site:
      * If its circuit breaker allows operation, generate a demand multivector
        (based on the number of assigned points) and update the site’s resource
        via the geometric product: R ← R ⊙ D.
      * If the breaker is open, record a failure and leave the resource unchanged.
    Returns the updated resource dictionary.
    """
    updated_resources: Dict[int, Multivector] = {}
    for idx, pts in regions.items():
        breaker = site_breakers[idx]
        if breaker.allow():
            # Simple demand: magnitude proportional to number of points,
            # direction random in the *dim*‑dimensional vector space.
            magnitude = len(pts) * 0.1  # scaling factor
            random_dir = np.random.randn(dim)
            random_dir /= np.linalg.norm(random_dir) if np.linalg.norm(random_dir) else 1.0
            demand_vec = magnitude * random_dir
            demand_mv = vector_to_multivector(demand_vec)

            # Optional extra custom demand from user‑provided generator
            if demand_generator is not None:
                extra_mv = demand_generator(idx, pts, dim)
                if extra_mv:
                    demand_mv = geometric_product(demand_mv, extra_mv)

            # Update resource via geometric product
            new_res = geometric_product(site_resources[idx], demand_mv)
            updated_resources[idx] = new_res
            breaker.record_success()
        else:
            # Breaker open → failure recorded, resource unchanged
            breaker.record_failure()
            updated_resources[idx] = site_resources[idx]
    return updated_resources


def create_initial_resources(num_sites: int, dim: int) -> Dict[int, Multivector]:
    """
    Initialise each site with a random grade‑1 multivector (vector) and a scalar mass.
    """
    resources: Dict[int, Multivector] = {}
    for i in range(num_sites):
        vec = np.random.randn(dim)
        vec_mv = vector_to_multivector(vec)
        mass_mv = scalar_to_multivector(random.uniform(1.0, 10.0))
        # Combine by simple addition (geometric product of scalar with vector is just scaling)
        combined = {**vec_mv}
        for blade, coeff in mass_mv.items():
            combined[blade] = combined.get(blade, 0.0) + coeff
        resources[i] = combined
    return resources


def example_demand_generator(site_idx: int,
                             points: List[Point],
                             dim: int) -> Multivector | None:
    """
    Optional user‑supplied demand generator.
    Here we create a small scalar demand proportional to the site index.
    """
    scalar = 0.05 * (site_idx + 1)
    return scalar_to_multivector(scalar)


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    DIM = 3                     # dimension of the Clifford algebra (e1, e2, e3)
    NUM_SITES = 4
    NUM_POINTS = 50

    # Randomly generate site locations and request points
    np.random.seed(42)
    sites = [tuple(coord) for coord in np.random.uniform(0, 100, size=(NUM_SITES, 2))]
    points = [tuple(coord) for coord in np.random.uniform(0, 100, size=(NUM_POINTS, 2))]

    # Build Voronoi regions
    regions = compute_voronoi_regions(points, sites)

    # Initialise resources and circuit breakers
    resources = create_initial_resources(NUM_SITES, DIM)
    breakers = {i: EndpointCircuitBreaker(failure_threshold=2) for i in range(NUM_SITES)}

    # Perform a single allocation step
    updated = allocate_resources(
        regions,
        resources,
        breakers,
        demand_generator=example_demand_generator,
        dim=DIM,
    )

    # Display results
    print("=== Voronoi Regions (site → #points) ===")
    for idx, pts in regions.items():
        print(f"Site {idx}: {len(pts)} points")

    print("\n=== Circuit Breaker States ===")
    for idx, br in breakers.items():
        print(f"Site {idx}: open={br.open}, failures={br.failures}")

    print("\n=== Updated Resources (grade‑1 components) ===")
    for idx, mv in updated.items():
        vec = multivector_to_array(mv, DIM)
        print(f"Site {idx}: vector={vec}")

    sys.exit(0)