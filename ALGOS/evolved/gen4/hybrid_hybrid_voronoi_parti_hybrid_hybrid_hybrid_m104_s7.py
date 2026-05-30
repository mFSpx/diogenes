# DARWIN HAMMER — match 104, survivor 7
# gen: 4
# parent_a: hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py (gen2)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s4.py (gen3)
# born: 2026-05-29T23:27:06Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = int                     # bitmask representation, e.g. e1^e3 -> 0b1010
Multivector = Dict[Blade, float]   # mapping blade → coefficient


# ----------------------------------------------------------------------
# Utility – Bit‑wise blade handling (Euclidean metric)
# ----------------------------------------------------------------------
def _grade(blade: Blade) -> int:
    """Number of set bits = grade of the blade."""
    return blade.bit_count()


def _blade_sign(a: Blade, b: Blade) -> int:
    """
    Sign of the geometric product of two basis blades a and b
    under a Euclidean metric (e_i^2 = +1).
    Implements the rule:
        e_i e_j = - e_j e_i  for i != j
    The sign is (-1)^{N} where N is the number of swaps needed to
    reorder the concatenated index list.
    """
    # Count swaps using the classic "grade involution" method:
    # For each set bit in a, count how many bits in b are lower index.
    sign = 1
    while a:
        lowest = a & -a               # isolate lowest set bit of a
        idx = (lowest.bit_length() - 1)
        # bits in b with index < idx cause a swap
        lower = b & ((1 << idx) - 1)
        if lower.bit_count() & 1:
            sign = -sign
        a ^= lowest
    return sign


def geometric_product(mv_a: Multivector, mv_b: Multivector) -> Multivector:
    """
    Euclidean Clifford geometric product using bit‑mask blades.
    Returns a new multivector.
    """
    result: Multivector = {}
    for blade_a, coeff_a in mv_a.items():
        for blade_b, coeff_b in mv_b.items():
            blade_res = blade_a ^ blade_b
            sign = _blade_sign(blade_a, blade_b)
            coeff_res = coeff_a * coeff_b * sign
            result[blade_res] = result.get(blade_res, 0.0) + coeff_res
    # prune near‑zero entries
    return {b: c for b, c in result.items() if abs(c) > 1e-12}


def scalar_to_mv(s: float) -> Multivector:
    return {0: float(s)}   # empty blade = 0b0


def vector_to_mv(vec: np.ndarray) -> Multivector:
    """Convert a 1‑D array to a grade‑1 multivector (e_i ↔ bit i)."""
    mv: Multivector = {}
    for i, v in enumerate(vec):
        if abs(v) > 1e-12:
            mv[1 << i] = float(v)
    return mv


def mv_to_vector(mv: Multivector, dim: int) -> np.ndarray:
    """Extract the vector (grade‑1) part."""
    arr = np.zeros(dim)
    for blade, coeff in mv.items():
        if _grade(blade) == 1:
            idx = blade.bit_length() - 1
            if idx < dim:
                arr[idx] = coeff
    return arr


# ----------------------------------------------------------------------
# Voronoi helpers (2‑D)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point], sites: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        dists = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(dists))
        regions[nearest].append(pt)
    return regions


def region_centroid(region: List[Point]) -> Point:
    if not region:
        return (0.0, 0.0)
    xs, ys = zip(*region)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ----------------------------------------------------------------------
# Circuit‑breaker with exponential back‑off
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, backoff_factor: float = 2.0):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.backoff_factor = backoff_factor
        self.failures = 0
        self.open = False
        self.last_event_at = _now_z()
        self._next_retry_delay = 1.0  # seconds, placeholder for future async use

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = _now_z()
        self._next_retry_delay = 1.0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True
            self._next_retry_delay *= self.backoff_factor
        self.last_event_at = _now_z()

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
            "next_retry_delay": self._next_retry_delay,
        }


def _now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Morphology → multivector embedding
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    def to_multivector(self) -> Multivector:
        """Embed physical attributes as a mixed‑grade multivector:
        scalar → mass, vector → (length, width, height)."""
        scalar = scalar_to_mv(self.mass)
        vector = vector_to_mv(np.array([self.length, self.width, self.height]))
        return {**scalar, **vector}


# ----------------------------------------------------------------------
# Site abstraction (combines location, resource, breaker)
# ----------------------------------------------------------------------
@dataclass
class Site:
    index: int
    location: Point
    resource: Multivector = field(default_factory=lambda: scalar_to_mv(1.0))
    breaker: EndpointCircuitBreaker = field(default_factory=EndpointCircuitBreaker)
    morphology: Morphology | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "index": self.index,
            "location": self.location,
            "resource": {tuple(sorted(_blade_to_indices(b))): c for b, c in self.resource.items()},
            "breaker": self.breaker.as_dict(),
        }
        if self.morphology:
            d["morphology"] = asdict(self.morphology)
        return d


def _blade_to_indices(blade: Blade) -> List[int]:
    """Convert bitmask to list of basis indices."""
    indices = []
    i = 0
    while blade:
        if blade & 1:
            indices.append(i)
        blade >>= 1
        i += 1
    return indices


# ----------------------------------------------------------------------
# Demand generation – geometry‑aware
# ----------------------------------------------------------------------
def generate_demand(site: Site, region: List[Point], dim: int) -> Multivector:
    """
    Demand multivector derived from:
      * magnitude proportional to region size,
      * direction pointing from site to region centroid,
      * optional scalar load proportional to average distance.
    """
    if not region:
        return scalar_to_mv(0.0)

    centroid = region_centroid(region)
    dir_vec = np.array([centroid[0] - site.location[0], centroid[1] - site.location[1]])
    if np.linalg.norm(dir_vec) == 0:
        dir_vec = np.random.randn(2)  # fallback random direction
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    # Pad to requested dimension (allow dim > 2)
    if dim > 2:
        extra = np.random.randn(dim - 2)
        dir_vec = np.concatenate([dir_vec, extra])
    magnitude = len(region) * 0.05  # tunable scaling
    demand_vec = dir_vec * magnitude

    # Scalar component: average Euclidean distance as load factor
    distances = [euclidean_distance(pt, site.location) for pt in region]
    avg_dist = sum(distances) / len(distances) if distances else 0.0
    scalar_part = avg_dist * 0.01

    return {**scalar_to_mv(scalar_part), **vector_to_mv(demand_vec)}


# ----------------------------------------------------------------------
# Core allocation routine – deeper integration
# ----------------------------------------------------------------------
def allocate_resources(
    sites: List[Site],
    points: List[Point],
    dim: int,
    demand_generator: Callable[[Site, List[Point], int], Multivector] = generate_demand,
) -> List[Site]:
    """
    1. Compute Voronoi partition based on current site locations.
    2. For each site:
        * If breaker permits, generate a demand multivector and update
          the site’s resource via the geometric product.
        * Record success/failure accordingly.
        * Optionally move the site slightly towards the resulting vector
          part of its resource (feedback loop).
    3. Return the updated list of sites.
    """
    site_locations = [s.location for s in sites]
    regions = compute_voronoi_regions(points, site_locations)

    updated_sites: List[Site] = []
    for site in sites:
        region = regions.get(site.index, [])
        if site.breaker.allow():
            demand = demand_generator(site, region, dim)
            new_resource = geometric_product(site.resource, demand)
            site.resource = new_resource
            site.breaker.record_success()
        else:
            site.breaker.record_failure()

        # Feedback: shift location by a fraction of the vector component
        vec_part = mv_to_vector(site.resource, dim)
        if np.linalg.norm(vec_part) > 0:
            shift = (vec_part[:2] / np.linalg.norm(vec_part[:2])) * 0.01
            new_x = site.location[0] + shift[0]
            new_y = site.location[1] + shift[1]
            site.location = (new_x, new_y)

        updated_sites.append(site)

    return updated_sites


# ----------------------------------------------------------------------
# Example driver (can be removed or adapted)
# ----------------------------------------------------------------------
def _example_run() -> None:
    random.seed(42)
    np.random.seed(42)

    # initialise sites
    sites = [
        Site(index=i, location=(random.uniform(0, 10), random.uniform(0, 10)))
        for i in range(5)
    ]

    # random request points
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(200)]

    # run allocation cycles
    for _ in range(10):
        sites = allocate_resources(sites, points, dim=5)

    # output final state
    for s in sites:
        print(s.as_dict())


if __name__ == "__main__":
    _example_run()