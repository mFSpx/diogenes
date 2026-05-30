# DARWIN HAMMER — match 3816, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_fractional_hdc_m2289_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s1.py (gen3)
# born: 2026-05-29T23:51:46Z

"""
Hybrid Algorithm: hybrid_hybrid_hybrid_hybrid_fractional_hdc_m2289_s0 + hybrid_hybrid_hybrid_geomet_hybrid_rbf_surrogate_m409_s1
Mathematical Bridge
-------------------
Parent A provides a circuit‑breaker together with a *Morphology* description of an
endpoint.  The haversine distance is used to measure the angular separation of
two morphologies after mapping the three size dimensions (length, width,
height) to spherical coordinates via the *sphericity_index*.

Parent B supplies a radial‑basis‑function (RBF) similarity kernel and a
geometric‑algebra *Multivector* type whose geometric product can be modulated
by a scalar weight.

The fusion proceeds as follows:

1. Two endpoints are mapped to points on a unit sphere using their
   *sphericity_index* and the normalized length/width as latitude/longitude.
2. The haversine distance between the two points is computed.
3. An RBF kernel converts that distance into a similarity weight `w ∈ (0,1]`.
4. The weight `w` scales the geometric product of the endpoints’ multivectors,
   yielding a *weighted geometric product* that simultaneously reflects
   morphological similarity, circuit‑breaker health, and hyper‑dimensional
   binding.

The module implements three high‑level functions that showcase this hybrid
behaviour and a small smoke‑test under ``if __name__ == "__main__"``.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – circuit breaker & morphology utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    return round(float(value), 6)


class EndpointCircuitBreaker:
    """Simple circuit‑breaker tracking consecutive failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint may receive work)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0,1]."""
        return min(self.failures / self.failure_threshold, 1.0)


@dataclass
class Morphology:
    """Geometric description of an endpoint."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximal dimension, ∈ (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)


def haversine_distance(m1: Morphology, m2: Morphology) -> float:
    """
    Treat each morphology as a point on a sphere:

    * latitude  = asin( sphericity_index )
    * longitude = atan2(width, length)   (both normalised)

    The sphere radius is taken as the average of the two masses so that
    heavier endpoints are slightly farther apart in Euclidean space.
    """
    # Normalise length/width to avoid overflow in atan2
    phi1 = math.asin(sphericity_index(m1.length, m1.width, m1.height))
    phi2 = math.asin(sphericity_index(m2.length, m2.width, m2.height))

    lambda1 = math.atan2(m1.width, m1.length)
    lambda2 = math.atan2(m2.width, m2.length)

    # Haversine formula
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    c = 2 * math.asin(min(1.0, math.sqrt(a)))  # central angle

    # Use mean mass as a surrogate radius
    R = (m1.mass + m2.mass) / 2.0
    return R * c


# ----------------------------------------------------------------------
# Parent B – RBF similarity & geometric algebra utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def rbf_kernel(dist: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(- (epsilon * dist) ** 2)


class Multivector:
    """
    Very light‑weight geometric‑algebra multivector.

    Internally a dictionary maps a frozenset of basis indices to a scalar.
    For example, the scalar part uses ``frozenset()`` as key, a vector e1 uses
    ``frozenset({1})``, a bivector e12 uses ``frozenset({1,2})`` and so on.
    """
    def __init__(self, components: Dict[frozenset, float] | None = None, n: int = 3):
        self.n = int(n)
        self.components: Dict[frozenset, float] = {}
        if components:
            for blade, coeff in components.items():
                if coeff != 0.0:
                    self.components[frozenset(blade)] = float(coeff)

    def grade(self, k: int) -> "Multivector":
        """Return the part of grade *k*."""
        return Multivector(
            {blade: c for blade, c in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coeff in sorted(self.components.items(),
                                   key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coeff:+.6g}*{label}")
        return " + ".join(terms)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Naïve geometric product:
    - concatenate blade indices,
    - cancel duplicate indices (they square to +1),
    - ignore sign changes from reordering (acceptable for a prototype).
    """
    result: Dict[frozenset, float] = {}
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            combined = list(blade_a) + list(blade_b)
            # Cancel pairs of identical indices
            counts = {}
            for idx in combined:
                counts[idx] = counts.get(idx, 0) + 1
            reduced = [idx for idx, cnt in counts.items() if cnt % 2 == 1]
            new_blade = frozenset(sorted(reduced))
            result[new_blade] = result.get(new_blade, 0.0) + coeff_a * coeff_b
    return Multivector(result, a.n)


def weighted_geometric_product(a: Multivector, b: Multivector, weight: float) -> Multivector:
    """
    Scale the geometric product by a scalar weight.
    The weight can encode similarity, circuit‑breaker health, etc.
    """
    prod = geometric_product(a, b)
    scaled = {blade: coeff * weight for blade, coeff in prod.components.items()}
    return Multivector(scaled, a.n)


# ----------------------------------------------------------------------
# Fusion layer – three demonstrative functions
# ----------------------------------------------------------------------
def endpoint_factory(name: str,
                     length: float,
                     width: float,
                     height: float,
                     mass: float,
                     failure_threshold: int = 3) -> Dict[str, Any]:
    """
    Build a dictionary that bundles:
    - a circuit breaker,
    - a morphology,
    - a multivector encoding the morphology (mass as scalar, dimensions as vector).
    """
    cb = EndpointCircuitBreaker(failure_threshold)
    morph = Morphology(length, width, height, mass)

    # Encode morphology into a multivector:
    # scalar part = mass,
    # e1 = length, e2 = width, e3 = height
    components = {
        frozenset(): mass,
        frozenset({1}): length,
        frozenset({2}): width,
        frozenset({3}): height,
    }
    mv = Multivector(components, n=3)

    return {"name": name, "circuit": cb, "morph": morph, "mv": mv}


def hybrid_interaction(ep1: Dict[str, Any], ep2: Dict[str, Any],
                       epsilon: float = 1.0) -> Multivector:
    """
    Compute the weighted geometric product between two endpoints.

    Steps:
    1. Morphological haversine distance → d.
    2. RBF kernel on d → similarity weight w.
    3. Blend circuit‑breaker health:
       w ← w * (1 - avg_failure_rate)
    4. Return w‑scaled geometric product of the underlying multivectors.
    """
    # 1. Morphological distance
    d = haversine_distance(ep1["morph"], ep2["morph"])

    # 2. RBF similarity
    w = rbf_kernel(d, epsilon)

    # 3. Incorporate circuit‑breaker health
    fr1 = ep1["circuit"].failure_rate()
    fr2 = ep2["circuit"].failure_rate()
    health_factor = 1.0 - (fr1 + fr2) / 2.0
    w *= health_factor

    # 4. Weighted geometric product
    return weighted_geometric_product(ep1["mv"], ep2["mv"], w)


def network_hybrid_aggregation(endpoints: List[Dict[str, Any]],
                               seed_count: int = 2,
                               epsilon: float = 1.0) -> Dict[int, Multivector]:
    """
    Treat each endpoint as a 2‑D point (length, width) and partition them using
    the *assign* routine from Parent B.  For each region compute the cumulative
    hybrid interaction of the region’s members with the region’s first member
    (as a simple prototype of region‑wise binding).
    """
    # Extract 2‑D points for clustering
    points: List[Point] = [(e["morph"].length, e["morph"].width) for e in endpoints]

    # Randomly pick seeds (ensure reproducibility)
    random.seed(0)
    seeds_idx = random.sample(range(len(points)), min(seed_count, len(points)))
    seeds = [points[i] for i in seeds_idx]

    # --- Parent B's assign logic -------------------------------------------------
    def nearest(point: Point, seeds_local: List[Point]) -> int:
        return min(
            range(len(seeds_local)),
            key=lambda i: (euclidean_distance(point, seeds_local[i]), i),
        )

    def assign(points_local: List[Point], seeds_local: List[Point]) -> Dict[int, List[int]]:
        regions: Dict[int, List[int]] = {i: [] for i in range(len(seeds_local))}
        for idx, p in enumerate(points_local):
            regions[nearest(p, seeds_local)].append(idx)
        return regions

    regions = assign(points, seeds)

    # --- Compute region‑wise hybrid aggregation ---------------------------------
    region_mv: Dict[int, Multivector] = {}
    for region_id, idx_list in regions.items():
        if not idx_list:
            continue
        # Use the first endpoint as the "anchor"
        anchor = endpoints[idx_list[0]]
        agg = Multivector({frozenset(): 0.0}, n=3)
        for idx in idx_list[1:]:
            partner = endpoints[idx]
            inter = hybrid_interaction(anchor, partner, epsilon)
            # Simple additive accumulation
            for blade, coeff in inter.components.items():
                agg.components[blade] = agg.components.get(blade, 0.0) + coeff
        region_mv[region_id] = agg
    return region_mv


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a handful of synthetic endpoints
    eps = [
        endpoint_factory("A", length=1.2, width=0.8, height=0.5, mass=2.0),
        endpoint_factory("B", length=1.0, width=1.1, height=0.6, mass=1.8),
        endpoint_factory("C", length=0.9, width=0.7, height=0.4, mass=2.2),
        endpoint_factory("D", length=1.3, width=0.9, height=0.55, mass=1.9),
    ]

    # Simulate a few failures to see health‑factor influence
    eps[1]["circuit"].record_failure()
    eps[2]["circuit"].record_failure()
    eps[2]["circuit"].record_failure()  # two failures for C

    # Pairwise hybrid interactions
    print("Pairwise hybrid interactions:")
    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            mv = hybrid_interaction(eps[i], eps[j], epsilon=0.5)
            print(f"{eps[i]['name']} ↔ {eps[j]['name']}: {mv}")

    # Region‑wise aggregation
    agg = network_hybrid_aggregation(eps, seed_count=2, epsilon=0.5)
    print("\nRegion‑wise aggregated multivectors:")
    for rid, mv in agg.items():
        print(f"Region {rid}: {mv}")

    sys.exit(0)