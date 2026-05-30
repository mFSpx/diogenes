# DARWIN HAMMER — match 5662, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s0.py (gen6)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (gen4)
# born: 2026-05-30T00:04:11Z

"""Hybrid Fusion of Regret‑Weighted Hoeffding Bandit (Parent A) and Voronoi‑Geometric‑Algebra (Parent B).

Mathematical Bridge
-------------------
* The **Gini coefficient** computed from a set of `MathAction`s (Parent A) quantifies
  the inequality of expected values.  This scalar is used to *scale* the Voronoi
  site coordinates, thereby biasing the spatial partition towards actions with
  higher expected value.
* Each Voronoi region is represented as a **multivector** (Parent B).  A
  **bilinear form projection** – a matrix‑vector product defined in the bandit
  formulation – is applied to the multivector coefficients, producing a scalar
  that modulates the **confidence bound** of the corresponding `BanditAction`.
* The resulting confidence‑adjusted bounds feed back into a simple routing
  decision, completing a closed mathematical loop that fuses both topologies.

The module therefore intertwines statistical regret‑based learning with
geometric‑algebraic spatial reasoning in a single unified system.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Core Types (shared)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient


# ----------------------------------------------------------------------
# Data structures (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action used in the regret‑weighted Hoeffding tree."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit arm with propensity‑adjusted confidence bound."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Utility – deterministic hashing (Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# ----------------------------------------------------------------------
# 1️⃣ Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def calculate_gini(actions: List[MathAction]) -> float:
    """Return the Gini coefficient of the expected values of ``actions``."""
    if not actions:
        return 0.0
    values = [a.expected_value for a in actions]
    values.sort()
    n = len(values)
    cumulative = 0.0
    for i, v in enumerate(values, start=1):
        cumulative += i * v
    total = sum(values)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))  # clamp to [0,1]


# ----------------------------------------------------------------------
# 2️⃣ Bilinear form projection (Parent A)
# ----------------------------------------------------------------------
def bilinear_form_projection(mv: Multivector, matrix: np.ndarray) -> float:
    """
    Project a multivector onto a scalar using a bilinear form.

    The multivector is first converted to a dense vector ``v`` whose length
    equals ``matrix.shape[0]`` (the matrix must be square).  Missing blades are
    treated as zero.  The projection is ``vᵀ @ matrix @ v``.
    """
    dim = matrix.shape[0]
    if dim != matrix.shape[1]:
        raise ValueError("Bilinear matrix must be square")
    v = np.zeros(dim)
    for blade, coeff in mv.items():
        # map each blade (a set of ints) to a deterministic index < dim
        idx = sum(1 << i for i in blade) % dim
        v[idx] += coeff
    return float(v @ matrix @ v)


# ----------------------------------------------------------------------
# 3️⃣ Voronoi helpers (Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


# ----------------------------------------------------------------------
# 4️⃣ Circuit breaker (Parent B)
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

    def _now_z(self) -> str:
        """Return a simple ISO‑like timestamp."""
        from datetime import datetime, timezone
        return datetime.now(tz=timezone.utc).isoformat()

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = self._now_z()

    def allow(self) -> bool:
        """Return ``True`` if the breaker is closed (i.e. operations allowed)."""
        return not self.open


# ----------------------------------------------------------------------
# 5️⃣ Hybrid Functions
# ----------------------------------------------------------------------
def weighted_voronoi_sites(actions: List[MathAction],
                           base_sites: List[Point],
                           scale_factor: float = 1.0) -> List[Point]:
    """
    Produce a new list of Voronoi sites whose positions are biased by the Gini
    coefficient of ``actions``.

    Each base site ``s = (x, y)`` is transformed to:
        s' = s + scale_factor * gini * (rand_x, rand_y)
    where ``rand_x, rand_y`` are pseudo‑random numbers in ``[-1, 1]``.
    """
    gini = calculate_gini(actions)
    rng = random.Random(_hash(0, "site_bias"))
    new_sites: List[Point] = []
    for (x, y) in base_sites:
        dx = rng.uniform(-1, 1) * gini * scale_factor
        dy = rng.uniform(-1, 1) * gini * scale_factor
        new_sites.append((x + dx, y + dy))
    return new_sites


def hybrid_confidence_bound(bandit: BanditAction,
                            gini: float,
                            exploration_coef: float = 1.0) -> float:
    """
    Adjust the classic Hoeffding‑style confidence bound with the Gini coefficient.

    Classic bound (simplified):
        cb = sqrt( (2 * log(1/propensity)) / n )
    where ``n`` is approximated by ``1 / propensity``.
    The hybrid bound multiplies by ``(1 + gini)`` and an optional
    ``exploration_coef``.
    """
    if bandit.propensity <= 0:
        raise ValueError("propensity must be positive")
    n_est = 1.0 / bandit.propensity
    base_cb = math.sqrt((2.0 * math.log(1.0 / bandit.propensity)) / n_est)
    return base_cb * (1.0 + gini) * exploration_coef


def hybrid_route(actions: List[MathAction],
                 points: List[Point],
                 base_sites: List[Point],
                 bilinear_matrix: np.ndarray) -> Dict[int, BanditAction]:
    """
    End‑to‑end hybrid operation:

    1. Compute Gini from ``actions``.
    2. Produce Gini‑biased Voronoi sites.
    3. Partition ``points`` into regions.
    4. For each region build a multivector from its points.
    5. Project the multivector via ``bilinear_form_projection`` to obtain a scalar.
    6. Create a ``BanditAction`` whose confidence bound is adjusted by the scalar
       and the Gini coefficient.

    Returns a mapping ``region_index → BanditAction``.
    """
    # Step 1
    gini = calculate_gini(actions)

    # Step 2
    sites = weighted_voronoi_sites(actions, base_sites, scale_factor=0.5)

    # Step 3
    regions = compute_voronoi_regions(points, sites)

    # Step 4 – build a simple multivector: each point contributes a blade
    # representing its quadrant (encoded as a small integer set).
    def point_to_blade(pt: Point) -> Blade:
        # Encode sign of x and y as bits 0 and 1
        bits = set()
        if pt[0] >= 0:
            bits.add(0)
        if pt[1] >= 0:
            bits.add(1)
        return frozenset(bits)

    result: Dict[int, BanditAction] = {}
    for region_idx, pts in regions.items():
        # Assemble multivector
        mv: Multivector = {}
        for pt in pts:
            blade = point_to_blade(pt)
            mv[blade] = mv.get(blade, 0.0) + 1.0  # count occurrences

        # Step 5 – scalar projection
        proj_scalar = bilinear_form_projection(mv, bilinear_matrix)

        # Step 6 – create a synthetic BanditAction
        # Choose an underlying MathAction (if any) to seed expected reward
        seed_action = actions[region_idx % len(actions)]
        propensity = max(0.01, min(1.0, random.random()))  # avoid zero
        base_bandit = BanditAction(
            action_id=seed_action.id,
            propensity=propensity,
            expected_reward=seed_action.expected_value,
            confidence_bound=0.0,  # placeholder
        )
        cb = hybrid_confidence_bound(base_bandit, gini, exploration_coef=1.0 + proj_scalar)
        result[region_idx] = BanditAction(
            action_id=base_bandit.action_id,
            propensity=base_bandit.propensity,
            expected_reward=base_bandit.expected_reward,
            confidence_bound=cb,
        )
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample actions
    actions = [
        MathAction(id="A", expected_value=10.0),
        MathAction(id="B", expected_value=5.0),
        MathAction(id="C", expected_value=1.0),
    ]

    # Random points in the unit square
    points = [(random.random(), random.random()) for _ in range(200)]

    # Base Voronoi sites (e.g., three fixed anchors)
    base_sites = [(0.2, 0.2), (0.8, 0.3), (0.5, 0.9)]

    # Simple symmetric positive‑definite matrix for bilinear projection
    bilinear_matrix = np.array([[2.0, 0.3],
                                [0.3, 1.5]])

    # Run hybrid routing
    routing = hybrid_route(actions, points, base_sites, bilinear_matrix)

    # Print a concise summary
    print("Hybrid routing results (region → BanditAction):")
    for idx, ba in routing.items():
        print(f"Region {idx}: action={ba.action_id}, "
              f"propensity={ba.propensity:.3f}, "
              f"expected_reward={ba.expected_reward:.2f}, "
              f"confidence_bound={ba.confidence_bound:.4f}")

    # Demonstrate circuit breaker usage
    cb = EndpointCircuitBreaker(failure_threshold=2)
    for i in range(5):
        if i % 2 == 0:
            cb.record_success()
        else:
            cb.record_failure()
        print(f"Step {i}: breaker open? {not cb.allow()}, failures={cb.failures}")

    sys.exit(0)