# DARWIN HAMMER — match 5662, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s0.py (gen6)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (gen4)
# born: 2026-05-30T00:04:11Z

"""Hybrid Fusion of:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m468_s1.py (regret‑weighted Hoeffding tree with Gini‑modulated bandit)
- hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (Voronoi region partitioning with geometric‑algebra utilities)

Mathematical bridge:
The Gini coefficient computed over a set of `MathAction` objects quantifies the inequality of their expected values.
We use this scalar as a global weighting factor that modulates:
1. The confidence bound of a bandit arm (`BanditAction.confidence_bound`), scaling the classic Hoeffding‑style term.
2. The influence of each action when computing weighted centroids of Voronoi regions, thereby coupling the tree‑based regret signal to the spatial routing performed by the ternary router.

The resulting hybrid system can route actions to spatial sites, evaluate bandit confidence with Gini‑adjusted bounds, and react to repeated high‑confidence failures via an endpoint circuit‑breaker."""


import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict, FrozenSet
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
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
# Utility functions (Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def calculate_gini(actions: List[MathAction]) -> float:
    """Calculate the Gini coefficient for a list of actions."""
    if not actions:
        return 0.0
    values = [a.expected_value for a in actions]
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = 0.0
    gini = 0.0
    for i, val in enumerate(sorted_vals, start=1):
        cumulative += val
        gini += (2 * i - len(values) - 1) * val
    return gini / (len(actions) * total)


def bandit_confidence_from_gini(gini: float, propensity: float, n: int = 1) -> float:
    """
    Hoeffding‑style confidence bound scaled by the Gini coefficient.
    Larger inequality (higher Gini) widens the bound, encouraging exploration.
    """
    if propensity <= 0.0:
        propensity = 1e-12
    # classic bound: sqrt( (log(1/propensity)) / (2 * n) )
    base = math.sqrt(math.log(1.0 / propensity) / (2.0 * max(n, 1)))
    return base * (1.0 + gini)  # linear scaling with Gini


# ----------------------------------------------------------------------
# Geometric / Voronoi utilities (Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient


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


def weighted_centroid(region: List[Point], weight: float) -> Point:
    """
    Compute the centroid of a region weighted by a scalar.
    If the region is empty, returns (0.0, 0.0).
    """
    if not region:
        return (0.0, 0.0)
    xs, ys = zip(*region)
    cx = sum(x * weight for x in xs) / (weight * len(region))
    cy = sum(y * weight for y in ys) / (weight * len(region))
    return (cx, cy)


# ----------------------------------------------------------------------
# Circuit breaker (Parent B)
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
        """ISO‑8601 timestamp in UTC."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = self._now_z()

    def allow(self) -> bool:
        """Return True if the circuit is closed (operations permitted)."""
        return not self.open


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def map_action_to_point(action: MathAction, domain: Tuple[float, float, float, float]) -> Point:
    """
    Deterministically map an action identifier to a point inside a rectangular domain.
    The domain is given as (xmin, xmax, ymin, ymax).
    """
    xmin, xmax, ymin, ymax = domain
    h = _hash(0xDEADBEEF, action.id)
    # Normalize hash to [0,1)
    rnd = (h % (10 ** 9)) / 1e9
    x = xmin + rnd * (xmax - xmin)
    y = ymin + rnd * (ymax - ymin)
    return (x, y)


def compute_weighted_gini_voronoi(actions: List[MathAction],
                                 points: List[Point],
                                 sites: List[Point]) -> Dict[int, Tuple[Point, float]]:
    """
    Fuse the Gini coefficient with Voronoi partitioning.
    For each Voronoi region we compute a Gini‑weighted centroid.
    Returns a mapping ``site_index → (centroid, region_gini)``.
    """
    # Global Gini over all actions
    global_gini = calculate_gini(actions)

    # Assign each point to a site
    regions = compute_voronoi_regions(points, sites)

    # Compute a region‑specific Gini using the expected values of actions whose
    # mapped points fall inside the region.
    region_info: Dict[int, Tuple[Point, float]] = {}
    # Pre‑map actions to points (using a fixed domain)
    domain = (0.0, 1.0, 0.0, 1.0)
    action_points = {a.id: map_action_to_point(a, domain) for a in actions}

    for site_idx, region_pts in regions.items():
        # Gather actions whose points are inside this region
        actions_in_region = [
            a for a in actions
            if any(euclidean_distance(action_points[a.id], rp) < 1e-9 for rp in region_pts)
        ]
        region_gini = calculate_gini(actions_in_region) if actions_in_region else 0.0
        # Weight the centroid by (1 + global_gini) to let the overall inequality
        # influence spatial decisions.
        weight = 1.0 + global_gini
        centroid = weighted_centroid(region_pts, weight)
        region_info[site_idx] = (centroid, region_gini)

    return region_info


def select_bandit_action(action: MathAction,
                         site_idx: int,
                         gini: float,
                         breaker: EndpointCircuitBreaker) -> BanditAction:
    """
    Create a `BanditAction` for a given `MathAction` after routing it to a site.
    The confidence bound uses the Gini‑scaled Hoeffding term.
    If the circuit breaker is open, the confidence is inflated to discourage selection.
    """
    # Propensity is inversely related to the site index (just for illustration)
    propensity = 1.0 / (site_idx + 1)
    confidence = bandit_confidence_from_gini(gini, propensity)

    if not breaker.allow():
        # Penalise confidence when the breaker is tripped
        confidence *= 2.0

    return BanditAction(
        action_id=action.id,
        propensity=propensity,
        expected_reward=action.expected_value,
        confidence_bound=confidence,
    )


def hybrid_decision_pipeline(actions: List[MathAction],
                             points: List[Point],
                             sites: List[Point],
                             breaker: EndpointCircuitBreaker) -> List[BanditAction]:
    """
    End‑to‑end hybrid pipeline:
    1. Compute global Gini.
    2. Partition space via Voronoi.
    3. Route each action to the nearest site (using hashed point).
    4. Produce a `BanditAction` with Gini‑adjusted confidence.
    5. Update the circuit breaker based on a simple threshold on confidence.
    """
    global_gini = calculate_gini(actions)
    region_map = compute_voronoi_regions(points, sites)

    # Map actions to points once
    domain = (0.0, 1.0, 0.0, 1.0)
    action_pts = {a.id: map_action_to_point(a, domain) for a in actions}

    bandit_actions: List[BanditAction] = []

    for action in actions:
        # Find nearest site for the action's point
        pt = action_pts[action.id]
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest_site = int(np.argmin(distances))

        # Create bandit arm
        ba = select_bandit_action(action, nearest_site, global_gini, breaker)
        bandit_actions.append(ba)

        # Simple failure logic: if confidence exceeds 1.5, treat as a failure
        if ba.confidence_bound > 1.5:
            breaker.record_failure()
        else:
            breaker.record_success()

    return bandit_actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic actions
    actions = [
        MathAction(id=f"a{i}", expected_value=random.uniform(0, 10))
        for i in range(10)
    ]

    # Random points in unit square
    points = [(random.random(), random.random()) for _ in range(30)]

    # Fixed sites (e.g., three routers)
    sites = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]

    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run the hybrid pipeline
    bandit_arms = hybrid_decision_pipeline(actions, points, sites, breaker)

    # Print a concise summary
    for arm in bandit_arms:
        print(
            f"Action {arm.action_id} → propensity {arm.propensity:.3f}, "
            f"expected {arm.expected_reward:.2f}, confidence {arm.confidence_bound:.3f}"
        )
    print(f"Circuit breaker open: {not breaker.allow()}, failures: {breaker.failures}")