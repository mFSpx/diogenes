# DARWIN HAMMER — match 5662, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1203_s0.py (gen6)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (gen4)
# born: 2026-05-30T00:04:11Z

import hashlib
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, FrozenSet, Any

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]               # basis blade as set of basis indices
Multivector = Dict[Blade, float]     # mapping blade → coefficient


# ----------------------------------------------------------------------
# Utility – deterministic hashing
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# ----------------------------------------------------------------------
# 1️⃣ Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def calculate_gini(actions: List["MathAction"]) -> float:
    if not actions:
        return 0.0
    values = sorted(a.expected_value for a in actions)
    n = len(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(values))
    total = sum(values)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


# ----------------------------------------------------------------------
# 2️⃣ Geometric Algebra helpers (deepened integration)
# ----------------------------------------------------------------------
def _blade_to_index(blade: Blade, dim: int) -> int:
    """Map a blade to a unique index in [0, 2**dim)."""
    idx = 0
    for i in blade:
        if i < 0 or i >= dim:
            raise ValueError("Blade index out of range")
        idx |= 1 << i
    return idx


def multivector_to_dense(mv: Multivector, dim: int) -> np.ndarray:
    """Convert a multivector to a dense vector of length 2**dim."""
    size = 1 << dim
    vec = np.zeros(size, dtype=float)
    for blade, coeff in mv.items():
        idx = _blade_to_index(blade, dim)
        vec[idx] = coeff
    return vec


def dense_to_multivector(vec: np.ndarray, dim: int) -> Multivector:
    """Convert a dense vector back to a multivector representation."""
    mv: Multivector = {}
    for idx, coeff in enumerate(vec):
        if coeff != 0.0:
            blade = frozenset(i for i in range(dim) if (idx >> i) & 1)
            mv[blade] = coeff
    return mv


def bilinear_form_projection(mv: Multivector, matrix: np.ndarray) -> float:
    """
    Project a multivector onto a scalar using a symmetric bilinear form.
    The matrix must be square of size 2**dim, where dim is the underlying
    geometric algebra dimension.
    """
    dim = int(math.log2(matrix.shape[0]))
    if matrix.shape != (1 << dim, 1 << dim):
        raise ValueError("Matrix size must be 2**dim × 2**dim")
    v = multivector_to_dense(mv, dim)
    return float(v @ matrix @ v)


def geometric_metric_matrix(dim: int) -> np.ndarray:
    """
    Build a simple Euclidean metric matrix for the algebra G(dim,0).
    For a scalar product we use the identity; for a more expressive
    bilinear form one could weight grades differently.
    """
    size = 1 << dim
    return np.identity(size, dtype=float)


# ----------------------------------------------------------------------
# 3️⃣ Data structures (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# 4️⃣ Voronoi helpers (Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        dists = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(dists))
        regions[nearest].append(pt)
    return regions


# ----------------------------------------------------------------------
# 5️⃣ Circuit breaker (Parent B)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def _now_z(self) -> str:
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
        return not self.open


# ----------------------------------------------------------------------
# 6️⃣ Deepened Hybrid Engine
# ----------------------------------------------------------------------
class HybridEngine:
    """
    Encapsulates the fused regret‑weighted bandit logic with geometric‑algebraic
    Voronoi reasoning.  The engine maintains a metric matrix, a circuit breaker
    and offers a single ``step`` method that updates confidence bounds,
    perturbs Voronoi sites in a value‑aware manner and propagates multivector
    information back to the bandit layer.
    """

    def __init__(self,
                 dim: int = 2,
                 base_sites: List[Point] | None = None,
                 failure_threshold: int = 3):
        self.dim = dim
        self.metric = geometric_metric_matrix(dim)
        self.breaker = EndpointCircuitBreaker(failure_threshold)
        self.base_sites = base_sites if base_sites is not None else [(0.0, 0.0), (1.0, 1.0)]

    # ------------------------------------------------------------------
    # Site weighting – deterministic, value‑driven (replaces random bias)
    # ------------------------------------------------------------------
    def weighted_voronoi_sites(self,
                               actions: List[MathAction],
                               scale_factor: float = 1.0) -> List[Point]:
        """
        Shift each base site towards the centroid of actions that are
        *closest* to it, weighted by expected value.  This creates a
        deterministic, value‑aware perturbation.
        """
        if not actions:
            return self.base_sites.copy()

        # Assign actions to nearest site (using expected_value as a proxy for
        # spatial location – we map the scalar to a 2‑D pseudo‑point).
        pseudo_points = [
            (math.sin(hash(a.id) % (2 * math.pi)),
             math.cos(hash(a.id) % (2 * math.pi)))
            for a in actions
        ]
        regions = compute_voronoi_regions(pseudo_points, self.base_sites)

        new_sites: List[Point] = []
        for i, site in enumerate(self.base_sites):
            assigned_actions = [actions[j] for j, pt in enumerate(pseudo_points)
                                if i == int(np.argmin([euclidean_distance(pt, s)
                                                       for s in self.base_sites]))]
            if not assigned_actions:
                new_sites.append(site)
                continue

            total_weight = sum(a.expected_value for a in assigned_actions)
            if total_weight == 0:
                new_sites.append(site)
                continue

            # Compute weighted centroid of the pseudo‑points
            cx = sum(p[0] * a.expected_value for p, a in zip(pseudo_points, assigned_actions)) / total_weight
            cy = sum(p[1] * a.expected_value for p, a in zip(pseudo_points, assigned_actions)) / total_weight

            # Shift original site towards centroid
            dx = (cx - site[0]) * scale_factor
            dy = (cy - site[1]) * scale_factor
            new_sites.append((site[0] + dx, site[1] + dy))
        return new_sites

    # ------------------------------------------------------------------
    # Confidence bound that incorporates geometric projection
    # ------------------------------------------------------------------
    def hybrid_confidence_bound(self,
                                bandit: BanditAction,
                                gini: float,
                                proj_scalar: float,
                                exploration_coef: float = 1.0) -> float:
        """
        Classic Hoeffding bound scaled by (1+gini) and by the geometric
        projection magnitude (normalized to [0,1]).
        """
        if bandit.propensity <= 0:
            raise ValueError("propensity must be positive")
        n_est = 1.0 / bandit.propensity
        base_cb = math.sqrt((2.0 * math.log(1.0 / bandit.propensity)) / n_est)
        proj_factor = 1.0 + min(max(proj_scalar, 0.0), 1.0)   # clamp
        return base_cb * (1.0 + gini) * proj_factor * exploration_coef

    # ------------------------------------------------------------------
    # One integration step
    # ------------------------------------------------------------------
    def step(self,
             actions: List[MathAction],
             points: List[Point],
             bandits: List[BanditAction],
             scale_factor: float = 0.5,
             exploration_coef: float = 1.0) -> Tuple[List[BanditAction], Dict[int, List[Point]]]:
        """
        Execute a full hybrid iteration:
        1. Compute Gini.
        2. Produce value‑aware Voronoi sites.
        3. Assign points to regions.
        4. For each region build a multivector from the contained MathActions.
        5. Project the multivector using the metric matrix.
        6. Update the corresponding BanditAction confidence bound.
        7. Return updated BanditActions and region mapping.
        """
        if not self.breaker.allow():
            raise RuntimeError("Circuit breaker open – aborting step")

        try:
            gini = calculate_gini(actions)
            sites = self.weighted_voronoi_sites(actions, scale_factor)
            regions = compute_voronoi_regions(points, sites)

            # Build a multivector per region: scalar = sum of expected values,
            # e1 coefficient = sum of costs, e2 = sum of risks, e12 = count.
            updated_bandits: List[BanditAction] = []
            for bandit in bandits:
                # Find region index matching bandit.action_id (assume same ordering)
                region_idx = int(bandit.action_id.split("_")[-1]) % len(sites)
                region_points = regions.get(region_idx, [])

                # Gather associated MathActions via nearest pseudo‑point heuristic
                region_actions = [
                    a for a in actions
                    if int(np.argmin([euclidean_distance(
                        (math.sin(hash(a.id) % (2 * math.pi)),
                         math.cos(hash(a.id) % (2 * math.pi))), s)
                        for s in sites])) == region_idx
                ]

                # Assemble multivector
                mv: Multivector = {
                    frozenset(): sum(a.expected_value for a in region_actions),          # scalar
                    frozenset({0}): sum(a.cost for a in region_actions),                 # e1
                    frozenset({1}): sum(a.risk for a in region_actions),                 # e2
                    frozenset({0, 1}): float(len(region_actions))                        # e12
                }

                proj = bilinear_form_projection(mv, self.metric)
                # Normalise projection by maximal possible (heuristic)
                max_proj = (sum(a.expected_value for a in actions) ** 2) + \
                           (sum(a.cost for a in actions) ** 2) + \
                           (sum(a.risk for a in actions) ** 2) + (len(actions) ** 2)
                norm_proj = proj / max_proj if max_proj != 0 else 0.0

                new_cb = self.hybrid_confidence_bound(bandit, gini, norm_proj, exploration_coef)

                updated_bandits.append(BanditAction(
                    action_id=bandit.action_id,
                    propensity=bandit.propensity,
                    expected_reward=bandit.expected_reward,
                    confidence_bound=new_cb,
                    algorithm=bandit.algorithm
                ))

            self.breaker.record_success()
            return updated_bandits, regions

        except Exception:
            self.breaker.record_failure()
            raise

# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data
    actions = [
        MathAction(id="a1", expected_value=10, cost=2, risk=1),
        MathAction(id="a2", expected_value=5, cost=1, risk=3),
        MathAction(id="a3", expected_value=8, cost=0, risk=2)
    ]
    points = [(0.1, 0.2), (0.8, 0.9), (0.4, 0.5), (0.6, 0.1)]
    bandits = [
        BanditAction(action_id="bandit_0", propensity=0.2, expected_reward=5, confidence_bound=0.0),
        BanditAction(action_id="bandit_1", propensity=0.5, expected_reward=7, confidence_bound=0.0)
    ]

    engine = HybridEngine(dim=2, base_sites=[(0.0, 0.0), (1.0, 1.0)])
    updated_bandits, regions = engine.step(actions, points, bandits)

    for b in updated_bandits:
        print(asdict(b))
    print("Regions:", regions)