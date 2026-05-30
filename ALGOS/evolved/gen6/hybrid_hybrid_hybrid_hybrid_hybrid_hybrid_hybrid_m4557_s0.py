# DARWIN HAMMER — match 4557, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m563_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s4.py (gen3)
# born: 2026-05-29T23:56:29Z

"""Hybrid Voronoi‑Tropical‑Sketch Algorithm

Parents:
    - hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m563_s1.py
    - hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s4.py

Mathematical bridge:
Both parents manipulate collections of vectors with matrix‑like operations.
Parent A clusters 2‑D points into Voronoi regions and can compute region centroids.
Parent B defines a tropical (max‑plus) semiring where “addition” is max and
“multiplication” is ordinary addition, together with a tropical matrix product
and a Count‑Min sketch for lossy frequency estimation.

The fusion treats each Voronoi centroid **c**∈ℝ² as a row vector of a matrix **C**.
A user‑provided weight matrix **W** (shape (2, k)) is multiplied with **C** using the
tropical matrix product **C ⊗ W**, yielding a transformed score matrix **S**.
Each entry of **S** is then fed to a Count‑Min sketch, providing an approximate
histogram of tropical scores while an endpoint circuit‑breaker guards the pipeline
against excessive failures.

The three core functions below expose this hybrid workflow:
    1. `voronoi_assign` – cluster points and return region‑wise point lists.
    2. `tropical_transform` – compute centroids and apply the tropical matrix product.
    3. `sketch_tropical_scores` – compress the tropical scores with a Count‑Min sketch
       while respecting the circuit‑breaker state.

All components are pure NumPy / std‑lib and can be combined arbitrarily.
"""

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A (Voronoi / CircuitBreaker)
# ----------------------------------------------------------------------


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    """Index of the nearest seed to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (_euclidean(point, seeds[i]), i))


def voronoi_assign(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Assign each point to its nearest seed, returning a dict ``region → points``.
    Mirrors ``assign`` from Parent A.
    """
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[_nearest(p, seeds)].append(p)
    return regions


def region_centroids(
    assignments: Dict[int, List[Tuple[float, float]]]
) -> np.ndarray:
    """
    Compute the centroid of every non‑empty region.
    Returns an ``(n_regions, 2)`` array; empty regions are filled with ``nan``.
    """
    n = len(assignments)
    centroids = np.empty((n, 2), dtype=float)
    for i in range(n):
        pts = assignments[i]
        if pts:
            arr = np.asarray(pts, dtype=float)
            centroids[i] = arr.mean(axis=0)
        else:
            centroids[i] = np.nan
    return centroids


class EndpointCircuitBreaker:
    """
    Simple circuit‑breaker: after ``failure_threshold`` consecutive failures
    the breaker opens and ``allow()`` returns ``False``.
    """

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at: datetime | None = None

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = None

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc)

    def allow(self) -> bool:
        return not self.open


# ----------------------------------------------------------------------
# Tropical semiring utilities (Parent B)
# ----------------------------------------------------------------------


class Tropical:
    """Max‑plus (tropical) semiring."""

    @staticmethod
    def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical addition = element‑wise maximum."""
        return np.maximum(x, y)

    @staticmethod
    def mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Tropical multiplication = ordinary addition."""
        return np.add(x, y)

    @staticmethod
    def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Tropical matrix multiplication.
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
        """
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Both A and B must be 2‑D matrices")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Inner dimensions must agree")
        # Broadcast to compute all pairwise sums over k
        # A: (i, k) → (i, k, 1); B: (k, j) → (1, k, j)
        A_exp = A[:, :, np.newaxis]          # shape (i, k, 1)
        B_exp = B[np.newaxis, :, :]          # shape (1, k, j)
        # Sum over k then max
        return np.max(A_exp + B_exp, axis=1)  # shape (i, j)


# ----------------------------------------------------------------------
# Count‑Min sketch (simplified version from Parent B)
# ----------------------------------------------------------------------


def _hash(item: bytes, seed: int) -> int:
    """Deterministic 32‑bit hash mixing the seed."""
    h = int.from_bytes(
        hashlib.sha256(item + seed.to_bytes(4, "little")).digest()[:4],
        "little",
        signed=False,
    )
    return h


def count_min_sketch(
    items: Iterable[bytes],
    width: int = 64,
    depth: int = 4,
) -> np.ndarray:
    """
    Classic Count‑Min sketch.
    Returns a ``depth × width`` integer array where each row corresponds to a hash
    function. The sketch can be queried with ``estimate(item)`` (implemented as a
    closure below).
    """
    sketch = np.zeros((depth, width), dtype=np.int64)

    for idx in range(depth):
        seed = idx + 12345  # simple deterministic seeds
        for it in items:
            col = _hash(it, seed) % width
            sketch[idx, col] += 1

    def estimate(item: bytes) -> int:
        mins = []
        for idx in range(depth):
            seed = idx + 12345
            col = _hash(item, seed) % width
            mins.append(sketch[idx, col])
        return min(mins)

    # Attach the estimator as an attribute for convenience
    sketch.estimate = estimate  # type: ignore
    return sketch


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def tropical_transform(
    assignments: Dict[int, List[Tuple[float, float]]],
    weight_matrix: np.ndarray,
) -> np.ndarray:
    """
    1. Compute centroids of Voronoi regions.
    2. Treat the centroid matrix **C** (n×2) as a tropical matrix.
    3. Return the tropical product **C ⊗ W**, where **W** is user supplied.

    Empty regions (centroid = NaN) are replaced with ``-inf`` so that they never
    dominate the max operation.
    """
    centroids = region_centroids(assignments)  # (n, 2)

    # Replace NaN with -inf (tropical additive identity)
    centroids = np.where(np.isnan(centroids), -np.inf, centroids)

    # Tropical multiplication expects ordinary addition, so we can feed centroids
    # directly to Tropical.matmul.
    return Tropical.matmul(centroids, weight_matrix)


def sketch_tropical_scores(
    scores: np.ndarray,
    breaker: EndpointCircuitBreaker,
    width: int = 128,
    depth: int = 5,
) -> np.ndarray:
    """
    Convert each tropical score to a byte representation and feed it into a
    Count‑Min sketch while respecting the circuit‑breaker.
    If the breaker is open, the function returns an empty sketch.
    """
    if not breaker.allow():
        # Early exit – breaker disallows further work
        return np.zeros((depth, width), dtype=np.int64)

    # Flatten scores and turn each float into a canonical byte string
    flat = scores.ravel()
    items = (np.float64(v).tobytes() for v in flat)

    try:
        sketch = count_min_sketch(items, width=width, depth=depth)
        breaker.record_success()
        return sketch
    except Exception:
        breaker.record_failure()
        return np.zeros((depth, width), dtype=np.int64)


def hybrid_process(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    weight_matrix: np.ndarray,
    breaker: EndpointCircuitBreaker | None = None,
) -> Tuple[Dict[int, List[Tuple[float, float]]], np.ndarray, np.ndarray]:
    """
    End‑to‑end pipeline:
        1. Voronoi assignment of *points* to *seeds*.
        2. Tropical transformation of region centroids with *weight_matrix*.
        3. Count‑Min sketch of the resulting tropical scores.

    Returns a tuple ``(assignments, tropical_scores, sketch)``.
    """
    if breaker is None:
        breaker = EndpointCircuitBreaker()

    # Step 1 – Voronoi clustering
    assignments = voronoi_assign(points, seeds)

    # Step 2 – Tropical linear map
    tropical_scores = tropical_transform(assignments, weight_matrix)

    # Step 3 – Sketching
    sketch = sketch_tropical_scores(tropical_scores, breaker)

    return assignments, tropical_scores, sketch


# ----------------------------------------------------------------------
# Helper: generate synthetic data for the smoke test
# ----------------------------------------------------------------------


def _random_points(n: int, low: float = -100.0, high: float = 100.0) -> List[Tuple[float, float]]:
    return [(random.uniform(low, high), random.uniform(low, high)) for _ in range(n)]


def _random_seeds(k: int) -> List[Tuple[float, float]]:
    return _random_points(k)


def _random_weight_matrix(rows: int = 2, cols: int = 3) -> np.ndarray:
    # Tropical matrices often contain -inf as the additive identity; we keep finite numbers
    return np.random.uniform(-10, 10, size=(rows, cols))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(0)

    # Generate synthetic inputs
    pts = _random_points(500)
    sd = _random_seeds(7)
    W = _random_weight_matrix(rows=2, cols=4)

    # Run the hybrid pipeline
    cb = EndpointCircuitBreaker(failure_threshold=2)
    assign_dict, trop_scores, cms = hybrid_process(pts, sd, W, breaker=cb)

    # Simple sanity checks (will raise if shapes mismatch)
    assert isinstance(assign_dict, dict)
    assert trop_scores.shape == (len(sd), W.shape[1])
    assert cms.shape[0] == 5 and cms.shape[1] == 128

    # Demonstrate sketch query on a few scores
    sample_bytes = np.float64(trop_scores[0, 0]).tobytes()
    estimate = cms.estimate(sample_bytes)  # type: ignore
    print(f"Sample tropical score: {trop_scores[0,0]:.3f}, sketch estimate: {estimate}")

    # Report circuit‑breaker status
    print(f"Circuit breaker open? {not cb.allow()}")
    sys.exit(0)