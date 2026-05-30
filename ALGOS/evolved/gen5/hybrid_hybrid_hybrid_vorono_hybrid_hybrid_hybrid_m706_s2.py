# DARWIN HAMMER — match 706, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s5.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s1.py (gen4)
# born: 2026-05-29T23:30:29Z

"""Hybrid Voronoi‑Epistemic Partitioning

This module fuses two parent algorithms:

* **Parent A** – geometric Voronoi partitioning with a circuit‑breaker.
* **Parent B** – epistemic‑certainty handling via ``CertaintyFlag`` objects and a
  simple bandit‑style labeling/sketching system.

**Mathematical bridge** – Both parents treat *uncertainty* as a scalar that
modulates a core operation.  In the Voronoi case the scalar is the Euclidean
distance to a site; in the epistemic case it is the confidence (basis points)
attached to a ``CertaintyFlag``.  By scaling the geometric distance with the
inverse confidence we obtain a *weighted distance*:


d̂(i, p) = ‖p – s_i‖ · (1 – confidence_i / 10_000)


where ``s_i`` is site *i* and ``confidence_i`` is the confidence of the
epistemic flag associated with that site.  The point ``p`` is assigned to the
site that minimises ``d̂``.  This yields a unified partition that respects both
spatial proximity and epistemic certainty.

The module provides three high‑level functions demonstrating the hybrid
behaviour and a lightweight circuit‑breaker that can halt further processing
if too many contradictory assignments occur.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable, Callable

import numpy as np

# ----------------------------------------------------------------------
# Core Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = FrozenSet[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Epistemic certainty helpers (Parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable description of epistemic certainty."""
    label: str
    confidence_bps: int               # 0 … 10 000 basis points = 0 % … 100 %
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", _now_z())

# ----------------------------------------------------------------------
# Circuit‑breaker (Parent A)
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
        """True if the breaker is closed (operations permitted)."""
        return not self.open

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def _now_z() -> str:
    """ISO‑8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_voronoi_partition(
    points: List[Point],
    sites: List[Point],
    site_flags: List[CertaintyFlag],
) -> Dict[int, List[Point]]:
    """
    Assign each point to the site that minimises the *certainty‑weighted* distance.

    Weighted distance for site i:
        d̂_i(p) = ‖p – s_i‖ · (1 – confidence_i / 10_000)

    Returns a mapping ``site_index → list[points]``.
    """
    if len(sites) != len(site_flags):
        raise ValueError("sites and site_flags must have the same length")

    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        weighted_distances = []
        for s, flag in zip(sites, site_flags):
            base = euclidean_distance(pt, s)
            weight = 1.0 - flag.confidence_bps / 10_000.0
            weighted_distances.append(base * weight)
        nearest = int(np.argmin(weighted_distances))
        regions[nearest].append(pt)
    return regions


def label_points(points: Iterable[Point]) -> Dict[Point, str]:
    """
    Very light‑weight “sketch” that hashes a point to a deterministic label.
    The label mimics a bandit‑style arm identifier.
    """
    labels: Dict[Point, str] = {}
    for pt in points:
        # Convert coordinates to a byte string and hash
        raw = f"{pt[0]:.12f},{pt[1]:.12f}".encode("utf-8")
        h = hashlib.blake2b(raw, digest_size=4).hexdigest()
        labels[pt] = f"arm_{h}"
    return labels


def region_reward_estimate(
    regions: Dict[int, List[Point]],
    point_labels: Dict[Point, str],
    reward_function: Callable[[str], float],
    cert_flags: List[CertaintyFlag],
) -> Dict[int, float]:
    """
    Estimate a mean reward for each Voronoi region.

    Each point contributes its reward weighted by the confidence of the
    region’s associated ``CertaintyFlag``.  The ``reward_function`` maps a
    label (arm) to a scalar reward (e.g. sampled from a stochastic process).
    """
    estimates: Dict[int, float] = {}
    for idx, pts in regions.items():
        if not pts:
            estimates[idx] = 0.0
            continue
        flag = cert_flags[idx]
        weight = flag.confidence_bps / 10_000.0
        rewards = [reward_function(point_labels[p]) * weight for p in pts]
        estimates[idx] = float(np.mean(rewards))
    return estimates


# ----------------------------------------------------------------------
# Example reward generator (stochastic bandit arm)
# ----------------------------------------------------------------------
def simple_reward_generator(seed: int = 0) -> Callable[[str], float]:
    """
    Returns a function that maps an arm label to a pseudo‑random reward.
    The same label always yields the same reward for reproducibility.
    """
    rng = random.Random(seed)

    # Pre‑sample a reward for each possible arm prefix (first 4 hex chars)
    cache: Dict[str, float] = {}

    def reward(arm_label: str) -> float:
        prefix = arm_label.split("_")[1][:4]
        if prefix not in cache:
            # Reward in [0, 1)
            cache[prefix] = rng.random()
        return cache[prefix]

    return reward

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Generate random points and sites
    random.seed(42)
    np.random.seed(42)

    points = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(200)]
    sites = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(5)]

    # 2. Attach a CertaintyFlag to each site (confidence varies)
    site_flags = [
        CertaintyFlag(
            label=random.choice(EPISTEMIC_FLAGS),
            confidence_bps=random.randint(2000, 8000),
            authority_class="auto",
            rationale="synthetic test",
        )
        for _ in sites
    ]

    # 3. Perform weighted Voronoi partition
    regions = weighted_voronoi_partition(points, sites, site_flags)

    # 4. Label points (sketch‑style)
    point_labels = label_points(points)

    # 5. Estimate rewards per region
    reward_fn = simple_reward_generator(seed=123)
    estimates = region_reward_estimate(regions, point_labels, reward_fn, site_flags)

    # 6. Simple circuit‑breaker usage: fail if any region has < 2 points
    breaker = EndpointCircuitBreaker(failure_threshold=2)
    for idx, pts in regions.items():
        if len(pts) < 2:
            breaker.record_failure()
        else:
            breaker.record_success()
    status = "OPEN" if not breaker.allow() else "CLOSED"
    print(f"Circuit‑breaker status: {status}")
    print("Region reward estimates:")
    for idx, val in estimates.items():
        print(f"  Site {idx} (confidence {site_flags[idx].confidence_bps/100:.1f}%): {val:.4f}")