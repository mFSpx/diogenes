# DARWIN HAMMER — match 5758, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s1.py (gen3)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s2.py (gen1)
# born: 2026-05-30T00:04:31Z

"""
Hybrid module unifying:

- `hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s1` (spans → pheromone decay)
- `hybrid_geometric_product_voronoi_partition_m4_s2` (geometric algebra → Voronoi)

Mathematical bridge:
A text span is represented as a 2‑D grade‑1 multivector **p = start·e₁ + end·e₂**.
The Euclidean distance between two spans is the scalar part of the inner product
⟨p‑q, p‑q⟩, i.e. the standard squared distance in the (start,end) plane.
Pheromone signals attached to a span act as a multiplicative weight on this distance:
`d_weighted = d * (1 / (1 + signal_value))`.  
Thus the entropy‑gain intuition of the original hybrid becomes a concrete
biasing factor inside the geometric‑product Voronoi assignment.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data structures from Parent A
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """Immutable representation of a labeled text span."""
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """Single pheromone record with exponential decay."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(pathlib.Path(__file__).absolute())  # placeholder UUID
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay since the last update."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Very small in‑memory singleton‑like store."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PheromoneStore, cls).__new__(cls)
            cls._instance._entries: Dict[str, PheromoneEntry] = {}
        return cls._instance

    def get(self, surface_key: str) -> PheromoneEntry | None:
        return self._entries.get(surface_key)

    def add_or_update(self,
                      surface_key: str,
                      signal_kind: str,
                      delta: float,
                      half_life_seconds: int = 60) -> None:
        entry = self._entries.get(surface_key)
        if entry is None:
            entry = PheromoneEntry(surface_key, signal_kind, delta, half_life_seconds)
            self._entries[surface_key] = entry
        else:
            entry.apply_decay()
            entry.signal_value += delta
            entry.signal_kind = signal_kind
            entry.half_life_seconds = half_life_seconds

    def decay_all(self) -> None:
        for entry in self._entries.values():
            entry.apply_decay()


# ---------------------------------------------------------------------------
# Geometric algebra core from Parent B (2‑D Euclidean Cl(2,0))
# ---------------------------------------------------------------------------

def point_to_mv(point: Tuple[int, int]) -> Tuple[int, int]:
    """Convert a 2‑tuple (x, y) into a grade‑1 multivector (x·e₁ + y·e₂)."""
    return point  # stored as simple (x, y) pair


def mv_inner(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Scalar inner product ⟨a, b⟩ = a·e₁·b·e₁ + a·e₂·b·e₂."""
    return a[0] * b[0] + a[1] * b[1]


def mv_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance derived from the inner product."""
    diff = (a[0] - b[0], a[1] - b[1])
    return math.sqrt(mv_inner(diff, diff))


def rotate_toward(p: Tuple[int, int],
                  target: Tuple[int, int],
                  angle_rad: float) -> Tuple[float, float]:
    """
    Simple rotor that rotates point `p` toward `target` by `angle_rad`.
    Returns the new coordinates.
    """
    # vector from p to target
    dx, dy = target[0] - p[0], target[1] - p[1]
    r = math.hypot(dx, dy)
    if r == 0:
        return p
    # unit direction
    ux, uy = dx / r, dy / r
    # rotation matrix
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    # rotate the direction vector
    vx, vy = ux * cos_a - uy * sin_a, ux * sin_a + uy * cos_a
    # step a small fraction toward the rotated direction
    step = min(r, 1.0)  # limit step size
    return (p[0] + vx * step, p[1] + vy * step)


# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def span_to_mv(span: Span) -> Tuple[int, int]:
    """
    Map a Span to a 2‑D multivector using its `start` and `end` indices.
    The textual content does not enter the geometric space, keeping the bridge
    simple and deterministic.
    """
    return point_to_mv((span.start, span.end))


def weighted_mv_distance(span_mv: Tuple[int, int],
                         seed_mv: Tuple[int, int],
                         pher_store: PheromoneStore,
                         surface_key: str) -> float:
    """
    Euclidean distance between two multivectors, biased by the pheromone
    signal attached to `surface_key`.  The bias is a multiplicative factor:
        d_weighted = d / (1 + signal_value)
    where `signal_value` decays over time.
    """
    base_dist = mv_distance(span_mv, seed_mv)
    entry = pher_store.get(surface_key)
    if entry is None:
        return base_dist
    entry.apply_decay()
    bias = 1.0 + max(entry.signal_value, 0.0)
    return base_dist / bias


def assign_spans_to_seeds(spans: List[Span],
                          seed_points: List[Tuple[int, int]],
                          pher_store: PheromoneStore) -> Dict[int, int]:
    """
    Voronoi‑style assignment of each span to the nearest seed.
    Returns a dict mapping `span_index` → `seed_index`.
    The distance metric is the pheromone‑weighted multivector distance.
    """
    assignment: Dict[int, int] = {}
    for i, span in enumerate(spans):
        span_mv = span_to_mv(span)
        best_idx = -1
        best_dist = math.inf
        for j, seed_mv in enumerate(seed_points):
            d = weighted_mv_distance(span_mv, seed_mv, pher_store,
                                     surface_key=span.text)
            if d < best_dist:
                best_dist = d
                best_idx = j
        assignment[i] = best_idx
    return assignment


def deposit_pheromones_from_assignments(spans: List[Span],
                                        assignments: Dict[int, int],
                                        pher_store: PheromoneStore) -> None:
    """
    After a Voronoi assignment, reinforce the pheromone associated with the
    chosen seed using the span's `score`.  Higher scores increase the signal.
    """
    for span_idx, seed_idx in assignments.items():
        span = spans[span_idx]
        # Use the seed index as part of the surface key to keep signals separate
        key = f"seed_{seed_idx}_text_{span.text}"
        pher_store.add_or_update(surface_key=key,
                                 signal_kind="reinforcement",
                                 delta=span.score,
                                 half_life_seconds=120)


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create a few dummy spans
    demo_spans = [
        Span(start=0, end=5, text="hello", label="greeting", score=1.2),
        Span(start=10, end=15, text="world", label="object", score=0.8),
        Span(start=20, end=30, text="example", label="noun", score=1.5),
    ]

    # Random seed points in the same (start,end) space
    random.seed(42)
    seed_pts = [point_to_mv((random.randint(0, 30), random.randint(0, 30)))
                for _ in range(2)]

    store = PheromoneStore()
    # Initial assignment
    assign = assign_spans_to_seeds(demo_spans, seed_pts, store)
    print("Initial assignment:", assign)

    # Deposit pheromones based on assignment
    deposit_pheromones_from_assignments(demo_spans, assign, store)

    # Decay a bit and re‑assign to see the effect
    store.decay_all()
    assign2 = assign_spans_to_seeds(demo_spans, seed_pts, store)
    print("Re‑assignment after pheromone decay:", assign2)

    # Simple rotor demonstration
    p = (5, 10)
    t = (15, 20)
    rotated = rotate_toward(p, t, angle_rad=math.pi / 6)
    print(f"Rotated {p} toward {t} by 30° → {rotated}")