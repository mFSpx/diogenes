# DARWIN HAMMER — match 2303, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s3.py (gen2)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s2.py (gen1)
# born: 2026-05-29T23:41:50Z

"""Hybrid module combining:
- hybrid_hybrid_doomsday_cale_hybrid_hdc_serpentin_m313_s3.py (Gini coefficient,
  Doomsday weekday → symbolic hypervector, hyperdimensional binding/bundling)
- hybrid_model_vram_scheduler_ttt_linear_m11_s2.py (VRAM budgeting, Test‑Time
  Training weight matrix updates)

Mathematical bridge:
1. The Gini coefficient of a set of scalar “artifact” sizes is used as a
   scaling factor for the bundling of hypervectors that represent those
   artifacts.
2. The Doomsday algorithm yields a weekday symbol for a given date; this
   symbol is turned into a hypervector and bound to the scaled bundle,
   producing a single “date‑aware” hypervector.
3. The VRAM planner treats every hypervector (including the date‑aware one)
   and the mutable Test‑Time Training (TTT) weight matrix W as memory
   artifacts.  Before each TTT update the planner checks whether the
   temporary memory required for the update fits inside the remaining budget.
   Thus the learning dynamics are coupled to a hardware‑aware budgeting policy.

The module provides three core hybrid functions:
- `gini_weighted_date_hypervector`
- `vram_aware_ttt_step`
- `plan_and_execute_hybrid_workflow`
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (Parent A)
# ----------------------------------------------------------------------
Vector = List[int]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: Iterable[Vector]) -> Vector:
    vecs = list(vectors)
    if not vecs:
        return []
    dim = len(vecs[0])
    return [sum(vecs[i][j] for i in range(len(vecs))) / len(vecs) for j in range(dim)]


# ----------------------------------------------------------------------
# Gini coefficient (Parent A)
# ----------------------------------------------------------------------
def gini_coefficient(values: Iterable[float]) -> float:
    """Return the Gini coefficient of a list of numbers."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    cumulative = 0.0
    for i, x in enumerate(xs, start=1):
        cumulative += i * x
    total = sum(xs)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


# ----------------------------------------------------------------------
# Doomsday weekday algorithm (Parent A)
# ----------------------------------------------------------------------
def doomsday_weekday(year: int, month: int, day: int) -> str:
    """
    Return the weekday name (e.g., 'Monday') for the given Gregorian date
    using the Doomsday algorithm.
    """
    # Anchor days for centuries
    century = year // 100
    anchor = (5 * (century % 4) + 2) % 7  # 0=Sunday, 1=Monday, ...

    # Doomsday for the year
    y = year % 100
    doomsday = (y // 12 + (y % 12) + (y % 12) // 4 + anchor) % 7

    # Month‑specific doomsday dates
    month_dooms = {
        1: 3 if not is_leap_year(year) else 4,
        2: 28 if not is_leap_year(year) else 29,
        3: 14,
        4: 4,
        5: 9,
        6: 6,
        7: 11,
        8: 8,
        9: 5,
        10: 10,
        11: 7,
        12: 12,
    }
    diff = day - month_dooms[month]
    weekday_index = (doomsday + diff) % 7
    names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return names[weekday_index]


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


# ----------------------------------------------------------------------
# VRAM planner (Parent B)
# ----------------------------------------------------------------------
BYTES_PER_FLOAT32 = 4  # numpy default for float32


@dataclass
class Artifact:
    name: str
    size_bytes: int
    timestamp: str = dt.datetime.now(dt.timezone.utc).isoformat()


class VRAMPlanner:
    """Tracks memory usage of artifacts and enforces a budget."""

    def __init__(self, budget_mb: int = 4096, reserve_mb: int = 768):
        self.budget_bytes = budget_mb * 1024 * 1024
        self.reserve_bytes = reserve_mb * 1024 * 1024
        self.artifacts: List[Artifact] = []

    def add_artifact(self, name: str, size_bytes: int) -> None:
        self.artifacts.append(Artifact(name=name, size_bytes=size_bytes))

    def current_usage(self) -> int:
        return sum(a.size_bytes for a in self.artifacts)

    def remaining(self) -> int:
        used = self.current_usage()
        return max(0, self.budget_bytes - self.reserve_bytes - used)

    def can_allocate(self, size_bytes: int) -> bool:
        return size_bytes <= self.remaining()

    def snapshot(self) -> dict[str, Any]:
        return {
            "budget_mb": self.budget_bytes // (1024 * 1024),
            "reserve_mb": self.reserve_bytes // (1024 * 1024),
            "usage_mb": self.current_usage() // (1024 * 1024),
            "remaining_mb": self.remaining() // (1024 * 1024),
            "artifacts": [asdict(a) for a in self.artifacts],
        }

    def log(self, path: Path | None = None) -> None:
        target = path or Path("vram_plan_snapshot.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.snapshot(), fh, indent=2)


def estimate_matrix_memory(matrix: np.ndarray) -> int:
    """Return memory consumption in bytes for a numpy matrix."""
    return matrix.nbytes


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------


def gini_weighted_date_hypervector(
    scalar_sizes: Iterable[float],
    date: dt.date,
    dim: int = 10000,
) -> Vector:
    """
    1. Compute Gini coefficient of `scalar_sizes`.
    2. Turn each size into a hypervector via `symbol_vector`.
    3. Bundle the hypervectors and scale the bundle by the Gini coefficient.
    4. Generate a weekday symbol from `date` using Doomsday, convert it to a
       hypervector, and bind it with the scaled bundle.
    Returns the final date‑aware hypervector.
    """
    # Step 1
    gini = gini_coefficient(scalar_sizes)

    # Step 2 – each size gets a symbolic hypervector
    size_vectors = [symbol_vector(str(s), dim) for s in scalar_sizes]

    # Step 3 – bundle and scale
    bundled = bundle(size_vectors)
    scaled_bundle = [gini * x for x in bundled]

    # Step 4 – weekday symbol vector and bind
    weekday = doomsday_weekday(date.year, date.month, date.day)
    weekday_vec = symbol_vector(weekday, dim)
    final_vec = bind(scaled_bundle, weekday_vec)
    return final_vec


def vram_aware_ttt_step(
    W: np.ndarray,
    token_vec: Vector,
    lr: float,
    planner: VRAMPlanner,
    temp_name: str = "ttt_temp_buffer",
) -> np.ndarray:
    """
    Perform a single Test‑Time Training (TTT) update on weight matrix `W`.

    The gradient is approximated as the outer product of the token hypervector
    with itself (a simple Hebbian rule).  Before applying the update we allocate
    a temporary buffer of the same size as `W`; the planner must approve this
    allocation.  If the allocation would exceed the budget, the update is
    skipped and the original `W` is returned unchanged.
    """
    # Estimate temporary memory needed for the gradient buffer
    temp_bytes = estimate_matrix_memory(W)

    if not planner.can_allocate(temp_bytes):
        # Not enough VRAM – abort the update
        return W

    # Register the temporary buffer (it will be released after the step)
    planner.add_artifact(temp_name, temp_bytes)

    # Convert token hypervector to a numpy float32 vector
    token_np = np.array(token_vec, dtype=np.float32)

    # Simple Hebbian gradient: outer product
    grad = np.outer(token_np, token_np)  # shape (dim, dim)

    # Gradient descent update
    W_new = W - lr * grad

    # The temporary buffer is conceptually freed after the step; we remove it
    # from the planner's list to keep the snapshot accurate.
    planner.artifacts = [a for a in planner.artifacts if a.name != temp_name]

    # Register the (unchanged) size of W as a persistent artifact if not already present
    if not any(a.name == "W_matrix" for a in planner.artifacts):
        planner.add_artifact("W_matrix", estimate_matrix_memory(W_new))

    return W_new


def plan_and_execute_hybrid_workflow(
    scalar_sizes: List[float],
    date: dt.date,
    dim: int = 10000,
    lr: float = 1e-7,
    steps: int = 3,
) -> Tuple[Vector, np.ndarray, dict]:
    """
    End‑to‑end demonstration:

    1. Build a Gini‑weighted date hypervector.
    2. Register it as a VRAM artifact.
    3. Initialise a TTT weight matrix `W`.
    4. Run `steps` TTT updates, each time checking the VRAM planner.
    5. Return the final hypervector, final weight matrix, and a planner snapshot.
    """
    planner = VRAMPlanner()

    # 1. Create the hybrid hypervector
    hybrid_vec = gini_weighted_date_hypervector(scalar_sizes, date, dim)

    # 2. Register its memory consumption (int size of int list)
    vec_memory = len(hybrid_vec) * sys.getsizeof(0)  # rough estimate per int
    planner.add_artifact("date_hypervector", vec_memory)

    # 3. Initialise weight matrix W (float32 for deterministic size)
    W = np.zeros((dim, dim), dtype=np.float32)

    # Register W
    planner.add_artifact("W_matrix", estimate_matrix_memory(W))

    # 4. Perform TTT steps
    token_vec = random_vector(dim, seed="initial_token")
    for step in range(steps):
        W = vram_aware_ttt_step(W, token_vec, lr, planner, temp_name=f"ttt_temp_step_{step}")

    # 5. Gather snapshot
    snapshot = planner.snapshot()
    return hybrid_vec, W, snapshot


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example scalar sizes (in MB) representing different model artifacts
    example_sizes = [120.5, 45.2, 300.0, 78.9]

    # Use today's date
    today = dt.date.today()

    hv, final_W, plan = plan_and_execute_hybrid_workflow(
        scalar_sizes=example_sizes,
        date=today,
        dim=2048,          # smaller dimension for quick test
        lr=1e-8,
        steps=5,
    )

    print("Hybrid date hypervector (first 10 components):", hv[:10])
    print("Final W shape:", final_W.shape)
    print("VRAM planner snapshot:")
    print(json.dumps(plan, indent=2))