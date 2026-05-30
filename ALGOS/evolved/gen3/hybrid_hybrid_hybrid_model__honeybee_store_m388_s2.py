# DARWIN HAMMER — match 388, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s0.py (gen2)
# parent_b: honeybee_store.py (gen0)
# born: 2026-05-29T23:28:30Z

"""Hybrid Vram Planner + Honeybee Store

This module fuses:
- **Parent A**: `hybrid_hybrid_model_vram_sc_hybrid_krampus_brain_m3_s0.py` – a VRAM
  planner that treats each artifact allocation as a node attribute in a graph.
- **Parent B**: `honeybee_store.py` – a simple store dynamics primitive that updates
  a scalar store based on inflow/outflow rates.

**Mathematical bridge**

Each artifact’s estimated VRAM (`estimated_mb`) forms a vector **x** ∈ ℝⁿ.
We treat **x** as a node‑weight vector on a fully‑connected graph with adjacency
matrix **A** (Aᵢⱼ = 1 for i ≠ j).  The Ollivier‑Ricci curvature on edge (i,j) can be
approximated by  

    κᵢⱼ = 1 - (|xᵢ - xⱼ| / dᵢⱼ) ,

where dᵢⱼ is the graph distance (here simply 1).  This curvature matrix **K**
captures how evenly VRAM is distributed.

The Honeybee store dynamics provide a scalar “budget store” **S** that evolves
according to  

    Sₜ₊₁ = max(0, Sₜ + Δt·(α·∑inflow - β·∑outflow)) .

We close the loop by feeding the curvature‑averaged feedback **c = mean(K)**
into the store update as an additional inflow term, thereby letting the graph‑
theoretic balance influence the available VRAM budget, while the store value
constrains the planner’s admissible allocations.

The three core functions below demonstrate this hybrid operation:
`allocation_vector`, `curvature_matrix`, and `hybrid_update`.
"""

import json
import os
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Constants & helpers (mirroring Parent A)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)

# ----------------------------------------------------------------------
# Minimal VRAM planner (simplified version of Parent A)
# ----------------------------------------------------------------------
class VramPlanner:
    """Collect artifact plans and expose them as a weighted vector."""
    def __init__(self, static_budget_mb: int = DEFAULT_BUDGET_MB,
                 reserve_mb: int = DEFAULT_RESERVE_MB):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._plans: dict[str, VramSlotPlan] = {}

    def add_plan(self, plan: VramSlotPlan) -> None:
        """Insert or replace a plan for a given artifact."""
        self._plans[plan.artifact_id] = plan

    def remove_plan(self, artifact_id: str) -> None:
        self._plans.pop(artifact_id, None)

    def allocation_vector(self) -> np.ndarray:
        """Return the current VRAM allocation vector **x** (ordered by artifact_id)."""
        if not self._plans:
            return np.array([], dtype=float)
        # Sort keys to obtain deterministic ordering
        sorted_keys = sorted(self._plans.keys())
        values = [self._plans[k].estimated_mb for k in sorted_keys]
        return np.array(values, dtype=float)

    def artifact_ids(self) -> list[str]:
        return sorted(self._plans.keys())

# ----------------------------------------------------------------------
# Curvature computation (core of Parent A)
# ----------------------------------------------------------------------
def curvature_matrix(allocation_vec: np.ndarray) -> np.ndarray:
    """
    Compute an approximate Ollivier‑Ricci curvature matrix **K** for a fully
    connected graph where edge distance dᵢⱼ = 1.

    κᵢⱼ = 1 - |xᵢ - xⱼ|   (clipped to [-1, 1]).

    Returns a symmetric matrix with zeros on the diagonal.
    """
    n = allocation_vec.size
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    # Broadcast differences |x_i - x_j|
    diff = np.abs(allocation_vec[:, None] - allocation_vec[None, :])
    K = 1.0 - diff  # since d_ij = 1
    np.fill_diagonal(K, 0.0)  # no self‑curvature
    # Clip to reasonable range
    np.clip(K, -1.0, 1.0, out=K)
    return K

# ----------------------------------------------------------------------
# Store dynamics (Parent B)
# ----------------------------------------------------------------------
def update_store(store: float,
                 inflow: list[float],
                 outflow: list[float],
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 dt: float = 1.0) -> tuple[float, float]:
    """
    Update a scalar store S using inflow/outflow rates.

    Returns (new_store, delta) where delta = α·∑inflow - β·∑outflow.
    """
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

def dance_duration(delta_store: float,
                   base: float = 1.0,
                   gain: float = 1.0,
                   limit: float = 10.0) -> float:
    """Translate a store delta into a bounded “dance” duration."""
    return max(0.0, min(limit, base + gain * delta_store))

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_update(planner: VramPlanner,
                  store: float,
                  inflow: list[float],
                  outflow: list[float],
                  alpha: float = 1.0,
                  beta: float = 1.0,
                  dt: float = 1.0) -> tuple[float, np.ndarray, float]:
    """
    Perform a single hybrid step:

    1. Compute curvature matrix K from current allocations.
    2. Reduce K to a scalar feedback `c = mean(K)`.
    3. Feed `c` as an extra inflow term to the store dynamics.
    4. Return updated store, curvature matrix, and the dance duration derived
       from the store delta.

    The returned curvature matrix can be inspected or used to re‑weight future
    allocations.
    """
    # 1. Current allocation vector
    x = planner.allocation_vector()
    K = curvature_matrix(x)

    # 2. Scalar curvature feedback (mean of off‑diagonal entries)
    if K.size == 0:
        c = 0.0
    else:
        off_diag = K[~np.eye(K.shape[0], dtype=bool)]
        c = float(np.mean(off_diag))

    # 3. Augment inflow with curvature feedback
    augmented_inflow = inflow + [c]

    # 4. Store update
    new_store, delta = update_store(store, augmented_inflow, outflow,
                                    alpha=alpha, beta=beta, dt=dt)

    # 5. Map delta to a “dance” duration (purely illustrative)
    duration = dance_duration(delta, base=1.0, gain=0.5, limit=5.0)

    return new_store, K, duration

# ----------------------------------------------------------------------
# Utility: generate dummy plans for testing
# ----------------------------------------------------------------------
def generate_dummy_plans(num: int) -> list[VramSlotPlan]:
    """Create `num` synthetic VramSlotPlan objects with random sizes."""
    plans = []
    for i in range(num):
        plan = VramSlotPlan(
            artifact_id=f"artifact_{i}",
            artifact_kind="model",
            action="load",
            estimated_mb=random.randint(100, 1500),
            reason="test",
            detail={}
        )
        plans.append(plan)
    return plans

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise planner and populate with dummy artifacts
    planner = VramPlanner()
    for p in generate_dummy_plans(5):
        planner.add_plan(p)

    # Initial store is the static budget minus reserve
    initial_store = planner.static_budget_mb - planner.reserve_mb

    # Define arbitrary inflow/outflow streams
    inflow = [50.0, 30.0]          # e.g., newly freed VRAM, predicted releases
    outflow = [120.0]             # e.g., upcoming loads

    # Run a few hybrid steps
    store = initial_store
    for step in range(3):
        store, K, dur = hybrid_update(planner, store, inflow, outflow,
                                      alpha=0.9, beta=1.1, dt=0.5)
        print(f"Step {step+1}:")
        print(f"  Store = {store:.2f} MB")
        print(f"  Curvature matrix (mean off‑diag) = {K[~np.eye(K.shape[0], dtype=bool)].mean():.4f}")
        print(f"  Dance duration = {dur:.2f} s")
        # Simulate a change in allocations based on curvature feedback:
        # if curvature is low (negative), we spread load more evenly.
        if K.size > 0:
            mean_alloc = planner.allocation_vector().mean()
            for aid in planner.artifact_ids():
                plan = planner._plans[aid]
                # Pull allocation towards the mean by up to 10%
                delta = int(0.1 * (mean_alloc - plan.estimated_mb))
                new_mb = max(50, plan.estimated_mb + delta)
                planner._plans[aid] = VramSlotPlan(
                    artifact_id=plan.artifact_id,
                    artifact_kind=plan.artifact_kind,
                    action=plan.action,
                    estimated_mb=new_mb,
                    reason=plan.reason,
                    detail=plan.detail,
                )
    print("Hybrid test completed without errors.")