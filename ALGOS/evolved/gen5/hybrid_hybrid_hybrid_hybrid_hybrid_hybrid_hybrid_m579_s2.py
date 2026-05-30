# DARWIN HAMMER — match 579, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_krampus_stick_m356_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_model__honeybee_store_m388_s1.py (gen3)
# born: 2026-05-29T23:29:46Z

"""Hybrid Algorithm integrating:
- Privacy risk and pheromone dynamics (Parent A)
- VRAM planning, Ollivier‑Ricci curvature, and Honeybee Store feedback (Parent B)

Mathematical Bridge:
The privacy risk score `r` modulates pheromone signal decay `δ = 2^{-Δt/τ}`.
The decayed pheromone value influences a *health* metric derived from the
remaining VRAM budget.  The health metric is then used as a node weight in a
graph whose edges are VRAM allocation plans.  Ollivier‑Ricci curvature `κ`
computed on this graph feeds back into the Honeybee Store, which adjusts the
allocation rate.  The final hybrid score combines all three influences:

    score = health * (1 - r) * δ * (1 + κ)

Thus risk, temporal pheromone decay, and curvature‑driven resource control are
fused into a single unified decision value.
"""

import sys
import math
import random
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Shared primitives from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def decay_factor(self, now: datetime | None = None) -> float:
        """Return the exponential decay factor based on elapsed time."""
        now = now or datetime.now(timezone.utc)
        elapsed = (now - self.last_decay).total_seconds()
        if self.half_life_seconds <= 0:
            return 1.0
        factor = 2 ** (-elapsed / self.half_life_seconds)
        self.last_decay = now
        self.signal_value *= factor
        return factor

# ----------------------------------------------------------------------
# Shared primitives from Parent B
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

class VramPlanner:
    """Very light‑weight VRAM planner."""
    def __init__(self, static_budget_mb: int = 4096, reserve_mb: int = 768):
        self.static_budget_mb = static_budget_mb
        self.reserve_mb = reserve_mb
        self._allocations: dict[str, VramSlotPlan] = {}

    @property
    def used_mb(self) -> int:
        return sum(p.estimated_mb for p in self._allocations.values())

    @property
    def free_mb(self) -> int:
        return max(0, self.static_budget_mb - self.used_mb - self.reserve_mb)

    def add_plan(self, plan: VramSlotPlan) -> None:
        if plan.estimated_mb > self.free_mb:
            raise RuntimeError("Insufficient VRAM for plan")
        self._allocations[plan.artifact_id] = plan

    def get_plans(self) -> list[VramSlotPlan]:
        return list(self._allocations.values())

# Simple graph representation for curvature computation
class SimpleGraph:
    """Undirected graph where nodes are artifact IDs and edge weights are
    absolute differences in estimated VRAM."""
    def __init__(self):
        self.nodes: set[str] = set()
        self.edges: dict[tuple[str, str], float] = {}

    def add_edge(self, u: str, v: str, weight: float) -> None:
        self.nodes.update([u, v])
        self.edges[(u, v)] = weight
        self.edges[(v, u)] = weight

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        return [(nbr, w) for (a, nbr), w in self.edges.items() if a == node]

# ----------------------------------------------------------------------
# Honeybee Store (feedback primitive)
# ----------------------------------------------------------------------
class HoneybeeStore:
    """Keeps a moving average of allocation rates per artifact."""
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.rates: dict[str, float] = {}

    def update(self, artifact_id: str, new_rate: float) -> None:
        old = self.rates.get(artifact_id, 0.0)
        self.rates[artifact_id] = self.decay * old + (1 - self.decay) * new_rate

    def get_rate(self, artifact_id: str) -> float:
        return self.rates.get(artifact_id, 0.0)

# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def privacy_risk_score(tier: ModelTier) -> float:
    """Compute a normalized risk score from model tier resources.
    Higher VRAM relative to RAM yields lower risk."""
    ratio = tier.vram_mb / max(1, tier.ram_mb)
    # Map ratio in [0, ∞) to risk in [0,1] via a logistic function
    risk = 1 / (1 + math.exp(5 * (ratio - 1.5)))  # centre around ratio≈1.5
    return risk

def compute_hybrid_score(tier: ModelTier,
                         pheromone: PheromoneEntry,
                         planner: VramPlanner,
                         curvature: float) -> float:
    """Combine health, risk, pheromone decay and curvature into one score."""
    # Health: proportion of free VRAM to static budget
    health = planner.free_mb / planner.static_budget_mb
    r = privacy_risk_score(tier)
    decay = pheromone.decay_factor()
    # curvature may be negative; shift to [0,2] by adding 1
    cur_factor = 1 + curvature
    score = health * (1 - r) * decay * cur_factor
    return score

def build_curvature_graph(planner: VramPlanner) -> tuple[SimpleGraph, dict]:
    """Create a graph from VRAM plans and compute Ollivier‑Ricci curvature
    approximations for each edge using a simple transport metric."""
    plans = planner.get_plans()
    g = SimpleGraph()
    # Connect every pair (complete graph) with weight = |ΔVRAM|
    for i, p_i in enumerate(plans):
        for p_j in plans[i + 1:]:
            w = abs(p_i.estimated_mb - p_j.estimated_mb) + 1e-6
            g.add_edge(p_i.artifact_id, p_j.artifact_id, w)

    # Simple curvature: κ = 1 - (d_ij' / d_ij)
    # where d_ij' = min(weight, avg_weight) as a proxy for transport cost
    avg_w = np.mean(list(g.edges.values())) if g.edges else 1.0
    curvature: dict[tuple[str, str], float] = {}
    for (u, v), w in g.edges.items():
        d_prime = min(w, avg_w)
        curvature[(u, v)] = 1 - (d_prime / w)
    return g, curvature

def adjust_vram_allocation(planner: VramPlanner,
                           store: HoneybeeStore,
                           curvature_map: dict) -> None:
    """Use curvature feedback to throttle or boost future allocations."""
    for (u, v), κ in curvature_map.items():
        # Positive curvature (κ>0) suggests redundancy – slow down
        # Negative curvature suggests scarcity – speed up
        factor = 1.0 - 0.2 * κ  # simple linear mapping
        for art_id in (u, v):
            current_rate = store.get_rate(art_id)
            new_rate = max(0.0, current_rate * factor)
            store.update(art_id, new_rate)

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    # Initialise components
    tier = TIER_T2_REASONING
    pher = PheromoneEntry(surface_key="model:privacy",
                          signal_kind="risk_mod",
                          signal_value=1.0,
                          half_life_seconds=300)

    planner = VramPlanner(static_budget_mb=8192, reserve_mb=1024)
    # Add a few synthetic allocation plans
    for i in range(3):
        plan = VramSlotPlan(
            artifact_id=f"art{i}",
            artifact_kind="model",
            action="load",
            estimated_mb=1024 + i * 256,
            reason="demo",
            detail={}
        )
        planner.add_plan(plan)

    # Build curvature graph
    graph, curv_map = build_curvature_graph(planner)

    # Initialise Honeybee store and feed curvature
    store = HoneybeeStore()
    # Seed store with dummy rates
    for plan in planner.get_plans():
        store.update(plan.artifact_id, random.uniform(0.5, 1.5))

    # Adjust allocations based on curvature
    adjust_vram_allocation(planner, store, curv_map)

    # Compute final hybrid score (use average curvature)
    avg_curv = np.mean(list(curv_map.values())) if curv_map else 0.0
    score = compute_hybrid_score(tier, pher, planner, avg_curv)

    print("Hybrid score:", score)
    print("Free VRAM MB:", planner.free_mb)
    print("Average curvature:", avg_curv)
    print("Store rates:", {k: round(v,3) for k,v in store.rates.items()})

if __name__ == "__main__":
    _demo()