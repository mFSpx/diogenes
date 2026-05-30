# DARWIN HAMMER — match 16, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (gen2)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s2.py (gen2)
# born: 2026-05-29T23:25:08Z

"""Hybrid Model‑VRAM Scheduler & Endpoint Workshare Allocator

Parents:
- hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (privacy risk + VRAM scheduling)
- hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s2.py (circuit‑breaker health + Doomsday workshare)

Mathematical bridge:
Each model tier is treated as an *endpoint*.  
Its **privacy reconstruction risk** `r` (0 ≤ r ≤ 1) reduces the *health* derived from the
circuit‑breaker:

    health = (1 - failure_rate) * (1 - recovery_priority)

The **combined score** used for scheduling and work‑share allocation is

    score = health * (1 - r)

Thus a model that is both reliable (low failure_rate, low recovery_priority) and
low‑risk (small r) receives a larger share of deterministic VRAM slots and
residual work units.  The deterministic portion follows the Doomsday factor
`d = (weekday+1) % 7` as in the original workshare allocator.

The module provides three core functions demonstrating this unified system:
`combined_model_score`, `allocate_workshare`, and `schedule_models`.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared primitives (from Parent A)
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


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Privacy risk: proportion of quasi‑identifiers to total records, clipped to [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Deterministic sum placeholder for a DP aggregate."""
    return sum(values)


class ModelPool:
    """Tracks RAM/VRAM usage and enforces ceiling constraints."""

    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.sensitive_records: List[dict] = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """Return True if loading `model` would stay within both ceilings."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            return False  # mutual exclusion rule from original code
        new_ram = self._used_ram() + model.ram_mb
        new_vram = self._used_vram() + model.vram_mb
        return new_ram <= self.ram_ceiling_mb and new_vram <= self.vram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load a model if constraints allow; raise otherwise."""
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(f"Cannot load {model.name}: resource ceiling exceeded or tier conflict.")
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)

    def snapshot(self) -> dict:
        """Return a simple dict describing current usage."""
        return {
            "used_ram_mb": self._used_ram(),
            "used_vram_mb": self._used_vram(),
            "loaded_models": list(self.loaded.keys()),
        }


# ----------------------------------------------------------------------
# Endpoint primitives (from Parent B)
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Failure counter that opens after a threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (endpoint usable)."""
        return not self.open

    def failure_rate(self) -> float:
        """Proportion of failures to threshold, clipped to [0,1]."""
        return min(1.0, self.failures / self.failure_threshold)


def health_score(cb: EndpointCircuitBreaker, recovery_priority: float) -> float:
    """Combined health from circuit‑breaker and morphology."""
    fr = cb.failure_rate()
    rp = max(0.0, min(1.0, recovery_priority))  # ensure within [0,1]
    return (1.0 - fr) * (1.0 - rp)


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------


def combined_model_score(
    model: ModelTier,
    cb: EndpointCircuitBreaker,
    recovery_priority: float,
    unique_quasi_identifiers: int,
    total_records: int,
) -> float:
    """
    Unified score = health * (1 - reconstruction_risk).

    Higher score → more likely to be kept in VRAM and receive residual work units.
    """
    health = health_score(cb, recovery_priority)
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    return health * (1.0 - risk)


def allocate_workshare(
    models: List[ModelTier],
    pool: ModelPool,
    total_units: int,
    deterministic_target_pct: float,
    unique_quasi_identifiers: int,
    total_records: int,
    cb_map: Dict[str, EndpointCircuitBreaker],
    recovery_priorities: Dict[str, float],
) -> Dict[str, int]:
    """
    Allocate `total_units` of work (e.g., VRAM slots) across `models`.

    - Deterministic portion follows the Doomsday factor.
    - Residual portion is distributed proportionally to `combined_model_score`.
    - Allocation respects the pool's RAM/VRAM ceilings; models that cannot be loaded
      receive zero units.
    Returns a mapping from model name to allocated unit count.
    """
    # ----- Doomsday deterministic component -----
    weekday = datetime.now(timezone.utc).weekday()  # Monday=0 … Sunday=6
    d = (weekday + 1) % 7
    deterministic_units = int(
        total_units * deterministic_target_pct / 100.0 * (1.0 + d / 7.0)
    )
    deterministic_units = min(deterministic_units, total_units)
    residual_units = total_units - deterministic_units

    # ----- Compute scores for models that can be loaded -----
    scores = {}
    for m in models:
        if not pool.can_load(m):
            scores[m.name] = 0.0
            continue
        cb = cb_map.get(m.name, EndpointCircuitBreaker())
        rp = recovery_priorities.get(m.name, 0.0)
        scores[m.name] = combined_model_score(
            m,
            cb,
            rp,
            unique_quasi_identifiers,
            total_records,
        )

    # Normalize residual scores
    total_score = sum(scores.values())
    allocation: Dict[str, int] = {m.name: 0 for m in models}
    # Give each model an equal share of deterministic units (rounded down)
    det_share = deterministic_units // len(models) if models else 0
    for m in models:
        allocation[m.name] += det_share

    leftover_det = deterministic_units - det_share * len(models)
    # Distribute any leftover deterministic units to highest‑scoring models
    if leftover_det > 0:
        top_models = sorted(models, key=lambda x: scores[x.name], reverse=True)[:leftover_det]
        for m in top_models:
            allocation[m.name] += 1

    # ----- Distribute residual units proportionally -----
    if total_score > 0 and residual_units > 0:
        for m in models:
            proportion = scores[m.name] / total_score
            allocation[m.name] += int(round(proportion * residual_units))

    # Ensure we do not exceed total_units due to rounding
    allocated_sum = sum(allocation.values())
    while allocated_sum > total_units:
        # Remove one unit from the model with the smallest score that still has >0 allocation
        low_model = min(models, key=lambda x: (scores[x.name], allocation[x.name]))
        if allocation[low_model.name] > 0:
            allocation[low_model.name] -= 1
            allocated_sum -= 1
        else:
            break

    return allocation


def schedule_models(
    models: List[ModelTier],
    pool: ModelPool,
    unique_quasi_identifiers: int,
    total_records: int,
    cb_map: Dict[str, EndpointCircuitBreaker],
    recovery_priorities: Dict[str, float],
) -> List[ModelTier]:
    """
    Load models into the `pool` in descending order of combined score until
    resource ceilings are hit. Returns the list of successfully loaded models.
    """
    scored_models = sorted(
        models,
        key=lambda m: combined_model_score(
            m,
            cb_map.get(m.name, EndpointCircuitBreaker()),
            recovery_priorities.get(m.name, 0.0),
            unique_quasi_identifiers,
            total_records,
        ),
        reverse=True,
    )
    loaded: List[ModelTier] = []
    for m in scored_models:
        try:
            pool.load(m)
            loaded.append(m)
        except RuntimeError:
            continue
    return loaded


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a pool with modest ceilings
    pool = ModelPool(ram_ceiling_mb=12000, vram_ceiling_mb=8192)

    # Define a set of models (including a T3 that conflicts with T2)
    all_models = [
        TIER_T1_QWEN_0_5B,
        TIER_T2_REASONING,
        TIER_T2_TOOL,
        TIER_T3_QWEN_7B,
    ]

    # Simulated privacy dataset stats
    uqis = 120  # unique quasi‑identifiers
    total = 1000  # total records

    # Create circuit breakers per model with random failure histories
    cb_map = {
        m.name: EndpointCircuitBreaker(failure_threshold=3) for m in all_models
    }
    for cb in cb_map.values():
        # Randomly inject failures
        for _ in range(random.randint(0, 4)):
            cb.record_failure()

    # Random recovery priorities (morphology‑driven)
    recovery_priorities = {m.name: random.random() for m in all_models}

    # Allocate workshare
    allocation = allocate_workshare(
        models=all_models,
        pool=pool,
        total_units=100,
        deterministic_target_pct=60.0,
        unique_quasi_identifiers=uqis,
        total_records=total,
        cb_map=cb_map,
        recovery_priorities=recovery_priorities,
    )
    print("Workshare allocation per model:")
    for name, units in allocation.items():
        print(f"  {name}: {units} units")

    # Schedule models based on combined scores
    loaded_models = schedule_models(
        models=all_models,
        pool=pool,
        unique_quasi_identifiers=uqis,
        total_records=total,
        cb_map=cb_map,
        recovery_priorities=recovery_priorities,
    )
    print("\nModels successfully loaded into the pool:")
    for m in loaded_models:
        print(f"  {m.name} (RAM {m.ram_mb} MB, VRAM {m.vram_mb} MB)")

    # Final pool snapshot
    print("\nPool snapshot:", pool.snapshot())