# DARWIN HAMMER — match 12, survivor 2
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3.py (gen1)
# parent_b: hybrid_workshare_allocator_doomsday_calendar_m14_s1.py (gen1)
# born: 2026-05-29T23:22:30Z

"""Hybrid Endpoint Workshare Allocator

Parents:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s3 (circuit‑breaker + morphology)
- hybrid_workshare_allocator_doomsday_calendar_m14_s1 (deterministic/LLM workshare with Doomsday factor)

Mathematical bridge:
Each endpoint receives a *health* score  

    health = (1 - failure_rate) * (1 - recovery_priority)

where `failure_rate = failures / failure_threshold` and
`recovery_priority` comes from the morphology‑driven righting‑time model.

The total workshare is split into a deterministic part and a residual (LLM) part
using the Doomsday factor `d = (weekday+1) % 7`.  

    deterministic_units = total_units * deterministic_target_pct/100 * (1 + d/7)

The residual units are distributed across endpoints proportionally to their
health scores, yielding a single unified allocation that respects both
operational reliability and geometric recovery difficulty.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives and morphology
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

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
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0,1]."""
        return min(1.0, self.failures / self.failure_threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width) / (2·height); larger ⇒ flatter."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    """
    Exponential model of self‑righting time.
    Larger mass and flatter shapes increase the index.
    """
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority (0 → 1) where a larger righting‑time yields a higher
    need for external assistance.
    """
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EngineEndpoint:
    """Endpoint enriched with a Morphology for hybrid scoring."""
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d


class HybridEngineEndpointPool:
    """
    Holds EngineEndpoint objects together with per‑endpoint circuit breakers.
    Provides health scoring and selection utilities.
    """

    def __init__(self, failure_threshold: int = 3):
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.endpoints: Dict[str, EngineEndpoint] = {}

        # Populate with two representative endpoints (same as parent A)
        self._add_endpoint(
            EngineEndpoint(
                engine_id="cpu_fairyfuse_ternary",
                channel="cpu_fairyfuse_ternary",
                residency="always_on",
                runtime="python_ctypes_mmap",
                resource_class="cpu_ram_mmap",
                always_on=True,
                endpoint="ALGOS/ternary_router.py",
                capabilities=[
                    "semantic_stream",
                    "fast_negative",
                    "routing",
                    "telemetry",
                    "mtime_fragility",
                ],
                morphology=Morphology(length=0.12, width=0.08, height=0.02, mass=0.5),
            ),
            failure_threshold,
        )
        self._add_endpoint(
            EngineEndpoint(
                engine_id="gpu_q4_deepseek",
                channel="gpu_q4_deepseek",
                residency="always_on",
                runtime="llama_cpp_q4_k_m",
                resource_class="gpu_vram_4gb",
                always_on=True,
                endpoint="http://127.0.0.1:8080",
                capabilities=[
                    "semantic_stream",
                    "fast_negative",
                    "routing",
                    "telemetry",
                    "mtime_fragility",
                ],
                morphology=Morphology(length=0.30, width=0.22, height=0.10, mass=2.0),
            ),
            failure_threshold,
        )

    def _add_endpoint(self, ep: EngineEndpoint, failure_threshold: int) -> None:
        self.endpoints[ep.engine_id] = ep
        self._breakers[ep.engine_id] = EndpointCircuitBreaker(failure_threshold)

    def breaker(self, engine_id: str) -> EndpointCircuitBreaker:
        return self._breakers[engine_id]

    def health_score(self, engine_id: str) -> float:
        """
        health = (1 - failure_rate) * (1 - recovery_priority)
        """
        br = self._breakers[engine_id]
        ep = self.endpoints[engine_id]
        failure_rate = br.failure_rate()
        rec_prio = recovery_priority(ep.morphology)
        return (1.0 - failure_rate) * (1.0 - rec_prio)

    def all_health_scores(self) -> Dict[str, float]:
        return {eid: self.health_score(eid) for eid in self.endpoints}


# ----------------------------------------------------------------------
# Parent B – Doomsday‑aware workshare allocation (adapted)
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round to six decimal places for deterministic output."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Returns a value in [0,6] where 0 == Sunday, 1 == Monday, …, 6 == Saturday.
    The algorithm used by the parent simply maps Python's weekday (Mon=0) to
    this range via (weekday+1) % 7.
    """
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Hybrid functions (the required three+)
# ----------------------------------------------------------------------


def compute_endpoint_health(pool: HybridEngineEndpointPool) -> Dict[str, float]:
    """
    Compute and return the health scores for every endpoint in the pool.
    """
    return pool.all_health_scores()


def allocate_workshare_hybrid(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    pool: HybridEngineEndpointPool,
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
) -> Dict[str, Any]:
    """
    Allocate ``total_units`` of work across endpoints.

    1. Deterministic portion is inflated by the Doomsday factor.
    2. Residual (LLM) portion is split proportionally to each endpoint's health.
    3. Returns a rich structure mirroring the parent B output plus per‑endpoint
       allocation details.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")

    # ---- Doomsday‑adjusted deterministic share ---------------------------------
    d = doomsday(year, month, day)  # 0‑6
    deterministic_units = total_units * deterministic_target_pct / 100.0 * (1 + d / 7.0)
    deterministic_units = min(deterministic_units, total_units)  # guard against overflow
    llm_units = total_units - deterministic_units

    # ---- Health‑weighted residual distribution ---------------------------------
    healths = compute_endpoint_health(pool)
    health_array = np.array(list(healths.values()), dtype=float)
    total_health = health_array.sum()
    if total_health == 0:
        # If every endpoint is unhealthy, fall back to equal split
        weights = np.full_like(health_array, 1.0 / len(health_array))
    else:
        weights = health_array / total_health

    endpoint_ids = list(healths.keys())
    residual_alloc = weights * llm_units

    lanes: List[Dict[str, Any]] = []
    for eid, units in zip(endpoint_ids, residual_alloc):
        lanes.append(
            {
                "engine_id": eid,
                "llm_units": _pct(units),
                "llm_share_pct": _pct(100.0 * units / llm_units) if llm_units else 0.0,
                "health_score": _pct(healths[eid]),
                "allow": pool.breaker(eid).allow(),
            }
        )

    # ---- Assemble final payload -------------------------------------------------
    jzloads = [
        {
            "kind": "OBJECT",
            "id": "project2501_workshare_policy",
            "type": "workshare_policy",
            "deterministic_target_pct": _pct(deterministic_target_pct),
            "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        },
        {
            "kind": "EVENT",
            "id": "project2501_workshare_allocation",
            "type": "allocation_computed",
            "total_units": _pct(total_units),
            "deterministic_units": _pct(deterministic_units),
            "llm_units": _pct(llm_units),
        },
    ]

    for lane in lanes:
        jzloads.append(
            {
                "kind": "EDGE",
                "from": "project2501_workshare_policy",
                "to": f"engine:{lane['engine_id']}",
                "type": "ASSIGNS_LLM_RESIDUAL",
                "llm_units": lane["llm_units"],
                "llm_share_pct": lane["llm_share_pct"],
                "health_score": lane["health_score"],
                "endpoint_allowed": lane["allow"],
            }
        )

    return {
        "schema": "lucidota.project2501.hybrid_workshare_allocation.v1",
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "doomsday_factor": d,
        "lanes": lanes,
        "jzloads": jzloads,
    }


def summarize_hybrid_allocation(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    pool: HybridEngineEndpointPool,
    year: int = date.today().year,
    month: int = date.today().month,
    day: int = date.today().day,
) -> Dict[str, Any]:
    """
    Produce a high‑level summary akin to the parent B ``summarize_savings_*``
    but enriched with health‑aware metrics.
    """
    allocation = allocate_workshare_hybrid(
        total_units=total_units,
        deterministic_target_pct=deterministic_target_pct,
        pool=pool,
        year=year,
        month=month,
        day=day,
    )
    llm_units = allocation["llm_units"]
    deterministic_units = allocation["deterministic_units"]
    token_savings_pct = _pct((total_units - llm_units) / total_units * 100.0)

    per_engine = {
        lane["engine_id"]: {"llm_units": lane["llm_units"], "health": lane["health_score"]}
        for lane in allocation["lanes"]
    }

    return {
        "schema": "lucidota.project2501.hybrid_token_savings.v1",
        "baseline_llm_units": _pct(total_units),
        "planned_llm_units": llm_units,
        "deterministic_units": deterministic_units,
        "token_savings_pct": token_savings_pct,
        "per_engine_allocation": per_engine,
        "doomsday_factor": allocation["doomsday_factor"],
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a pool and deliberately inject failures to see health impact
    pool = HybridEngineEndpointPool(failure_threshold=3)

    # Simulate some activity
    cpu_br = pool.breaker("cpu_fairyfuse_ternary")
    gpu_br = pool.breaker("gpu_q4_deepseek")

    # Record two failures on GPU, one success on CPU
    gpu_br.record_failure()
    gpu_br.record_failure()
    cpu_br.record_success()

    # Show health scores
    print("Health scores:", compute_endpoint_health(pool))

    # Run hybrid allocation
    alloc = allocate_workshare_hybrid(
        total_units=120.0,
        deterministic_target_pct=85.0,
        pool=pool,
    )
    print("\nHybrid allocation payload:")
    for k, v in alloc.items():
        if k != "jzloads":
            print(f"{k}: {v}")

    # Summarize
    summary = summarize_hybrid_allocation(
        total_units=120.0,
        deterministic_target_pct=85.0,
        pool=pool,
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")