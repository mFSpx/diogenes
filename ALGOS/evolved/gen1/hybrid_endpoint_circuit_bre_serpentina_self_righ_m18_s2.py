# DARWIN HAMMER — match 18, survivor 2
# gen: 1
# parent_a: endpoint_circuit_breaker.py (gen0)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:21:12Z

"""Hybrid Endpoint Morphology Pool

This module fuses two distinct parent algorithms:

* **Endpoint circuit‑breaker & dual‑engine pool** – manages failure counters,
  open/closed states and selects an engine based on capability flags.
* **Serpentina self‑righting morphology** – computes geometric indices
  (flatness, sphericity) and a recovery priority based on mass and shape.

The mathematical bridge is a *health score* that multiplies a normalized
circuit‑breaker reliability term with the complementary recovery priority
derived from the endpoint’s morphology.  The health score therefore encodes
both operational reliability (failures vs. threshold) and intrinsic “self‑righting”
ability (lower righting‑time ⇒ higher priority).  Selection, scoring and
record‑keeping are performed on this unified representation.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives (unchanged, except minor typing)
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Parent B – morphology and recovery priority
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


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


# ----------------------------------------------------------------------
# Hybrid pool – merges circuit‑breaker state with morphology‑driven priority
# ----------------------------------------------------------------------


class HybridEngineEndpointPool:
    """
    Manages a pool of EngineEndpoint objects, each equipped with a circuit breaker
    and a Morphology.  Selection uses a *health score*:

        health = (1 - failure_rate) * (1 - recovery_priority)

    where ``failure_rate = failures / failure_threshold``.  The score lives in
    [0, 1] and favours endpoints that are both operationally reliable and
    intrinsically easy to “right”.
    """

    def __init__(self, failure_threshold: int = 3):
        # Define two representative endpoints with plausible morphologies.
        self.endpoints: dict[str, EngineEndpoint] = {
            "cpu_fairyfuse_ternary": EngineEndpoint(
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
            "gpu_q4_deepseek": EngineEndpoint(
                engine_id="gpu_q4_deepseek",
                channel="gpu_q4_deepseek",
                residency="always_on",
                runtime="llama_cpp_q4_k_m",
                resource_class="gpu_vram_4gb",
                always_on=True,
                endpoint="http://127.0.0.1:8080",
                capabilities=[
                    "synthesis",
                    "cross_exam",
                    "lora_hot_swap",
                    "abductive_validation",
                    "context_reaper",
                ],
                morphology=Morphology(length=0.20, width=0.20, height=0.05, mass=1.2),
            ),
        }
        self.breakers: dict[str, EndpointCircuitBreaker] = {
            k: EndpointCircuitBreaker(failure_threshold) for k in self.endpoints
        }

    # ------------------------------------------------------------------
    # Core pool operations
    # ------------------------------------------------------------------

    def available(self) -> List[EngineEndpoint]:
        """Return endpoints whose circuit breakers are closed."""
        return [
            self.endpoints[k]
            for k, b in self.breakers.items()
            if b.allow()
        ]

    def _failure_rate(self, engine_id: str) -> float:
        """Normalized failure count in [0, 1]."""
        b = self.breakers[engine_id]
        return min(1.0, b.failures / b.failure_threshold)

    def health_score(self, endpoint: EngineEndpoint) -> float:
        """
        Compute hybrid health = (1 - failure_rate) * (1 - recovery_priority).
        Higher scores indicate a more desirable endpoint.
        """
        fr = self._failure_rate(endpoint.engine_id)
        rp = recovery_priority(endpoint.morphology)
        return (1.0 - fr) * (1.0 - rp)

    def select(self, *, require_vram: bool = False,
               prefer_cpu: bool = False,
               task_class: str = "semantic_stream") -> EngineEndpoint:
        """
        Hybrid selection:
        1. Filter by availability.
        2. Apply hard constraints (VRAM, CPU preference, task‑specific caps).
        3. Choose the endpoint with the maximal health score.
        """
        candidates = {e.engine_id: e for e in self.available()}

        # Apply hard constraints first.
        if require_vram and "gpu_q4_deepseek" in candidates:
            constrained = {"gpu_q4_deepseek"}
        elif prefer_cpu and "cpu_fairyfuse_ternary" in candidates:
            constrained = {"cpu_fairyfuse_ternary"}
        elif task_class in {
            "synthesis",
            "cross_exam",
            "lora_hot_swap",
            "abductive_validation",
        } and "gpu_q4_deepseek" in candidates:
            constrained = {"gpu_q4_deepseek"}
        else:
            constrained = set(candidates.keys())

        # Compute health scores for the constrained set.
        scores = {
            eid: self.health_score(candidates[eid]) for eid in constrained
        }

        if not scores:
            raise RuntimeError("No endpoints satisfy the selection constraints")

        # Pick the endpoint with the highest health score.
        best_id = max(scores, key=scores.get)
        return candidates[best_id]

    def record(self, engine_id: str, success: bool) -> None:
        """Update the circuit‑breaker based on the outcome of a request."""
        breaker = self.breakers[engine_id]
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()

    def plan(self) -> dict[str, Any]:
        """Diagnostic snapshot of the pool."""
        return {
            "schema": "lucidota.hybrid_engine_endpoint_pool.v1",
            "generated_at": now_z(),
            "endpoints": {
                k: e.as_dict() for k, e in self.endpoints.items()
            },
            "breakers": {
                k: b.as_dict() for k, b in self.breakers.items()
            },
            "available": [e.engine_id for e in self.available()],
            "health_scores": {
                e.engine_id: self.health_score(e) for e in self.available()
            },
        }


# ----------------------------------------------------------------------
# Demonstration functions (the required three)
# ----------------------------------------------------------------------


def compute_hybrid_score(pool: HybridEngineEndpointPool, engine_id: str) -> float:
    """Return the current health score for a given endpoint."""
    endpoint = pool.endpoints[engine_id]
    return pool.health_score(endpoint)


def hybrid_select(pool: HybridEngineEndpointPool, **kwargs) -> EngineEndpoint:
    """
    Thin wrapper around ``HybridEngineEndpointPool.select`` that prints the
    chosen engine and its health score.
    """
    chosen = pool.select(**kwargs)
    score = pool.health_score(chosen)
    print(
        f"Selected engine '{chosen.engine_id}' with health score {score:.3f}"
    )
    return chosen


def simulate_workflow(pool: HybridEngineEndpointPool, steps: int = 10) -> None:
    """
    Simulate a series of requests against the pool.
    Randomly decide success/failure and invoke ``record``.
    After each step print a short status report.
    """
    for step in range(1, steps + 1):
        # Randomly decide task characteristics.
        require_vram = random.random() < 0.3
        prefer_cpu = random.random() < 0.2
        task_class = random.choice(
            [
                "semantic_stream",
                "synthesis",
                "cross_exam",
                "lora_hot_swap",
                "abductive_validation",
            ]
        )
        try:
            engine = hybrid_select(
                pool,
                require_vram=require_vram,
                prefer_cpu=prefer_cpu,
                task_class=task_class,
            )
        except RuntimeError as exc:
            print(f"[step {step}] Selection failed: {exc}")
            break

        # Simulate outcome: higher mass & flatter shapes are more prone to failure.
        morphology = engine.morphology
        base_failure_prob = 0.1 + 0.2 * flatness_index(
            morphology.length, morphology.width, morphology.height
        )
        success = random.random() > base_failure_prob
        pool.record(engine.engine_id, success)

        status = "SUCCESS" if success else "FAILURE"
        print(
            f"[step {step}] Engine={engine.engine_id} "
            f"Task={task_class} "
            f"Outcome={status}"
        )

        # Optional: dump a brief plan every few steps.
        if step % 5 == 0:
            plan = pool.plan()
            print(f"--- Diagnostic snapshot at step {step} ---")
            print(f"Available: {plan['available']}")
            print(f"Health scores: {plan['health_scores']}")
            print("--- End snapshot ---\n")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Ensure deterministic behaviour for the smoke test.
    random.seed(42)
    np.random.seed(0)

    hybrid_pool = HybridEngineEndpointPool(failure_threshold=3)
    print("=== Hybrid Endpoint Pool Smoke Test ===")
    simulate_workflow(hybrid_pool, steps=12)
    print("=== Test completed without unhandled exceptions ===")