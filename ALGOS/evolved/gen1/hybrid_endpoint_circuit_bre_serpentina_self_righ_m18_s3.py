# DARWIN HAMMER — match 18, survivor 3
# gen: 1
# parent_a: endpoint_circuit_breaker.py (gen0)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:21:12Z

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
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


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
        return (1 - fr) * (1 - rp)

    def select_best(self) -> EngineEndpoint | None:
        """Return the endpoint with the highest health score."""
        available_endpoints = self.available()
        if not available_endpoints:
            return None
        return max(available_endpoints, key=self.health_score)

    def record_failure(self, engine_id: str) -> None:
        """Record a failure for the given endpoint."""
        self.breakers[engine_id].record_failure()

    def record_success(self, engine_id: str) -> None:
        """Record a success for the given endpoint."""
        self.breakers[engine_id].record_success()