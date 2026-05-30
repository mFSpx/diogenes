# DARWIN HAMMER — match 18, survivor 1
# gen: 1
# parent_a: endpoint_circuit_breaker.py (gen0)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:21:12Z

"""This module fuses the endpoint_circuit_breaker.py and serpentina_self_righting.py algorithms.
It creates a novel hybrid system by integrating the circuit breaker concept with the serpentina self-righting morphology.
The mathematical bridge between the two structures is the concept of "recovery priority," which is used to determine the likelihood of an endpoint recovering from a failure.
The recovery priority is calculated based on the morphology of the endpoint, and this value is then used to adjust the circuit breaker's threshold for determining when to open or close the circuit."""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp
from random import random
from sys import exit
from pathlib import Path
from typing import Any, Dict


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, morphology: Morphology = None):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.morphology = morphology

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        if self.morphology is not None:
            recovery_p = recovery_priority(self.morphology)
            self.open = self.failures >= self.failure_threshold * (1 - recovery_p)
        else:
            self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> Dict[str, Any]:
        return {"failure_threshold": self.failure_threshold, "failures": self.failures, "open": self.open, "last_event_at": self.last_event_at}


class EndpointPool:
    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.breakers = {e: EndpointCircuitBreaker() for e in endpoints}

    def available(self) -> list[str]:
        return [e for e in self.endpoints if self.breakers[e].allow()]


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: list[str]
    outbound_state: str = "draft_only"
    morphology: Morphology = None

    def as_dict(self) -> Dict[str, Any]:
        return {"engine_id": self.engine_id, "channel": self.channel, "residency": self.residency, "runtime": self.runtime, "resource_class": self.resource_class, "always_on": self.always_on, "endpoint": self.endpoint, "capabilities": self.capabilities, "outbound_state": self.outbound_state, "morphology": self.morphology.__dict__ if self.morphology is not None else None}


class DualEngineEndpointPool:
    def __init__(self, failure_threshold: int = 3):
        self.endpoints: Dict[str, EngineEndpoint] = {
            "cpu_fairyfuse_ternary": EngineEndpoint(
                engine_id="cpu_fairyfuse_ternary",
                channel="cpu_fairyfuse_ternary",
                residency="always_on",
                runtime="python_ctypes_mmap",
                resource_class="cpu_ram_mmap",
                always_on=True,
                endpoint="ALGOS/ternary_router.py",
                capabilities=["semantic_stream", "fast_negative", "routing", "telemetry", "mtime_fragility"],
                morphology=Morphology(length=1.0, width=1.0, height=1.0, mass=1.0),
            ),
            "gpu_q4_deepseek": EngineEndpoint(
                engine_id="gpu_q4_deepseek",
                channel="gpu_q4_deepseek",
                residency="always_on",
                runtime="llama_cpp_q4_k_m",
                resource_class="gpu_vram_4gb",
                always_on=True,
                endpoint="http://127.0.0.1:8080",
                capabilities=["synthesis", "cross_exam", "lora_hot_swap", "abductive_validation", "context_reaper"],
                morphology=Morphology(length=2.0, width=2.0, height=2.0, mass=2.0),
            ),
        }
        self.breakers = {k: EndpointCircuitBreaker(failure_threshold, e.morphology) for k, e in self.endpoints.items()}

    def available(self) -> list[EngineEndpoint]:
        return [self.endpoints[k] for k, b in self.breakers.items() if b.allow()]

    def select(self, *, require_vram: bool = False, prefer_cpu: bool = False, task_class: str = "semantic_stream") -> EngineEndpoint:
        candidates = {e.engine_id: e for e in self.available()}
        if require_vram and "gpu_q4_deepseek" in candidates:
            return candidates["gpu_q4_deepseek"]
        if prefer_cpu and "cpu_fairyfuse_ternary" in candidates:
            return candidates["cpu_fairyfuse_ternary"]
        if task_class in {"synthesis", "cross_exam", "lora_hot_swap", "abductive_validation"} and "gpu_q4_deepseek" in candidates:
            return candidates["gpu_q4_deepseek"]
        if "cpu_fairyfuse_ternary" in candidates:
            return candidates["cpu_fairyfuse_ternary"]
        if "gpu_q4_deepseek" in candidates:
            return candidates["gpu_q4_deepseek"]
        raise RuntimeError("all dual-engine endpoints are circuit-open")

    def record(self, engine_id: str, success: bool) -> None:
        breaker = self.breakers[engine_id]
        breaker.record_success() if success else breaker.record_failure()

    def plan(self) -> Dict[str, Any]:
        return {
            "schema": "lucidota.dual_engine_endpoint_pool.v1",
            "generated_at": now_z(),
            "endpoints": {k: e.as_dict() for k, e in self.endpoints.items()},
            "breakers": {k: b.as_dict() for k, b in self.breakers.items()},
            "available": [e.engine_id for e in self.available()],
            "outbound_state": "draft_only",
        }


def test_recovery_priority():
    morphology = Morphology(length=1.0, width=1.0, height=1.0, mass=1.0)
    recovery_p = recovery_priority(morphology)
    assert 0.0 <= recovery_p <= 1.0


def test_circuit_breaker():
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert not breaker.allow()
    breaker.record_success()
    assert breaker.allow()


def test_dual_engine_endpoint_pool():
    pool = DualEngineEndpointPool(failure_threshold=3)
    available_endpoints = pool.available()
    assert len(available_endpoints) > 0


if __name__ == "__main__":
    test_recovery_priority()
    test_circuit_breaker()
    test_dual_engine_endpoint_pool()