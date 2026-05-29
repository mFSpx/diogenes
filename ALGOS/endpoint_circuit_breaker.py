#!/usr/bin/env python3
"""Endpoint circuit-breaker and dual-engine pool selection primitives."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
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
        return not self.open

    def as_dict(self) -> dict[str, Any]:
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DualEngineEndpointPool:
    """Route work across resident CPU FairyFuse and GPU DeepSeek Q4 lanes."""

    def __init__(self, failure_threshold: int = 3):
        self.endpoints: dict[str, EngineEndpoint] = {
            "cpu_fairyfuse_ternary": EngineEndpoint(
                engine_id="cpu_fairyfuse_ternary",
                channel="cpu_fairyfuse_ternary",
                residency="always_on",
                runtime="python_ctypes_mmap",
                resource_class="cpu_ram_mmap",
                always_on=True,
                endpoint="ALGOS/ternary_router.py",
                capabilities=["semantic_stream", "fast_negative", "routing", "telemetry", "mtime_fragility"],
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
            ),
        }
        self.breakers = {k: EndpointCircuitBreaker(failure_threshold) for k in self.endpoints}

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

    def plan(self) -> dict[str, Any]:
        return {
            "schema": "lucidota.dual_engine_endpoint_pool.v1",
            "generated_at": now_z(),
            "endpoints": {k: e.as_dict() for k, e in self.endpoints.items()},
            "breakers": {k: b.as_dict() for k, b in self.breakers.items()},
            "available": [e.engine_id for e in self.available()],
            "outbound_state": "draft_only",
        }
