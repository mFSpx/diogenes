# DARWIN HAMMER — match 18, survivor 5
# gen: 1
# parent_a: endpoint_circuit_breaker.py (gen0)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:21:13Z

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives
# ----------------------------------------------------------------------


class EndpointCircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
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

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("width", self.width),
            ("height", self.height),
            ("mass", self.mass),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return self.length, self.width, self.height


def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    longest = max(length, width, height)
    return (length * width * height) ** (1.0 / 3.0) / longest


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
    # Guard against overflow for extreme flatness values
    exp_part = math.exp(min(k * fi, 50.0))
    return (m.mass ** b) * exp_part / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """
    Normalised priority (0 → 1) where a larger righting‑time yields a higher
    need for external assistance.
    """
    if max_index <= 0:
        raise ValueError("max_index must be positive")
    raw = righting_time_index(m) / max_index
    return max(0.0, min(1.0, raw))


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
# Hybrid pool – deeper integration of reliability & morphology
# ----------------------------------------------------------------------


class HybridEngineEndpointPool:
    """
    Manages a pool of EngineEndpoint objects, each equipped with a circuit breaker
    and a Morphology.  Selection uses a *health score* that blends:

        • Operational reliability (failure_rate)
        • Intrinsic right‑ability (recovery_priority)
        • Shape compactness (sphericity)
        • Flatness (inverse flatness)

    The score is a weighted geometric mean, guaranteeing a value in [0, 1].
    """

    # Weight configuration – can be tuned per deployment
    _W_RELIABILITY = 0.4
    _W_RECOVERY = 0.3
    _W_SPHERICITY = 0.2
    _W_FLATNESS = 0.1

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
            eid: EndpointCircuitBreaker(failure_threshold) for eid in self.endpoints
        }

    # ------------------------------------------------------------------
    # Core pool operations
    # ------------------------------------------------------------------

    def available(self) -> List[EngineEndpoint]:
        """Return endpoints whose circuit breakers are closed."""
        return [
            self.endpoints[eid]
            for eid, brk in self.breakers.items()
            if brk.allow()
        ]

    def _failure_rate(self, engine_id: str) -> float:
        """Normalized failure count in [0, 1]."""
        brk = self.breakers[engine_id]
        return min(1.0, brk.failures / brk.failure_threshold)

    def _morphology_factors(self, m: Morphology) -> Tuple[float, float, float]:
        """
        Compute three normalized morphology factors in [0, 1]:

            • sphericity_factor   – higher for more spherical objects.
            • flatness_factor     – higher for less flat objects.
            • recovery_factor     – higher for easier self‑righting (i.e. low priority).

        The factors are linearly scaled against empirically chosen caps.
        """
        # Sphericity: already in (0,1]; clamp for safety.
        sph = max(0.0, min(1.0, sphericity_index(m.length, m.width, m.height)))

        # Flatness: we invert and normalise.  Flatness can be >1 for very flat objects.
        raw_flat = flatness_index(m.length, m.width, m.height)
        # Choose a reasonable ceiling; values >5 are considered extremely flat.
        flat_norm = max(0.0, min(1.0, 1.0 - (raw_flat - 1.0) / 4.0))

        # Recovery priority: lower priority → easier to right → higher factor.
        rec_prio = recovery_priority(m)
        rec_factor = 1.0 - rec_prio

        return sph, flat_norm, rec_factor

    def health_score(self, endpoint: EngineEndpoint) -> float:
        """
        Compute a blended health score in [0, 1] using a weighted geometric mean.
        The geometric mean penalises any single poor component more heavily than
        an arithmetic mean, encouraging balanced performance.
        """
        fr = self._failure_rate(endpoint.engine_id)          # 0 → 1 (bad → good)
        reliability = 1.0 - fr

        sph, flat, rec = self._morphology_factors(endpoint.morphology)

        # Apply weights; add a tiny epsilon to avoid zero‑power issues.
        eps = 1e-12
        weighted_product = (
            (reliability + eps) ** self._W_RELIABILITY
            * (rec + eps) ** self._W_RECOVERY
            * (sph + eps) ** self._W_SPHERICITY
            * (flat + eps) ** self._W_FLATNESS
        )
        # Since weights sum to 1, the product already lies in [0,1].
        return float(weighted_product)

    def best_endpoint(self) -> EngineEndpoint | None:
        """Return the available endpoint with the highest health score."""
        candidates = self.available()
        if not candidates:
            return None
        scored = [(self.health_score(ep), ep) for ep in candidates]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Recording outcomes
    # ------------------------------------------------------------------

    def record_success(self, engine_id: str) -> None:
        """Reset failure counter for a successful interaction."""
        if engine_id not in self.breakers:
            raise KeyError(f"Unknown engine_id {engine_id}")
        self.breakers[engine_id].record_success()

    def record_failure(self, engine_id: str) -> None:
        """Increment failure counter for a failed interaction."""
        if engine_id not in self.breakers:
            raise KeyError(f"Unknown engine_id {engine_id}")
        self.breakers[engine_id].record_failure()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of the pool's current state."""
        return {
            "endpoints": {eid: ep.as_dict() for eid, ep in self.endpoints.items()},
            "breakers": {eid: brk.as_dict() for eid, brk in self.breakers.items()},
            "health_scores": {
                eid: self.health_score(ep) for eid, ep in self.endpoints.items()
            },
        }

    # ------------------------------------------------------------------
    # Convenience iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        """Iterate over EngineEndpoint objects (ordered by health descending)."""
        sorted_eps = sorted(
            self.endpoints.values(),
            key=self.health_score,
            reverse=True,
        )
        return iter(sorted_eps)


__all__ = [
    "now_z",
    "EndpointCircuitBreaker",
    "Morphology",
    "sphericity_index",
    "flatness_index",
    "righting_time_index",
    "recovery_priority",
    "EngineEndpoint",
    "HybridEngineEndpointPool",
]