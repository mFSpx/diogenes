# DARWIN HAMMER — match 24, survivor 2
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s2.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py (gen1)
# born: 2026-05-29T23:22:54Z

"""Hybrid Endpoint‑Morphology‑BrainMap Fusion

Parents:
- **Parent A**: `hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s2.py`
  Provides a circuit‑breaker reliability model and morphology‑based
  recovery priority. The core equation is the *health score*  

      health = (1 - failures / threshold) * (1 - recovery_priority)

- **Parent B**: `hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s5.py`
  Generates a deterministic 24‑dimensional feature vector from text and
  maps it to a 3‑D brain coordinate using a curvature factor.

**Mathematical Bridge**

The bridge is a *curvature‑modulated health factor* that injects the
morphology‑derived reliability (health) into the curvature term used by
the brain‑map. Concretely:


morph_curvature = sphericity_index * flatness_index          # ∈ (0, ∞)
curvature_score = health * (0.5 + 0.5 * tanh(morph_curvature))


`curvature_score` replaces the pure curvature input of `brain_xyz`,
thereby fusing operational reliability, physical self‑righting ability,
and semantic feature extraction into a single unified representation.

The module implements three public hybrid functions:
1. `compute_health(endpoint, breaker)` – health score.
2. `compute_curvature_score(morph, health)` – curvature from health & morphology.
3. `hybrid_brain_map(text, endpoint, breaker)` – full pipeline returning
   a 3‑D coordinate and auxiliary diagnostics.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – circuit‑breaker primitives
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
# Parent A – morphology utilities
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
# Parent B – deterministic feature extraction
# ----------------------------------------------------------------------


def _rng_from_text(text: str) -> random.Random:
    """Create a deterministic RNG seeded from a stable hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑features for demonstration."""
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Human‑readable 24‑dimensional vector."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get(
            "rainmaker_asset_structuring_weight", 0.0
        ),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }


# ----------------------------------------------------------------------
# Hybrid core: health → curvature → brain map
# ----------------------------------------------------------------------


def compute_health(endpoint: EngineEndpoint, breaker: EndpointCircuitBreaker) -> float:
    """
    Compute the health score for a given endpoint.

    health = (1 - failures / threshold) * (1 - recovery_priority)

    Returns a float in [0, 1].
    """
    if breaker.failure_threshold == 0:
        raise ValueError("failure_threshold must be > 0")
    failure_rate = breaker.failures / breaker.failure_threshold
    failure_rate = max(0.0, min(1.0, failure_rate))

    rp = recovery_priority(endpoint.morphology)
    health = (1.0 - failure_rate) * (1.0 - rp)
    return max(0.0, min(1.0, health))


def compute_curvature_score(morph: Morphology, health: float) -> float:
    """
    Fuse morphology with health into a curvature factor.

    morph_curvature = sphericity * flatness
    curvature_score = health * (0.5 + 0.5 * tanh(morph_curvature))

    The tanh maps the potentially unbounded product into (‑1, 1),
    yielding a factor in [0, 1] that is finally weighted by health.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    morph_curvature = sph * flat
    factor = 0.5 + 0.5 * math.tanh(morph_curvature)
    curvature_score = health * factor
    return curvature_score


def brain_xyz(master: Dict[str, float], curvature_score: float) -> Dict[str, float]:
    """
    3‑D coordinate generation modulated by curvature_score.

    The base axes follow the original Krampus formulation; each axis is
    multiplied by ``(1 + curvature_score)`` and receives a small additive
    offset proportional to ``curvature_score`` to guarantee influence.
    """
    # Base axes (identical to original)
    x_base = (
        master.get("visceral_ratio", 0.0) * 8
        + master.get("ledger_density", 0.0) * 6
        + min(master.get("directive_ratio", 0.0), 8.0) / 8
        + master.get("recursion_score", 0.0) * 4
    )
    y_base = (
        master.get("forensic_shield_ratio", 0.0) * 6
        + master.get("poetic_entropy", 0.0) * 4
        + min(master.get("dissociative_index", 0.0), 8.0) / 8
        + master.get("resource_exhaustion_metric", 0.0) * 6
        + master.get("bureaucratic_weaponization_index", 0.0) * 4
    )
    z_base = (
        master.get("corporate_grit_tension", 0.0) * 6
        + master.get("countdown_density", 0.0) * 6
        + master.get("asset_structuring_weight", 0.0) * 4
        + master.get("swarm_orchestration_density", 0.0) * 4
        + master.get("chaotic_good_tax", 0.0) * 4
        + master.get("agent_symmetry_ratio", 0.0) * 0.5
        + master.get("protocol_discipline", 0.0) * 0.2
        + master.get("manic_velocity", 0.0) * 0.4
    )

    # Apply curvature modulation
    factor = 1.0 + curvature_score          # multiplicative boost
    offset = curvature_score * 0.1          # small additive term

    x = x_base * factor + offset
    y = y_base * factor + offset
    z = z_base * factor + offset

    return {"x": x, "y": y, "z": z}


def hybrid_brain_map(
    text: str, endpoint: EngineEndpoint, breaker: EndpointCircuitBreaker
) -> Dict[str, Any]:
    """
    Full hybrid pipeline:

    1. Compute health from circuit‑breaker + morphology.
    2. Derive curvature_score using health and morphology.
    3. Extract deterministic feature vector from *text*.
    4. Produce 3‑D coordinates with ``brain_xyz``.
    5. Return a diagnostics dictionary.
    """
    health = compute_health(endpoint, breaker)
    curvature = compute_curvature_score(endpoint.morphology, health)
    master_vec = extract_master_vector(text)
    coords = brain_xyz(master_vec, curvature)

    return {
        "endpoint_id": endpoint.engine_id,
        "health": health,
        "curvature_score": curvature,
        "coordinates": coords,
        "features": master_vec,
    }


# ----------------------------------------------------------------------
# Helper pool for managing multiple endpoints
# ----------------------------------------------------------------------


class HybridEngineEndpointPool:
    """
    Manages a collection of EngineEndpoint objects, each paired with its own
    EndpointCircuitBreaker. Provides selection based on the hybrid health
    metric.
    """

    def __init__(self, failure_threshold: int = 3):
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.endpoints: Dict[str, EngineEndpoint] = {}

    def add_endpoint(self, ep: EngineEndpoint) -> None:
        self.endpoints[ep.engine_id] = ep
        self._breakers[ep.engine_id] = EndpointCircuitBreaker(
            failure_threshold=3
        )

    def record_success(self, engine_id: str) -> None:
        self._breakers[engine_id].record_success()

    def record_failure(self, engine_id: str) -> None:
        self._breakers[engine_id].record_failure()

    def best_endpoint(self, text: str) -> Dict[str, Any]:
        """
        Evaluate all endpoints on the provided *text* and return the result
        dict of the endpoint with the highest health‑adjusted brain coordinate.
        """
        best: Dict[str, Any] | None = None
        best_score = -1.0
        for ep in self.endpoints.values():
            breaker = self._breakers[ep.engine_id]
            result = hybrid_brain_map(text, ep, breaker)
            # Use health as primary selector; tie‑break with sum of coordinates
            primary