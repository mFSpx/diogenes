# DARWIN HAMMER — match 199, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s1.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py (gen1)
# born: 2026-05-29T23:27:41Z

"""HybridModelEndpointPool
Combines:
- hybrid_hard_truth_math_model_pool (vector‑based model loading with a linear “rectified flow” schedule)
- serpentina_self_righting (morphology‑driven priority metrics and circuit‑breaker endpoint management)

Mathematical bridge
------------------
Let  **r** ∈ ℝ⁺ be the RAM footprint of a model and let  **p(m)** ∈ [0,1] be the recovery priority of an endpoint,
computed from its morphology (righting‑time index → normalized priority).
We define a scalar field  

    f(endpoint, model) = p(m) · (1 – r / R_max)

where **R_max** is the RAM ceiling of the ModelPool.  
The rectified‑flow algorithm replaces a curved schedule with a straight‑line interpolant
between a current allocation **a₀** and a target allocation **a₁**:

    a(α) = (1 – α)·a₀ + α·a₁ , α∈[0,1]

In the hybrid system the target allocation **a₁** for a given endpoint is chosen
proportionally to the field **f(endpoint, model)**, thus coupling the morphology‑derived
priority (parent B) with the linear load‑unload schedule (parent A).  The resulting
system simultaneously respects RAM limits, endpoint health (circuit breaker), and
morphology‑aware loading decisions.

The module implements three core hybrid operations:
1. `linear_interpolant` – generic rectified‑flow interpolation.
2. `hybrid_priority` – computes f(endpoint, model) using morphology indices.
3. `load_model_hybrid` – attempts to load a model for an endpoint, obeying the circuit
   breaker, RAM ceiling, and linear schedule.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List, Any

# ----------------------------------------------------------------------
# Parent A – Model pool with RAM ceiling and linear schedule utilities
# ----------------------------------------------------------------------
class ModelTier:
    """Lightweight descriptor of a model."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    """Manages loaded models under a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """Return True if the model fits within the remaining RAM."""
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        """Load a model if RAM permits; raise otherwise."""
        if self.is_loaded(model.name):
            return  # already loaded
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)

# ----------------------------------------------------------------------
# Parent B – Morphology metrics and endpoint health management
# ----------------------------------------------------------------------
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

def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority ∈[0,1] based on righting‑time index."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

@dataclass(frozen=True)
class EngineEndpoint:
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

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["morphology"] = asdict(self.morphology)
        return d

class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open

class HybridEngineEndpointPool:
    """Container for EngineEndpoint objects with per‑endpoint circuit breakers."""
    def __init__(self, failure_threshold: int = 3):
        self.endpoints: Dict[str, EngineEndpoint] = {}
        self.circuit_breakers: Dict[str, EndpointCircuitBreaker] = {}
        self.failure_threshold = failure_threshold

    def register(self, endpoint: EngineEndpoint) -> None:
        self.endpoints[endpoint.engine_id] = endpoint
        self.circuit_breakers[endpoint.engine_id] = EndpointCircuitBreaker(self.failure_threshold)

    def get(self, engine_id: str) -> EngineEndpoint:
        return self.endpoints[engine_id]

    def breaker(self, engine_id: str) -> EndpointCircuitBreaker:
        return self.circuit_breakers[engine_id]

# ----------------------------------------------------------------------
# Hybrid utilities – mathematical fusion of the two parent topologies
# ----------------------------------------------------------------------
def linear_interpolant(start_vec: np.ndarray, end_vec: np.ndarray, alpha: float) -> np.ndarray:
    """
    Rectified‑flow straight‑line interpolation.
    Returns (1‑α)·start + α·end, with α∈[0,1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    return (1.0 - alpha) * start_vec + alpha * end_vec

def hybrid_priority(endpoint: EngineEndpoint, model: ModelTier, model_pool: ModelPool) -> float:
    """
    Compute the fused priority f(endpoint, model) = p(m)·(1 – r / R_max).

    - p(m) is the recovery priority from morphology.
    - r is the RAM footprint of the model.
    - R_max is the RAM ceiling of the ModelPool.
    The result lies in [0,1]; higher values favour loading.
    """
    p = recovery_priority(endpoint.morphology)
    ram_factor = 1.0 - (model.ram_mb / model_pool.ram_ceiling_mb)
    return max(0.0, min(1.0, p * ram_factor))

def load_model_hybrid(
    endpoint_pool: HybridEngineEndpointPool,
    model_pool: ModelPool,
    engine_id: str,
    model: ModelTier,
    schedule_alpha: float = 0.5,
) -> bool:
    """
    Attempt to load *model* for the endpoint identified by *engine_id*.

    The procedure:
    1. Verify the circuit breaker allows the operation.
    2. Compute the hybrid priority.
    3. Construct a linear schedule between the current RAM usage vector
       and the target vector that would include the new model.
    4. If the priority exceeds a dynamic threshold (here 0.3) and the
       interpolated RAM usage stays within the ceiling, perform the load.
    5. Record success/failure on the breaker.

    Returns True on successful load, False otherwise.
    """
    # 1. Circuit‑breaker guard
    breaker = endpoint_pool.breaker(engine_id)
    if not breaker.allow():
        return False

    endpoint = endpoint_pool.get(engine_id)

    # 2. Hybrid priority
    priority = hybrid_priority(endpoint, model, model_pool)

    # 3. Linear schedule: current usage vs prospective usage
    current_usage = np.array([model_pool._used()], dtype=float)
    prospective_usage = np.array([model_pool._used() + model.ram_mb], dtype=float)

    scheduled_usage = linear_interpolant(current_usage, prospective_usage, schedule_alpha)[0]

    # 4. Decision rule
    if priority >= 0.3 and scheduled_usage <= model_pool.ram_ceiling_mb:
        try:
            model_pool.load(model)
            breaker.record_success()
            return True
        except RuntimeError:
            breaker.record_failure()
            return False
    else:
        breaker.record_failure()
        return False

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a model pool with a modest RAM ceiling
    mp = ModelPool(ram_ceiling_mb=2000)

    # Define two models of differing size
    small_model = ModelTier(name="tiny_gpt", ram_mb=300, tier="small")
    large_model = ModelTier(name="mega_gpt", ram_mb=1800, tier="large")

    # Create a morphology that yields a moderate priority
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Register an endpoint
    endpoint = EngineEndpoint(
        engine_id="cpu_fairyfuse_ternary",
        channel="cpu_fairyfuse_ternary",
        residency="local",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="http://localhost:8000",
        capabilities=["nlp", "generation"],
        morphology=morph,
    )
    ep_pool = HybridEngineEndpointPool(failure_threshold=2)
    ep_pool.register(endpoint)

    # Attempt to load the small model (should succeed)
    success_small = load_model_hybrid(ep_pool, mp, "cpu_fairyfuse_ternary", small_model, schedule_alpha=0.6)
    print(f"Loading small model succeeded: {success_small}")

    # Attempt to load the large model (should fail due to RAM limit)
    success_large = load_model_hybrid(ep_pool, mp, "cpu_fairyfuse_ternary", large_model, schedule_alpha=0.6)
    print(f"Loading large model succeeded: {success_large}")

    # Show final RAM usage
    print(f"Total RAM used after operations: {mp._used()} MB")