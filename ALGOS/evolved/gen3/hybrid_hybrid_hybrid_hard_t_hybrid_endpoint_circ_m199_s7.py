# DARWIN HAMMER — match 199, survivor 7
# gen: 3
# parent_a: hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s1.py (gen2)
# parent_b: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py (gen1)
# born: 2026-05-29T23:27:41Z

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List, Any

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        return (self._used() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: ModelTier) -> None:
        if self.is_loaded(model.name):
            return  
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used()} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)

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

def linear_interpolant(start_vec: np.ndarray, end_vec: np.ndarray, alpha: float) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    return (1.0 - alpha) * start_vec + alpha * end_vec

def hybrid_priority(endpoint: EngineEndpoint, model: ModelTier, model_pool: ModelPool) -> float:
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
    endpoint = endpoint_pool.get(engine_id)
    circuit_breaker = endpoint_pool.breaker(engine_id)
    if not circuit_breaker.allow():
        return False

    priority = hybrid_priority(endpoint, model, model_pool)
    current_ram_usage = np.array([model_pool._used()])
    target_ram_usage = np.array([model_pool._used() + model.ram_mb])
    interpolated_ram_usage = linear_interpolant(current_ram_usage, target_ram_usage, schedule_alpha)

    if priority > 0.3 and interpolated_ram_usage <= model_pool.ram_ceiling_mb:
        try:
            model_pool.load(model)
            circuit_breaker.record_success()
            return True
        except RuntimeError:
            circuit_breaker.record_failure()
            return False

    return False