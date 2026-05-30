# DARWIN HAMMER — match 1, survivor 1
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py (gen1)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:22:20Z

"""
hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4_state_space_duality.py

This module integrates the governing equations of two parent algorithms:
- hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py
- state_space_duality.py

The mathematical bridge between these two structures is the use of state space models (SSMs) to represent the state transitions of engine endpoints.
The SSMs are then used to compute the semiseparable causal matrix, which is applied to a sequence of input tokens to produce output projections.
The health score of an engine endpoint, which depends on its morphology and failure rate, is used to weight the output projections.
This allows the system to adaptively select the most suitable engine endpoint based on their current health scores.

The hybrid operation is demonstrated through three functions: hybrid_ssm_step, hybrid_ssm_sequential, and hybrid_ssm_parallel.
"""

import math
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List
import random
import sys
import pathlib

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

    def as_dict(self) -> Dict[str, any]:
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
        self.endpoints: Dict[str, EngineEndpoint] = {
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
        self.breakers: Dict[str, EndpointCircuitBreaker] = {
            k: EndpointCircuitBreaker(failure_threshold) for k in self.endpoints
        }

    def available(self) -> List[EngineEndpoint]:
        return [
            self.endpoints[k]
            for k, b in self.breakers.items()
            if b.allow()
        ]

    def _failure_rate(self, engine_id: str) -> float:
        b = self.breakers[engine_id]
        return min(1.0, b.failures / b.failure_threshold)

    def health_score(self, endpoint: EngineEndpoint) -> float:
        fr = self._failure_rate(endpoint.engine_id)
        rp = recovery_priority(endpoint.morphology)
        return (1 - fr) * (1 - rp)

    def select_endpoint(self) -> EngineEndpoint:
        available_endpoints = self.available()
        if not available_endpoints:
            raise ValueError("No available endpoints")
        return max(available_endpoints, key=self.health_score)


def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    endpoint: EngineEndpoint,
) -> tuple[np.ndarray, np.ndarray]:
    """Single hybrid SSM step.

    Parameters
    ----------
    h : (state_dim,)       current hidden state
    x : (input_dim,)       current input token
    A : (state_dim, state_dim)   state-transition matrix (diagonal ok)
    B : (state_dim, input_dim)   input projection
    C : (output_dim, state_dim)  output projection
    endpoint : EngineEndpoint    selected engine endpoint

    Returns
    -------
    h_new : (state_dim,)
    y     : (output_dim,)
    """
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y * self.health_score(endpoint)


def hybrid_ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    endpoint: EngineEndpoint,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run hybrid SSM sequentially over a sequence.

    Parameters
    ----------
    x_seq : (T, input_dim)
    A     : (state_dim, state_dim)   shared across time steps
    B     : (state_dim, input_dim)   shared across time steps
    C     : (output_dim, state_dim)  shared across time steps
    endpoint : EngineEndpoint    selected engine endpoint
    h0    : (state_dim,) or None     initial state; zeros if None

    Returns
    -------
    Y : (T, output_dim)
    """
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros(state_dim) if h0 is None else h0.copy()
    outputs = []
    for t in range(T):
        h, y = hybrid_ssm_step(h, x_seq[t], A, B, C, endpoint)
        outputs.append(y)
    return np.stack(outputs, axis=0)


def hybrid_ssm_parallel(
    x_seq: np.ndarray,
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
    endpoint: EngineEndpoint,
) -> np.ndarray:
    """Parallel hybrid semiseparable form: Y = M X.

    Builds the full (T, T) causal semiseparable matrix M then computes
    Y = M x_seq.  Equivalent to hybrid_ssm_sequential but fully parallelisable.

    Parameters
    ----------
    x_seq : (T, 1)          scalar-per-step inputs (1-D duality form)
    A_seq : (T, state_dim, state_dim)
    B_seq : (T, state_dim, 1)
    C_seq : (T, 1, state_dim)
    endpoint : EngineEndpoint    selected engine endpoint

    Returns
    -------
    Y : (T, 1)
    """
    M = np.zeros((x_seq.shape[0], x_seq.shape[0]))
    for i in range(x_seq.shape[0]):
        for j in range(i + 1):
            P = np.eye(A_seq.shape[1])
            for k in range(j + 1, i + 1):
                P = A_seq[k] @ P
            M[i, j] = float((C_seq[i] @ P @ B_seq[j]).squeeze())
    return M @ x_seq


if __name__ == "__main__":
    pool = HybridEngineEndpointPool()
    selected_endpoint = pool.select_endpoint()
    print(selected_endpoint.engine_id)
    x_seq = np.random.rand(10, 1)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 1)
    C = np.random.rand(1, 4)
    A_seq = np.stack([A] * x_seq.shape[0], axis=0)
    B_seq = np.stack([B] * x_seq.shape[0], axis=0)
    C_seq = np.stack([C] * x_seq.shape[0], axis=0)
    y = hybrid_ssm_parallel(x_seq, A_seq, B_seq, C_seq, selected_endpoint)
    print(y)