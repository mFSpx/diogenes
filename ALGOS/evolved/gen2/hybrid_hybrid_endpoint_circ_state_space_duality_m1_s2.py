# DARWIN HAMMER — match 1, survivor 2
# gen: 2
# parent_a: hybrid_endpoint_circuit_bre_serpentina_self_righ_m18_s4.py (gen1)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:22:20Z

"""
Hybrid Endpoint‑SSM Engine
-------------------------

This module fuses two distinct parent algorithms:

* **Parent A** – ``HybridEngineEndpointPool`` selects an engine endpoint
  using a *health_score* that combines a failure‑rate term (circuit‑breaker)
  with a morphology‑derived *recovery priority*.

* **Parent B** – ``state_space_duality`` provides a sequential‑vs‑parallel
  representation of a linear state‑space model (SSM).  The parallel form
  builds a *semiseparable* lower‑triangular matrix **M** such that
  **Y = M·X**.

### Mathematical bridge

We treat each endpoint as a *state dimension* of an SSM.  For a request
sequence of length **T** we define per‑step matrices:

* **Aₜ** – diagonal decay matrix whose entries are the *failure‑rate*
  (``breaker.failures / breaker.failure_threshold``).  It models how the
  “health” of an endpoint degrades from one request to the next.

* **Bₜ** – column vector encoding the *morphology* of each endpoint.
  We use the *righting_time_index* (a monotonic function of flatness) as
  the scalar projection for every endpoint; the same vector is broadcast
  to all time steps.

* **Cₜ** – row vector of the *health_score* (``(1‑fr)*(1‑rp)``) for each
  endpoint at step *t*.  This is exactly the output projection used in
  Parent A.

With these definitions the SSM update


hₜ   = Aₜ·hₜ₋₁ + Bₜ·xₜ
yₜ   = Cₜ·hₜ


produces a scalar **yₜ** that is a weighted combination of the endpoint
healths.  By the semiseparable duality, the same result can be obtained
with a dense causal matrix **M** and a single matrix‑vector product
``Y = M·X``.  The hybrid algorithm therefore:

1. Computes the health‑related quantities from the endpoint pool.
2. Builds the per‑step SSM matrices.
3. Uses the *parallel* semiseparable form to obtain a score for every
   request in **O(T²)** but fully vectorised.
4. Selects, at each time step, the endpoint with the highest instantaneous
   contribution to the score.

The three public functions below illustrate the hybrid workflow.
"""

import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – endpoint pool and morphology utilities
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
    """
    Holds a set of EngineEndpoint objects together with per‑endpoint
    circuit breakers.  Provides health‑score computation used later by the
    hybrid SSM.
    """

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

    # ------------------------------------------------------------------
    # API used by the hybrid SSM
    # ------------------------------------------------------------------

    def _failure_rate(self, engine_id: str) -> float:
        b = self.breakers[engine_id]
        return min(1.0, b.failures / b.failure_threshold)

    def health_score(self, endpoint: EngineEndpoint) -> float:
        fr = self._failure_rate(endpoint.engine_id)
        rp = recovery_priority(endpoint.morphology)
        return (1 - fr) * (1 - rp)

    def health_vector(self) -> np.ndarray:
        """
        Returns a column vector (N, 1) of health scores for all endpoints,
        ordered by the internal dict key order.
        """
        scores = [self.health_score(ep) for ep in self.endpoints.values()]
        return np.array(scores).reshape(-1, 1)

    def failure_rate_vector(self) -> np.ndarray:
        """
        Returns a column vector (N, 1) of failure rates for all endpoints.
        """
        rates = [self._failure_rate(eid) for eid in self.endpoints]
        return np.array(rates).reshape(-1, 1)

    def morphology_vector(self) -> np.ndarray:
        """
        Returns a column vector (N, 1) where each entry is the
        ``righting_time_index`` of the endpoint's morphology.
        """
        vals = [
            righting_time_index(ep.morphology) for ep in self.endpoints.values()
        ]
        return np.array(vals).reshape(-1, 1)

    def endpoint_ids(self) -> List[str]:
        return list(self.endpoints.keys())

    # ------------------------------------------------------------------
    # Simple simulation helpers (not required for the hybrid core)
    # ------------------------------------------------------------------

    def simulate_random_failure(self, prob: float = 0.1) -> None:
        for eid, breaker in self.breakers.items():
            if random.random() < prob:
                breaker.record_failure()
            else:
                breaker.record_success()


# ----------------------------------------------------------------------
# Parent B – state‑space model and semiseparable duality
# ----------------------------------------------------------------------


def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step."""
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y


def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run SSM sequentially over a sequence."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros(state_dim) if h0 is None else h0.copy()
    outputs = []
    for t in range(T):
        h, y = ssm_step(h, x_seq[t], A, B, C)
        outputs.append(y)
    return np.stack(outputs, axis=0)


def semiseparable_matrix(
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
) -> np.ndarray:
    """
    Build the (T, T) causal semiseparable matrix for scalar I/O.
    """
    T = A_seq.shape[0]
    state_dim = A_seq.shape[1]
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            # Compute product A_{j+1} .. A_i (identity if j == i)
            P = np.eye(state_dim)
            for k in range(j + 1, i + 1):
                P = A_seq[k] @ P
            M[i, j] = float((C_seq[i] @ P @ B_seq[j]).squeeze())
    return M


def ssm_parallel(
    x_seq: np.ndarray,
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
) -> np.ndarray:
    """
    Parallel semiseparable form: Y = M·X.
    """
    M = semiseparable_matrix(A_seq, B_seq, C_seq)   # (T, T)
    Y = M @ x_seq                                    # (T, 1)
    return Y


# ----------------------------------------------------------------------
# Hybrid layer – mapping endpoint health to an SSM
# ----------------------------------------------------------------------


def build_hybrid_matrices(
    pool: HybridEngineEndpointPool,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct per‑step SSM matrices (A_seq, B_seq, C_seq) that embed the
    endpoint pool state.

    * **A_seq[t]** – diagonal matrix with entries (1‑failure_rate) for each
      endpoint.  This models health decay between consecutive requests.

    * **B_seq[t]** – column vector of morphology‑derived righting‑time indices.
      The same vector is reused for every time step because morphology does
      not change during a short request burst.

    * **C_seq[t]** – row vector of health scores.  These act as the output
      projection at step *t*.
    """
    n_ep = len(pool.endpoints)

    # Failure‑rate and health vectors are static for the duration of the
    # sequence (they could be refreshed each step in a real system).
    fr_vec = pool.failure_rate_vector()          # (N,1)
    health_vec = pool.health_vector()            # (N,1)
    morph_vec = pool.morphology_vector()         # (N,1)

    # Build diagonal A matrices: A = diag(1 - fr)
    A_base = np.diagflat(1.0 - fr_vec)            # (N,N)
    A_seq = np.stack([A_base] * seq_len, axis=0)  # (T,N,N)

    # B is the same for all steps: column vector (N,1)
    B_seq = np.stack([morph_vec] * seq_len, axis=0)  # (T,N,1)

    # C is the health row vector (1,N) per step
    C_seq = np.stack([health_vec.T] * seq_len, axis=0)  # (T,1,N)

    return A_seq, B_seq, C_seq


def hybrid_score_sequence(
    pool: HybridEngineEndpointPool,
    request_seq: np.ndarray,
) -> np.ndarray:
    """
    Given a pool and a scalar request sequence ``request_seq`` of shape
    (T,1), compute the hybrid SSM output ``Y`` using the parallel semiseparable
    form.  ``Y`` contains a single score per request that aggregates the
    health‑aware contributions of all endpoints.
    """
    T = request_seq.shape[0]
    A_seq, B_seq, C_seq = build_hybrid_matrices(pool, T)
    Y = ssm_parallel(request_seq, A_seq, B_seq, C_seq)   # (T,1)
    return Y.squeeze()  # shape (T,)


def hybrid_endpoint_selection(
    pool: HybridEngineEndpointPool,
    request_seq: np.ndarray,
) -> List[str]:
    """
    Perform endpoint selection for each request in ``request_seq``.

    The algorithm proceeds in three stages:

    1. Compute the hybrid SSM scores (scalar per request).
    2. For each time step, combine the score with the per‑endpoint
       health vector to obtain a *local* contribution.
    3. Choose the endpoint with the maximal contribution.

    The returned list has length ``T`` and contains the chosen
    ``engine_id`` for each request.
    """
    T = request_seq.shape[0]
    # Step 1 – global hybrid scores
    global_scores = hybrid_score_sequence(pool, request_seq)  # (T,)

    # Step 2 – per‑endpoint contributions
    health_vec = pool.health_vector().flatten()               # (N,)
    endpoint_ids = pool.endpoint_ids()

    selections: List[str] = []
    for t in range(T):
        # The contribution of endpoint i at step t is:
        #    contrib_i = health_i * global_score_t
        # (the multiplication is a simple scalar weighting)
        contrib = health_vec * global_scores[t]
        best_idx = int(np.argmax(contrib))
        selections.append(endpoint_ids[best_idx])
    return selections


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Initialise a pool with two endpoints.
    pool = HybridEngineEndpointPool()

    # Simulate a few random failures to make the health landscape non‑trivial.
    pool.simulate_random_failure(prob=0.3)

    # Create a dummy request sequence: scalar load per time step.
    T = 10
    rng = np.random.default_rng(123)
    request_seq = rng.standard_normal((T, 1))

    # Run the hybrid selection.
    chosen = hybrid_endpoint_selection(pool, request_seq)

    print("Request sequence (scalar load):")
    print(request_seq.squeeze())
    print("\nChosen endpoints per step:")
    for t, eid in enumerate(chosen):
        print(f" step {t:02d}: {eid}")

    # Verify that the parallel SSM matches the sequential formulation
    # (using the same matrices) as a sanity check.
    A_seq, B_seq, C_seq = build_hybrid_matrices