# DARWIN HAMMER — match 147, survivor 0
# gen: 3
# parent_a: endpoint_circuit_breaker.py (gen0)
# parent_b: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s2.py (gen2)
# born: 2026-05-29T23:27:09Z

"""Hybrid Endpoint‑Circuit‑Breaker + Liquid‑Time‑Constant Diffusion Forcing (HB‑LTC‑DF)

Parent A: `endpoint_circuit_breaker.py` – provides per‑endpoint circuit‑breaker
state and a pool of EngineEndpoint objects.

Parent B: `hybrid_liquid_time_c_diffusion_forcing_m16_s2.py` – defines a Liquid
Time‑Constant (LTC) recurrent cell whose dynamics are modulated by a MinHash
similarity `s`. The similarity determines a diffusion timestep `t_i` that
injects noise into the input vector.

Mathematical bridge
-------------------
For each engine endpoint `e` we maintain a state vector `x_e`.  The circuit‑breaker
gate `g_e ∈ {0,1}` (open = 0, closed = 1) multiplies the LTC update, i.e.

    dx_e/dt = g_e * ( -(1/τ + f_e)·x_e + f_e·A_e )

where  

    f_e = σ( W·[x_e; I; s_e] + b )                (σ = sigmoid)  
    s_e = similarity( signature(tokens), accumulated_signature_e )  

The similarity `s_e` is the MinHash Jaccard estimate used in Parent B.  It also
drives the diffusion timestep

    t_i = round( (1 - s_e) * T )

and the noisy input injected into the LTC cell

    x_noisy_i = √α[t_i]·I_i + √(1-α[t_i])·ε_i .

The per‑token loss is weighted by λ(t_i)=1/(1+t_i); the aggregate loss for an
endpoint determines whether its circuit‑breaker records a success or a
failure.  Thus the routing logic of Parent A and the LTC‑DF dynamics of Parent B
are mathematically fused into a single closed‑loop system.

The module below implements this hybrid system with three public functions:
`minhash_signature`, `ltc_diffusion_step`, and `process_pool`.  A smoke test
executes a single iteration over a dual‑engine pool."""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – endpoint circuit‑breaker primitives (unchanged)
# ----------------------------------------------------------------------


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
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


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
    outbound_state: str = "draft_only"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DualEngineEndpointPool:
    """Pool of two hard‑coded engine endpoints (CPU + GPU)."""

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
            ),
            "gpu_q4_deepseek": EngineEndpoint(
                engine_id="gpu_q4_deepseek",
                channel="gpu_q4_deepseek",
                residency="always_on",
                runtime="cuda_fp16",
                resource_class="gpu_fp16",
                always_on=True,
                endpoint="ALGOS/deepseek_q4.py",
                capabilities=[
                    "semantic_stream",
                    "high_throughput",
                    "low_latency",
                    "telemetry",
                ],
            ),
        }
        self.breakers: Dict[str, EndpointCircuitBreaker] = {
            e: EndpointCircuitBreaker(failure_threshold) for e in self.endpoints
        }

    def available(self) -> List[str]:
        """Return endpoint IDs whose circuit‑breaker is closed (allowing work)."""
        return [e for e in self.endpoints if self.breakers[e].allow()]

    def record_success(self, endpoint_id: str) -> None:
        self.breakers[endpoint_id].record_success()

    def record_failure(self, endpoint_id: str) -> None:
        self.breakers[endpoint_id].record_failure()


# ----------------------------------------------------------------------
# Parent B – MinHash and LTC‑Diffusion utilities (adapted)
# ----------------------------------------------------------------------


MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length `k` for the given token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [
        min(_hash(seed, token) for token in toks) for seed in range(k)
    ]


def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signature lengths must match")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ltc_diffusion_step(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    tau: float,
    A: np.ndarray,
    similarity: float,
    T: int,
    alpha_cum: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Perform one LTC‑Diffusion update.

    Returns the new state vector and the scalar diffusion‑forcing loss.
    """
    # ---- compute gating factor f -------------------------------------------------
    aug = np.concatenate([x, I, np.array([similarity])])
    f = sigmoid(W @ aug + b)  # shape (1,) or scalar
    f = float(f.squeeze())

    # ---- deterministic LTC dynamics ------------------------------------------------
    dx = -(1.0 / tau + f) * x + f * A
    x_next = x + dx  # using dt = 1 for simplicity

    # ---- diffusion forcing --------------------------------------------------------
    # Compute per‑dimension timesteps from similarity
    t_i = int(round((1.0 - similarity) * T))
    t_i = max(0, min(T, t_i))  # clamp

    alpha = alpha_cum[t_i]
    sqrt_alpha = math.sqrt(alpha)
    sqrt_one_minus_alpha = math.sqrt(1.0 - alpha)

    eps = np.random.randn(*I.shape)
    I_noisy = sqrt_alpha * I + sqrt_one_minus_alpha * eps

    # loss weighting λ(t) = 1 / (1 + t)
    lam = 1.0 / (1.0 + t_i)
    loss = lam * np.mean((I_noisy - A) ** 2)

    return x_next, loss


# ----------------------------------------------------------------------
# Hybrid functions that combine the two parents
# ----------------------------------------------------------------------


def process_pool(
    pool: DualEngineEndpointPool,
    tokens: List[str],
    input_vec: np.ndarray,
    params: dict,
) -> Dict[str, float]:
    """
    Run a single hybrid LTC‑DF step on all available endpoints.

    `params` must contain:
        - W: weight matrix (np.ndarray)
        - b: bias vector (np.ndarray)
        - tau: float
        - A: target vector (np.ndarray)
        - T: int (max diffusion timestep)
        - alpha_cum: List[float] of length T+1 (cumulative diffusion schedule)
        - loss_threshold: float (below which a success is recorded)

    Returns a mapping endpoint_id → loss value.
    """
    # Global (across pool) accumulated signature; in a real system each endpoint
    # could keep its own, but for demonstration we share one.
    if not hasattr(pool, "_global_signature"):
        pool._global_signature = minhash_signature(tokens, k=64)

    # Update global signature with new tokens (simple union)
    new_sig = minhash_signature(tokens, k=64)
    # Estimate similarity between previous and new signatures
    similarity = jaccard_estimate(pool._global_signature, new_sig)

    # Store the new signature for the next round
    pool._global_signature = new_sig

    losses: Dict[str, float] = {}
    for eid in pool.available():
        # Retrieve per‑endpoint state; initialise if absent
        state_attr = f"_state_{eid}"
        if not hasattr(pool, state_attr):
            setattr(pool, state_attr, np.zeros_like(input_vec))
        x = getattr(pool, state_attr)

        # Perform LTC‑Diffusion update gated by the circuit‑breaker (implicitly open=0)
        x_next, loss = ltc_diffusion_step(
            x,
            input_vec,
            params["W"],
            params["b"],
            params["tau"],
            params["A"],
            similarity,
            params["T"],
            params["alpha_cum"],
        )
        setattr(pool, state_attr, x_next)
        losses[eid] = loss

        # Record success/failure based on loss
        if loss <= params["loss_threshold"]:
            pool.record_success(eid)
        else:
            pool.record_failure(eid)

    return losses


def build_random_params(dim: int, T: int = 10) -> dict:
    """
    Utility to build a random parameter dictionary for the hybrid algorithm.
    The diffusion schedule ᾱ[t] is a simple linear schedule from 0.1 to 0.9.
    """
    rng = np.random.default_rng()
    W = rng.normal(size=(1, dim * 2 + 1))  # x, I, similarity
    b = rng.normal(size=(1,))
    tau = float(rng.uniform(0.5, 2.0))
    A = rng.normal(size=(dim,))
    alpha_cum = np.linspace(0.1, 0.9, T + 1).tolist()
    loss_threshold = 0.5  # arbitrary
    return {
        "W": W,
        "b": b,
        "tau": tau,
        "A": A,
        "T": T,
        "alpha_cum": alpha_cum,
        "loss_threshold": loss_threshold,
    }


def demo_hybrid_step() -> None:
    """
    Demonstrates a single hybrid iteration:
    - creates a dual‑engine pool,
    - builds random parameters,
    - runs `process_pool` on synthetic tokens and input vector,
    - prints per‑endpoint loss and circuit‑breaker status.
    """
    pool = DualEngineEndpointPool(failure_threshold=2)
    tokens = ["alpha", "beta", "gamma", "delta"]
    dim = 16
    input_vec = np.random.randn(dim)

    params = build_random_params(dim, T=12)

    losses = process_pool(pool, tokens, input_vec, params)

    for eid, loss in losses.items():
        breaker = pool.breakers[eid]
        status = "CLOSED" if breaker.allow() else "OPEN"
        print(f"Endpoint {eid:>20}: loss={loss: .4f}, breaker={status}, failures={breaker.failures}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    demo_hybrid_step()