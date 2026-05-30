# DARWIN HAMMER — match 3572, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s1.py (gen3)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s5.py (gen2)
# born: 2026-05-29T23:50:44Z

"""Hybrid audit‑pruning & VRAM allocation with circuit‑breaker control.

Parents:
- hybrid_hybrid_ternary_lens__hybrid_hybrid_model__m716_s1.py (audit‑pruning, produces a weight
  vector **w** from a vendor manifest and a time‑decaying probability matrix **P(t)**).
- hybrid_hybrid_endpoint_circ_hybrid_path_signatur_m595_s5.py (circuit‑breaker, morphology
  indices, lead‑lag path transform and level‑1 signature).

Mathematical bridge:
1. The audit summary yields a count vector **s ∈ ℝᵏ** (k classifications). Normalising **s**
   gives a weight vector **w** that expresses priority for each task.
2. A morphology descriptor **m** provides shape indices *σ* (sphericity) and *φ* (flatness).
   Their product **μ = σ·φ** is a scalar quality factor.
3. Priorities are fused multiplicatively: **p_i = w_i·μ**.
4. A decreasing‑rate pruning schedule **α(t) = (1‑β)ᵗ**, β∈(0,1), modulates the probability
   matrix **P(t) = diag(p)·α(t)**, producing a stochastic allocation of the available VRAM.
5. The allocated VRAM over time forms a path **X(t)**. Applying the lead‑lag transform
   yields **Y = L·X**, and the level‑1 signature **Δ = X_T‑X_0** summarises net consumption.
6. The signature magnitude feeds an **EndpointCircuitBreaker**: if consumption exceeds a
   threshold the breaker trips, pausing further allocation.

The module implements this fused pipeline with three public functions:
- `audit_weights(manifest) → np.ndarray`
- `vram_allocation(weights, total_vram, steps, beta) → np.ndarray`
- `run_hybrid_cycle(manifest, morphology, total_vram, steps, beta, breaker) → dict`
"""

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Endpoint circuit breaker and morphology utilities
# ----------------------------------------------------------------------
def now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple failure‑counter circuit breaker."""
    def __init__(self, failure_threshold: int = 3, consumption_threshold: float = 1e9):
        self.failure_threshold = failure_threshold
        self.consumption_threshold = consumption_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""
        self.stress_history: List[float] = []

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def evaluate_consumption(self, consumption: float) -> None:
        """Trip the breaker if consumption exceeds the configured threshold."""
        self.stress_history.append(consumption)
        if consumption > self.consumption_threshold:
            self.record_failure()
        else:
            self.record_success()

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "consumption_threshold": self.consumption_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
            "stress_history": self.stress_history,
        }


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float  # mass is not used in the current indices but kept for extensibility


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Parent‑B lead‑lag transform."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Parent‑B level‑1 signature (net displacement)."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time × dim)")
    return path[-1] - path[0]


# ----------------------------------------------------------------------
# Parent A – Audit‑pruning helpers (simplified for the fusion)
# ----------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest. If the file does not exist, return a synthetic example."""
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Synthetic fallback – 5 classifications with random counts
    return {
        "usable_now": random.randint(10, 100),
        "research_only": random.randint(5, 50),
        "needs_conversion": random.randint(0, 30),
        "unsafe_for_fastpath": random.randint(0, 20),
        "unsupported": random.randint(0, 10),
    }


def audit_weights(manifest: dict[str, Any]) -> np.ndarray:
    """
    Convert the manifest counts into a normalised weight vector **w**.
    The order follows the CLASSIFICATIONS constant.
    """
    classifications = [
        "usable_now",
        "research_only",
        "needs_conversion",
        "unsafe_for_fastpath",
        "unsupported",
    ]
    counts = np.array([float(manifest.get(c, 0)) for c in classifications], dtype=float)
    total = counts.sum()
    if total == 0:
        # Avoid division by zero – assign uniform weights
        return np.full_like(counts, 1.0 / len(counts))
    return counts / total  # Normalised priority weights


def decreasing_pruning_factor(step: int, beta: float) -> float:
    """
    Decreasing‑rate pruning schedule α(t) = (1‑β)^t.
    beta ∈ (0, 1) controls the decay speed.
    """
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")
    return (1.0 - beta) ** step


def vram_allocation(
    weights: np.ndarray,
    total_vram: float,
    steps: int,
    beta: float = 0.05,
) -> np.ndarray:
    """
    Produce a (steps × k) allocation matrix **A** where each row corresponds to a time step.
    Allocation at step t:  A[t] = total_vram * diag(weights) * α(t)
    The rows sum to total_vram·α(t); a stochastic normalisation can be applied if required.
    """
    k = weights.shape[0]
    allocations = np.empty((steps, k), dtype=float)
    for t in range(steps):
        alpha = decreasing_pruning_factor(t, beta)
        allocations[t] = total_vram * weights * alpha
    return allocations


# ----------------------------------------------------------------------
# Fusion core – combine audit, morphology, allocation, and breaker
# ----------------------------------------------------------------------
def fuse_priority(weights: np.ndarray, morphology: Morphology) -> np.ndarray:
    """
    Fuse audit weights with a morphology quality factor μ = σ·φ.
    The result is a scaled priority vector p_i = w_i * μ.
    """
    sigma = sphericity_index(morphology.length, morphology.width, morphology.height)
    phi = flatness_index(morphology.length, morphology.width, morphology.height)
    mu = sigma * phi
    return weights * mu


def run_hybrid_cycle(
    manifest: dict[str, Any],
    morphology: Morphology,
    total_vram: float,
    steps: int,
    beta: float,
    breaker: EndpointCircuitBreaker,
) -> dict[str, Any]:
    """
    Execute one full hybrid cycle:
    1. Derive audit weights.
    2. Fuse with morphology to obtain priorities.
    3. Allocate VRAM across `steps` using the decreasing pruning schedule.
    4. Transform the allocation path with the lead‑lag operator.
    5. Compute the level‑1 signature (net VRAM change) and feed it to the circuit breaker.
    Returns a dictionary with intermediate results for inspection.
    """
    # 1. Audit weights
    w = audit_weights(manifest)

    # 2. Fuse with morphology
    p = fuse_priority(w, morphology)

    # 3. VRAM allocation matrix (steps × k)
    allocation_matrix = vram_allocation(p, total_vram, steps, beta)

    # 4. Build a path: cumulative VRAM per classification over time
    #    Shape (steps, k)
    cumulative_path = np.cumsum(allocation_matrix, axis=0)

    # 5. Lead‑lag transform (produces shape (2*steps‑1, 2*k))
    transformed = lead_lag_transform(cumulative_path)

    # 6. Level‑1 signature (net change per classification)
    signature = signature_level1(cumulative_path)

    # 7. Total net VRAM consumption (scalar)
    total_consumption = float(np.sum(signature))

    # 8. Update circuit breaker
    breaker.evaluate_consumption(total_consumption)

    return {
        "weights": w.tolist(),
        "fused_priorities": p.tolist(),
        "allocation_matrix": allocation_matrix.tolist(),
        "cumulative_path": cumulative_path.tolist(),
        "lead_lag": transformed.tolist(),
        "signature": signature.tolist(),
        "total_consumption": total_consumption,
        "breaker_state": breaker.as_dict(),
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 0. Prepare dummy inputs
    dummy_manifest_path = Path("dummy_vendor_manifest.json")
    manifest = load_manifest(dummy_manifest_path)

    dummy_morphology = Morphology(
        length=12.0,
        width=8.0,
        height=4.0,
        mass=3.5,
    )

    total_vram_bytes = 8 * 1024 ** 3  # 8 GiB
    steps = 10
    beta = 0.07

    breaker = EndpointCircuitBreaker(failure_threshold=2, consumption_threshold=1e10)

    # Run the hybrid pipeline
    result = run_hybrid_cycle(
        manifest=manifest,
        morphology=dummy_morphology,
        total_vram=total_vram_bytes,
        steps=steps,
        beta=beta,
        breaker=breaker,
    )

    # Simple sanity output
    print(json.dumps({
        "breaker_open": result["breaker_state"]["open"],
        "total_consumption_GiB": result["total_consumption"] / (1024 ** 3),
        "final_allocation": result["allocation_matrix"][-1],
    }, indent=2))