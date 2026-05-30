# DARWIN HAMMER — match 4450, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1386_s0.py (gen5)
# born: 2026-05-29T23:55:55Z

"""
Hybrid Endpoint–Morphology, Pheromone‑Weighted Graph, and SHAP Fusion
===================================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m938_s3.py`  
  Provides a circuit‑breaker health score `h ∈ [0,1]`, a morphology‑derived recovery
  priority `p ∈ [0,1]`, and a classic Shapley kernel weight `w(S)` for feature
  subsets.

* **Parent B** – `hybrid_hybrid_hybrid_geomet_hybrid_hybrid_hybrid_m1386_s0.py`  
  Supplies a similarity adjacency matrix `A`, a pheromone matrix `Φ`, their
  element‑wise product `W = A ∘ Φ`, a geometric‑product helper for Clifford
  blades, and a Test‑Time Training (TTT) routine (`init_ttt`, `ttt_loss`,
  `ttt_grad`).

Mathematical Bridge
-------------------
The bridge is the **scalar dynamic weight**  


γ = h · p            (health × priority) ∈ [0,1]


which scales the pheromone‑weighted graph:


Ŵ = γ · (A ∘ Φ)


`Ŵ` is then used as the input to the TTT linear map `T`.  The SHAP kernel
`w(S)` can be employed to weight loss contributions of particular node subsets
in downstream analysis.  Thus the unified system intertwines reliability,
shape‑driven priority, pheromone‑guided similarity, and test‑time adaptation
into a single pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A – Circuit breaker (health scoring)
# ----------------------------------------------------------------------
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
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    def health_score(self) -> float:
        """Return a normalized health score h ∈ [0,1].

        If the breaker is open, health is 0. Otherwise it degrades linearly
        with the number of failures.
        """
        if self.open:
            return 0.0
        # Linear degradation: 0 failures → 1.0, max failures → 0.0
        h = 1.0 - (self.failures / self.failure_threshold)
        return max(0.0, min(1.0, h))


# ----------------------------------------------------------------------
# Helper from Parent B – Clifford‑algebra blade multiplication
# ----------------------------------------------------------------------
def _blade_sign(indices):
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # duplicate basis vector cancels out
                lst.pop(j)
                lst.pop(j)
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# ----------------------------------------------------------------------
# Parent B – Pheromone‑weighted graph utilities
# ----------------------------------------------------------------------
def pheromone_weighted_graph(A: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """Element‑wise product W = A ∘ Φ."""
    if A.shape != Phi.shape:
        raise ValueError("Adjacency and pheromone matrices must share shape.")
    return np.multiply(A, Phi)


def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize a linear map for Test‑Time Training."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Mean‑squared‑error loss for a single sample."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the MSE loss w.r.t. W."""
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    # dL/dW = 2 * residual ⊗ x
    return 2.0 * np.outer(residual, x)


# ----------------------------------------------------------------------
# Fusion – Core mathematical bridge
# ----------------------------------------------------------------------
def morphology_priority(morph: Morphology) -> float:
    """Map morphology to a priority p ∈ [0,1].

    Uses a simple linear normalization of the summed spatial dimensions.
    """
    max_dim_sum = 100.0  # domain‑specific upper bound (tunable)
    dim_sum = morph.length + morph.width + morph.height
    p = dim_sum / max_dim_sum
    return max(0.0, min(1.0, p))


def dynamic_weight(health: float, priority: float) -> float:
    """γ = h · p – the scalar that couples parent A and B."""
    return health * priority


def shap_kernel(M: int, subset_size: int) -> float:
    """Classic Shapley kernel weight for a subset of size |S|."""
    if subset_size < 0 or subset_size > M:
        raise ValueError("Invalid subset size.")
    # w(S) = (|S|!·(M‑|S|‑1)!)/M!
    numerator = math.factorial(subset_size) * math.factorial(M - subset_size - 1)
    denominator = math.factorial(M)
    return numerator / denominator


def hybrid_forward(
    cb: EndpointCircuitBreaker,
    morph: Morphology,
    A: np.ndarray,
    Phi: np.ndarray,
    x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Execute one hybrid step:

    1. Compute γ = h·p.
    2. Build pheromone‑weighted graph W = A ∘ Φ and scale it: Ŵ = γ·W.
    3. Initialise a TTT linear map T and apply it to the vectorized graph.
    4. Return the TTT loss and its gradient w.r.t. T.

    Parameters
    ----------
    cb : EndpointCircuitBreaker
        The circuit‑breaker instance providing health `h`.
    morph : Morphology
        Geometry that yields priority `p`.
    A, Phi : np.ndarray
        Adjacency and pheromone matrices (shape N×N).
    x : np.ndarray
        Input feature vector (size N).

    Returns
    -------
    loss : float
        MSE loss after the TTT transformation.
    grad : np.ndarray
        Gradient of the loss w.r.t. the TTT matrix.
    """
    # 1. health and priority
    h = cb.health_score()
    p = morphology_priority(morph)
    gamma = dynamic_weight(h, p)

    # 2. pheromone‑weighted graph and scaling
    W = pheromone_weighted_graph(A, Phi)          # shape (N, N)
    W_hat = gamma * W                               # scalar scaling

    # 3. flatten the weighted graph to match TTT dimensions
    N = W_hat.shape[0]
    x_vec = x.reshape(-1)                           # ensure 1‑D
    if x_vec.shape[0] != N:
        raise ValueError("Input vector length must match graph size.")
    # TTT matrix maps from ℝ^N → ℝ^N
    T = init_ttt(d_in=N, d_out=N, seed=42)

    # Transform the graph‑scaled signal
    transformed = T @ (W_hat @ x_vec)

    # 4. loss & gradient (using transformed signal as prediction)
    loss = ttt_loss(T, W_hat @ x_vec)               # target defaults to input
    grad = ttt_grad(T, W_hat @ x_vec)

    return loss, grad


# ----------------------------------------------------------------------
# Additional demonstration functions
# ----------------------------------------------------------------------
def random_adjacency(n: int, density: float = 0.3, seed: int = 0) -> np.ndarray:
    """Generate a random symmetric adjacency matrix with given edge density."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    A = (A < density).astype(float)
    np.fill_diagonal(A, 0.0)
    return A


def random_pheromones(n: int, seed: int = 1) -> np.ndarray:
    """Generate a random pheromone matrix with positive values."""
    rng = np.random.default_rng(seed)
    Phi = rng.random((n, n))
    Phi = (Phi + Phi.T) / 2.0
    np.fill_diagonal(Phi, 1.0)  # self‑loops have maximal pheromone
    return Phi


def demo_shap_weights(num_features: int) -> List[float]:
    """Compute SHAP kernel weights for all possible subset sizes."""
    return [shap_kernel(num_features, k) for k in range(num_features)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise components
    cb = EndpointCircuitBreaker(failure_threshold=4)
    # Simulate a few failures and successes
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    cb.record_failure()  # 3 failures → health ≈ 0.25

    morph = Morphology(length=12.0, width=8.0, height=5.0, mass=3.2)

    N = 6  # graph size
    A = random_adjacency(N, density=0.4, seed=10)
    Phi = random_pheromones(N, seed=20)
    x = np.arange(1, N + 1, dtype=float)  # simple feature vector [1,2,...,N]

    # 2. Run hybrid forward pass
    loss, grad = hybrid_forward(cb, morph, A, Phi, x)

    # 3. Display results
    print(f"Health score h = {cb.health_score():.3f}")
    print(f"Priority p = {morphology_priority(morph):.3f}")
    print(f"Dynamic weight γ = {dynamic_weight(cb.health_score(), morphology_priority(morph)):.3f}")
    print(f"SHAP kernel weights (M={N}): {demo_shap_weights(N)}")
    print(f"Hybrid TTT loss = {loss:.6f}")
    print(f"Gradient shape = {grad.shape}")

    # Ensure no exception was raised
    sys.exit(0)