# DARWIN HAMMER — match 1902, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_hybrid_pherom_m1028_s1.py (gen4)
# parent_b: hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py (gen2)
# born: 2026-05-29T23:39:36Z

"""Hybrid Endpoint‑Circuit‑Breaker + Radial‑Basis‑Function Surrogate with Morphology‑Driven Burst Model

Parents:
- hybrid_hybrid_endpoint_circ_hybrid_hybrid_pherom_m1028_s1.py (EndpointCircuitBreaker, Morphology‑derived sphericity & flatness, burst‑admission model)
- hybrid_hybrid_rbf_surrogate_perceptual_dedupe_m57_s1.py (Gaussian RBF surrogate, linear system solve, perceptual‑hash penalty)

Mathematical bridge:
The morphology produces two scalar descriptors **S** (sphericity) and **F** (flatness).  
These descriptors are appended to the feature vectors fed to the RBF surrogate, making the surrogate’s kernel matrix a function of both the original data and the physical shape.  
The surrogate’s prediction **p(x)** is then used to *modulate* the circuit‑breaker’s failure threshold:


τ = τ₀ + ⌊α·B(S,F,work,drag,urgency) + β·p(x)⌋


where **B** is the burst‑admission score and **α,β** are scaling constants.  
Conversely, the circuit‑breaker’s open/closed state influences the perceptual‑hash penalty **hpen** that is subtracted from **τ**, closing a feedback loop between the two topologies.

The module below implements this fused system, exposing three core hybrid operations:
1. `compute_morphology_indices`
2. `fit_hybrid_surrogate`
3. `hybrid_event_process`
"""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – EndpointCircuitBreaker & Morphology
# ----------------------------------------------------------------------


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EndpointCircuitBreaker:
    """Simple circuit‑breaker whose failure threshold can be dynamically tuned."""

    def __init__(self, failure_threshold: int = 3):
        self.base_failure_threshold: int = failure_threshold
        self.failure_threshold: int = failure_threshold
        self.failures: int = 0
        self.open: bool = False
        self.last_event_at: str = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """Return True if the breaker is closed (i.e. requests are allowed)."""
        return not self.open

    def adjust_threshold(self, delta: int) -> None:
        """Adjust the current failure threshold by *delta* (can be negative)."""
        self.failure_threshold = max(1, self.base_failure_threshold + delta)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_failure_threshold": self.base_failure_threshold,
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float

    def volume(self) -> float:
        return self.length * self.width * self.height

    def surface_area(self) -> float:
        l, w, h = self.length, self.width, self.height
        return 2 * (l * w + w * h + h * l)


def compute_morphology_indices(morph: Morphology) -> Tuple[float, float]:
    """
    Compute:
    - sphericity S ∈ (0,1]   (1 for a perfect sphere)
    - flatness   F ∈ (0,1]   (1 for a perfect cube, smaller for elongated shapes)

    The formulas are simplified but retain the essential dependence on volume
    and surface area.
    """
    vol = morph.volume()
    surf = morph.surface_area()
    if surf == 0:
        raise ValueError("Surface area cannot be zero.")
    # Classic sphericity definition
    S = (math.pi ** (1.0 / 3.0) * (6 * vol) ** (2.0 / 3.0)) / surf
    # Flatness as ratio of smallest to largest dimension
    dims = [morph.length, morph.width, morph.height]
    F = min(dims) / max(dims) if max(dims) > 0 else 0.0
    return S, F


def burst_score(
    S: float,
    F: float,
    work: float,
    drag: float,
    urgency: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Linear‑combination burst model used to perturb the breaker threshold.
    The coefficients α,β allow external scaling.
    """
    return alpha * (work * S - drag * (1 - F)) + beta * (urgency * S * F)


# ----------------------------------------------------------------------
# Parent B – Radial‑Basis‑Function Surrogate & Perceptual‑Hash Penalty
# ----------------------------------------------------------------------


Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (no external libs)."""
    n = len(b)
    # Augmented matrix
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        # Swap rows
        m[col], m[pivot] = m[pivot], m[col]
        # Normalize pivot row
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        # Eliminate column entries
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    """Gaussian RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Return the surrogate prediction for input vector *x*."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def fit_rbf_surrogate(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> RBFSurrogate:
    """
    Fit a Gaussian RBF surrogate by solving the linear system
    (Φ + λI)w = y, where Φ_{ij}=gaussian(||c_i-c_j||,ε).
    """
    centers = [tuple(map(float, p)) for p in points]
    y = np.array(list(map(float, values)), dtype=float)
    n = len(centers)
    if n == 0 or n != len(y):
        raise ValueError("points and values must be non‑empty and of equal length.")

    # Build kernel matrix Φ
    Phi = np.empty((n, n), dtype=float)
    for i, ci in enumerate(centers):
        for j, cj in enumerate(centers):
            Phi[i, j] = gaussian(euclidean(ci, cj), epsilon)

    # Ridge regularisation
    Phi += ridge * np.eye(n)

    # Solve (Φ)w = y
    w = np.linalg.solve(Phi, y)
    return RBFSurrogate(centers=centers, weights=w.tolist(), epsilon=epsilon)


def perceptual_hash(s: str) -> int:
    """
    Very lightweight perceptual hash: lower 64 bits of built‑in hash,
    forced to be positive for deterministic bit‑counting.
    """
    return hash(s) & ((1 << 64) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two 64‑bit integers."""
    return bin(a ^ b).count("1")


def hash_penalty(event_hash: int, reference_hash: int, scale: float = 0.1) -> int:
    """
    Convert Hamming distance into an integer penalty that will be subtracted
    from the circuit‑breaker threshold. Larger distances → larger penalties.
    """
    dist = hamming_distance(event_hash, reference_hash)
    return int(scale * dist)


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------


def fit_hybrid_surrogate(
    base_points: Iterable[Vector],
    base_values: Iterable[float],
    morph: Morphology,
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """
    Extend each base point with the morphology descriptors (S,F) before fitting.
    This creates a joint feature space where shape influences the surrogate.
    """
    S, F = compute_morphology_indices(morph)
    extended_points = [
        tuple(p) + (S, F) for p in base_points
    ]  # type: ignore[arg-type]
    return fit_rbf_surrogate(extended_points, base_values, epsilon=epsilon)


def hybrid_event_process(
    breaker: EndpointCircuitBreaker,
    surrogate: RBFSurrogate,
    morph: Morphology,
    event_str: str,
    work: float,
    drag: float,
    urgency: float,
    input_vec: Vector,
    reference_hash: str = "reference",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> bool:
    """
    Process a single event:

    1. Compute morphology indices (S,F).
    2. Evaluate burst score B(S,F,work,drag,urgency).
    3. Predict surrogate value p(x) on the *augmented* input vector.
    4. Compute hash‑based penalty hpen.
    5. Adjust breaker threshold τ = τ₀ + ⌊α·B + β·p⌋ – hpen.
    6. Record success/failure based on whether the breaker currently allows the event.

    Returns True if the event was allowed (i.e. success), False otherwise.
    """
    # 1‑2. Morphology & burst
    S, F = compute_morphology_indices(morph)
    B = burst_score(S, F, work, drag, urgency, alpha=alpha, beta=beta)

    # 3. Surrogate prediction on augmented vector
    aug_vec = tuple(input_vec) + (S, F)  # type: ignore[arg-type]
    p = surrogate.predict(aug_vec)

    # 4. Hash penalty
    ev_hash = perceptual_hash(event_str)
    ref_hash = perceptual_hash(reference_hash)
    hpen = hash_penalty(ev_hash, ref_hash, scale=gamma)

    # 5. Threshold adjustment
    delta = int(alpha * B + beta * p) - hpen
    breaker.adjust_threshold(delta)

    # 6. Record outcome
    if breaker.allow():
        breaker.record_success()
        outcome = True
    else:
        breaker.record_failure()
        outcome = False

    # Debug output (optional, can be silenced by redirecting stdout)
    print(
        f"[Hybrid] Event='{event_str[:20]}...', allow={outcome}, "
        f"S={S:.3f}, F={F:.3f}, B={B:.3f}, p={p:.3f}, hpen={hpen}, "
        f"threshold={breaker.failure_threshold}"
    )
    return outcome


def simulate_hybrid_workflow(
    num_events: int = 10,
    seed: int = 42,
) -> None:
    """
    Small demonstration that builds a surrogate from synthetic data,
    creates a circuit‑breaker, and runs a sequence of hybrid events.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Synthetic training data (3‑D points)
    train_pts = [np.random.rand(3) for _ in range(15)]
    train_vals = [math.sin(p.sum()) for p in train_pts]

    # Fixed morphology for the whole simulation
    morph = Morphology(length=1.2, width=0.8, height=0.5)

    # Fit surrogate in the joint space (original 3‑D + S,F)
    surrogate = fit_hybrid_surrogate(train_pts, train_vals, morph, epsilon=1.5)

    # Initialise circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=3)

    # Run events
    for i in range(num_events):
        event_str = f"event_{i}_{random.choice(['alpha', 'beta', 'gamma'])}"
        work = random.uniform(0.5, 2.0)
        drag = random.uniform(0.1, 1.0)
        urgency = random.uniform(0.0, 1.0)
        input_vec = np.random.rand(3)
        hybrid_event_process(
            breaker,
            surrogate,
            morph,
            event_str,
            work,
            drag,
            urgency,
            input_vec,
        )


if __name__ == "__main__":
    # Smoke test – runs without raising exceptions
    simulate_hybrid_workflow(num_events=5)