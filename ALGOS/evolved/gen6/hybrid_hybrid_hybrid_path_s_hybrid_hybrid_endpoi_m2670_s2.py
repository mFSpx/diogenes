# DARWIN HAMMER — match 2670, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s1.py (gen5)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (gen4)
# born: 2026-05-29T23:43:31Z

"""Hybrid algorithm merging path‑signature mathematics (Parent A) with an
adaptive Endpoint Circuit Breaker (Parent B).

Mathematical bridge
-------------------
* From Parent A we obtain a *second‑order signature matrix*  
  `S₂ = Σ_{t}(X_{t} - X₀) ⊗ (X_{t+1} - X_{t})` via `signature_level2`.
  Its Frobenius norm `‖S₂‖_F` quantifies the geometric complexity of a
  trajectory.

* From Parent B the circuit breaker opens after a configurable number of
  failures.  We let the *failure threshold* be *data‑driven*:
  `τ = τ₀ + ⌊α·‖S₂‖_F⌋`, where `τ₀` is a base threshold and `α` is a
  scaling factor derived from a Fisher‑like score `F = var(ΔX)/mean(ΔX)` of
  the path increments.

Thus the signature matrix supplies a scalar that directly adjusts the
circuit‑breaker’s governing equation, producing a single unified system."""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – path‑signature utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

def lead_lag_transform(path):
    """Lead‑lag augmentation used for signatures."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def signature_level1(path):
    """First‑order signature (increment from start to end)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]

def signature_level2(path):
    """Second‑order signature matrix Σ (X_{t}‑X₀) ⊗ (X_{t+1}‑X_{t})."""
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)          # (T‑1, d)
    running    = path[:-1] - path[0]            # (T‑1, d)
    return running.T @ increments               # (d, d)

def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """Cox‑de Boor B‑spline basis (order k)."""
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = ((x - t[i]) / denom_l * B[:, i]) if denom_l > 0 else np.zeros(N)
            term_r = ((t[i + order] - x) / denom_r * B[:, i + 1]) if denom_r > 0 else np.zeros(N)
            B_new[:, i] = term_l + term_r
        B = B_new
    return B

# ----------------------------------------------------------------------
# Parent B – adaptive circuit breaker & morphology
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable (adaptive) threshold."""
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

    def as_dict(self) -> dict:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

# ----------------------------------------------------------------------
# Hybrid utilities – mathematical bridge
# ----------------------------------------------------------------------
def fisher_score(path: np.ndarray) -> float:
    """A Fisher‑like score: variance of increments divided by mean magnitude."""
    inc = np.diff(path, axis=0)
    var = np.var(inc, axis=0).mean()
    mean = np.mean(np.linalg.norm(inc, axis=1))
    return var / mean if mean != 0 else 0.0

def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Very lightweight SSIM for 1‑D signals (range 0‑1)."""
    C1 = (0.01 * np.max([x.max(), y.max()])) ** 2
    C2 = (0.03 * np.max([x.max(), y.max()])) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator if denominator != 0 else 0.0

def adaptive_threshold(path: np.ndarray, base: int = 3, alpha: float = 0.5) -> int:
    """
    Compute a data‑driven failure threshold:
        τ = base + floor( α · ‖S₂‖_F · fisher_score )
    """
    S2 = signature_level2(path)
    frob = np.linalg.norm(S2, ord='fro')
    fisher = fisher_score(path)
    τ = base + int(math.floor(alpha * frob * fisher))
    return max(1, τ)   # threshold must be at least 1

def morphology_to_path(morph: Morphology, steps: int = 20) -> np.ndarray:
    """
    Encode morphology as a smooth path in ℝ⁴ using a B‑spline basis.
    The four dimensions are (length, width, height, mass).
    """
    t = np.linspace(0, 1, steps)
    grid = np.linspace(0, 1, 5)          # modest knot vector
    B = bspline_basis(t, grid, k=3)      # (steps, n_basis)
    coeffs = np.vstack([
        np.full(B.shape[1], morph.length),
        np.full(B.shape[1], morph.width),
        np.full(B.shape[1], morph.height),
        np.full(B.shape[1], morph.mass),
    ]).T                                 # (n_basis, 4)
    path = B @ coeffs                    # (steps, 4)
    # Add a small random walk to avoid a perfectly flat trajectory
    noise = np.cumsum(np.random.randn(*path.shape) * 0.01, axis=0)
    return path + noise

def hybrid_circuit_breaker(morph: Morphology,
                           base_threshold: int = 3,
                           alpha: float = 0.5) -> EndpointCircuitBreaker:
    """
    Build an EndpointCircuitBreaker whose failure threshold is tuned
    by the signature of the morphology‑derived path.
    """
    path = morphology_to_path(morph)
    τ = adaptive_threshold(path, base=base_threshold, alpha=alpha)
    return EndpointCircuitBreaker(failure_threshold=τ)

def evaluate_hybrid(morph: Morphology,
                    cb: EndpointCircuitBreaker,
                    trials: int = 10) -> dict:
    """
    Simulate a series of events.  An event is considered a *failure* if
    the SSIM between the current morphology path segment and the full
    path falls below a random tolerance; otherwise it is a *success*.
    Returns a summary dictionary.
    """
    full_path = morphology_to_path(morph)
    successes = 0
    failures = 0
    for _ in range(trials):
        # Random sub‑segment
        start = random.randint(0, len(full_path) - 5)
        seg = full_path[start:start + 5]
        similarity = ssim(full_path.mean(axis=0), seg.mean(axis=0))
        tolerance = random.uniform(0.4, 0.9)
        if similarity >= tolerance:
            cb.record_success()
            successes += 1
        else:
            cb.record_failure()
            failures += 1
        # Early exit if breaker opened
        if not cb.allow():
            break
    return {
        "final_state": cb.as_dict(),
        "successes": successes,
        "failures": failures,
        "breaker_open": not cb.allow(),
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a random morphology
    morph = Morphology(
        length=random.uniform(1.0, 10.0),
        width=random.uniform(0.5, 5.0),
        height=random.uniform(0.2, 3.0),
        mass=random.uniform(10.0, 100.0),
    )
    # Build hybrid breaker
    breaker = hybrid_circuit_breaker(morph, base_threshold=3, alpha=0.7)
    # Run evaluation
    result = evaluate_hybrid(morph, breaker, trials=15)

    print("Morphology:", morph)
    print("Adaptive failure threshold:", breaker.failure_threshold)
    print("Result summary:", result)