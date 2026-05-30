# DARWIN HAMMER — match 142, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s1.py (gen2)
# parent_b: hybrid_nlms_omni_chaotic_sprint_m59_s0.py (gen1)
# born: 2026-05-29T23:25:58Z

"""Hybrid Endpoint-NLMS Workshare Engine
Parents:
- hybrid_hybrid_endpoint_circ_hybrid_workshare_all_m12_s1.py (Endpoint circuit‑breaker + morphology health model)
- hybrid_nlms_omni_chaotic_sprint_m59_s0.py (Normalized Least Mean Squares adaptive filter)

Mathematical Bridge:
Each endpoint possesses a health score H ∈ [0,1] derived from its circuit‑breaker state,
morphology (sphericity S and flatness F) and a day‑of‑week modulation D(t).  
The NLMS weight update Δw = μ·e·x / (‖x‖²+ε) is scaled by the composite factor μ·H·D(t),
thereby fusing the adaptive filtering dynamics of NLMS with the morphology‑driven
priority logic of the endpoint work‑share algorithm. The resulting system simultaneously
learns optimal graph weights while allocating work proportionally to endpoint health."""

import sys
import math
import random
from pathlib import Path
from datetime import date
import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Return a cyclic day index in [0,6] (0 = Monday, 6 = Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7

class EndpointCircuitBreaker:
    """Simple circuit‑breaker tracking consecutive failures."""
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
        """True if the circuit is closed (i.e. endpoint may receive work)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0,1]."""
        return min(self.failures / self.failure_threshold, 1.0)

class Morphology:
    """Geometric description of an endpoint."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All dimensions must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of geometric mean to maximal dimension, ∈ (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    gm = (length * width * height) ** (1.0 / 3.0)
    return gm / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Relative flatness, larger for plate‑like shapes."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

# ----------------------------------------------------------------------
# Parent B building blocks (NLMS core)
# ----------------------------------------------------------------------
class NLMSFilter:
    """Normalized Least Mean Squares adaptive filter."""
    def __init__(self, dim: int, mu: float = 0.5, eps: float = 1e-9):
        self.weights = np.random.rand(dim)
        self.mu = mu
        self.eps = eps

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, x))

    def update(self, x: np.ndarray, target: float, mu_factor: float = 1.0) -> float:
        """
        Perform a single NLMS weight update.
        mu_factor scales the base step size (μ) – this is where the health
        score from Parent A enters the mathematics.
        """
        y = self.predict(x)
        error = target - y
        power = float(np.dot(x, x) + self.eps)
        step = self.mu * mu_factor * error * x / power
        self.weights += step
        return error

# ----------------------------------------------------------------------
# Hybrid constructs
# ----------------------------------------------------------------------
def compute_endpoint_health(cb: EndpointCircuitBreaker, morph: Morphology) -> float:
    """
    Composite health H ∈ [0,1] = (1‑failure_rate)·S·F·C,
    where S = sphericity, F = flatness, C = circuit‑breaker allowance (0/1).
    """
    if not cb.allow():
        return 0.0
    s = sphericity_index(morph.length, morph.width, morph.height)
    f = flatness_index(morph.length, morph.width, morph.height)
    reliability = 1.0 - cb.failure_rate()
    health = reliability * s * f
    return max(0.0, min(health, 1.0))

def day_modulation_factor(year: int, month: int, day: int) -> float:
    """
    Day‑of‑week factor D(t) ∈ (0,1] that cycles weekly.
    D = (doomsday + 1) / 7  → Monday ≈0.29, Sunday =1.0.
    """
    idx = doomsday(year, month, day)
    return (idx + 1) / 7.0

def allocate_workshare(endpoints: dict, year: int, month: int, day: int) -> dict:
    """
    Distribute a unit workload across endpoints proportionally to their
    health·day_factor product.
    `endpoints` maps an identifier to a tuple (EndpointCircuitBreaker, Morphology).
    Returns a dict identifier → allocated share (sums to 1.0).
    """
    day_factor = day_modulation_factor(year, month, day)
    scores = {}
    total = 0.0
    for eid, (cb, morph) in endpoints.items():
        h = compute_endpoint_health(cb, morph)
        score = h * day_factor
        scores[eid] = score
        total += score
    if total == 0.0:
        # fallback: equal split
        n = len(endpoints)
        return {eid: 1.0 / n for eid in endpoints}
    return {eid: score / total for eid, score in scores.items()}

def hybrid_nlms_step(filter_: NLMSFilter,
                     x: np.ndarray,
                     target: float,
                     health: float,
                     day_factor: float) -> float:
    """
    Perform an NLMS update where the effective step size is
    μ_eff = μ * health * day_factor.
    Returns the instantaneous error.
    """
    mu_factor = health * day_factor
    return filter_.update(x, target, mu_factor=mu_factor)

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a small set of synthetic endpoints
    endpoints = {}
    random.seed(42)
    for i in range(5):
        cb = EndpointCircuitBreaker(failure_threshold=3)
        # Randomly inject failures
        for _ in range(random.randint(0, 4)):
            cb.record_failure()
        # Random morphology
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0)
        )
        endpoints[f"ep_{i}"] = (cb, morph)

    # Choose a date for modulation
    today = date.today()
    year, month, day = today.year, today.month, today.day

    # Allocate workshare based on health and day factor
    allocation = allocate_workshare(endpoints, year, month, day)
    print("Workshare allocation:")
    for eid, share in allocation.items():
        print(f"  {eid}: {share:.4f}")

    # Initialise NLMS filter (dimension matches number of endpoints)
    dim = len(endpoints)
    nlms = NLMSFilter(dim=dim, mu=0.3)

    # Build an input vector x where each component is the allocated share
    x = np.array([allocation[eid] for eid in sorted(endpoints.keys())])

    # Define a synthetic target (e.g., desired aggregate performance)
    target = 0.75

    # Compute composite health factor for the whole system (mean health)
    healths = [compute_endpoint_health(cb, morph) for cb, morph in endpoints.values()]
    mean_health = sum(healths) / len(healths)

    day_factor = day_modulation_factor(year, month, day)

    # Perform a hybrid NLMS step
    error = hybrid_nlms_step(nlms, x, target, mean_health, day_factor)
    print(f"\nNLMS update performed. Error after update: {error:.6f}")
    print(f"Updated weights: {nlms.weights}")

    sys.exit(0)