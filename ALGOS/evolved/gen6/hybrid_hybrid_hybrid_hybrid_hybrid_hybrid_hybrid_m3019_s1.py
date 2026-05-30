# DARWIN HAMMER — match 3019, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_hybrid_m1683_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hard_truth_ma_m2396_s1.py (gen3)
# born: 2026-05-29T23:47:21Z

import numpy as np
from collections.abc import Mapping, Hashable
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import math


# ----------------------------- Core Types ---------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]


@dataclass(frozen=True)
class StrikeState:
    """Result of the physical integration step."""
    final_velocity: float
    total_distance: float
    peak_velocity: float


@dataclass(frozen=True)
class Morphology:
    """Geometric and inertial description of an entity."""
    length: float
    width: float
    height: float
    mass: float

    def __post_init__(self):
        if min(self.length, self.width, self.height, self.mass) <= 0:
            raise ValueError("All morphology dimensions and mass must be positive")


# --------------------------- Helper Functions -----------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Return a dimensionless sphericity (0,1] where 1 is a perfect sphere."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    # Equivalent sphere radius / max dimension
    return (length * width * height) ** (1 / 3) / max(length, width, height)


def compute_dhash(values: List[float]) -> int:
    """Directional hash – 1‑bit per adjacent pair, 1 if descending."""
    if len(values) < 2:
        return 0
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Positional hash – 1‑bit per element, 1 if >= mean (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


# -------------------------- Circuit Breaker -----------------------------
class EndpointCircuitBreaker:
    """
    Tracks endpoint health and provides a *cognitive‑risk* score.
    The cognitive‑risk score blends failure frequency with a user‑defined
    recovery priority (higher priority → lower perceived risk).
    """

    def __init__(self, failure_threshold: int = 3, recovery_priority: float = 1.0):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        if not (0 < recovery_priority):
            raise ValueError("recovery_priority must be positive")
        self.failure_threshold: int = failure_threshold
        self.recovery_priority: float = recovery_priority
        self.failures: int = 0
        self.open: bool = False
        self.last_event_at: Optional[float] = None

    # -----------------------------------------------------------------
    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = time.time()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = time.time()

    def allow(self) -> bool:
        """Endpoint is usable only when the breaker is closed."""
        return not self.open

    # -----------------------------------------------------------------
    @property
    def health_score(self) -> float:
        """Linear health in [0,1] based on failures vs threshold."""
        return max(0.0, 1.0 - self.failures / self.failure_threshold)

    @property
    def cognitive_risk(self) -> float:
        """
        A risk metric that grows with failures but is mitigated
        by the recovery priority. Normalised to [0,1].
        """
        raw = self.failures / self.failure_threshold
        mitigated = raw / (self.recovery_priority + raw)
        return min(1.0, mitigated)


# -------------------------- Physical Integration -------------------------
def integrate_strike(
    force_series: List[float],
    dt: float,
    morphology: Morphology,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area_factor: float = 1.0,
) -> StrikeState:
    """
    Numerically integrates a 1‑D strike using Euler's method.
    The drag term is scaled by the sphericity index to reflect shape.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if not force_series:
        raise ValueError("force_series cannot be empty")

    # Shape‑dependent cross‑sectional area (baseline * sphericity)
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    area = area_factor * sphericity  # dimensionless scaling

    velocity = 0.0
    distance = 0.0
    peak_velocity = 0.0

    for force in force_series:
        # Quadratic drag: ½·C_d·ρ·A·v²
        drag = 0.5 * drag_cd * fluid_density * area * velocity ** 2
        acceleration = (force - drag) / morphology.mass
        velocity += acceleration * dt
        distance += velocity * dt
        if velocity > peak_velocity:
            peak_velocity = velocity

    return StrikeState(final_velocity=velocity, total_distance=distance, peak_velocity=peak_velocity)


# -------------------------- Fusion Logic --------------------------------
def burst_admission_score(
    breaker: EndpointCircuitBreaker,
    morphology: Morphology,
) -> float:
    """
    Combines health, sphericity, and cognitive‑risk into a single
    admission metric. The risk term *reduces* the score, while sphericity
    and health *increase* it.
    """
    health = breaker.health_score
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)
    risk_penalty = 1.0 - breaker.cognitive_risk  # higher risk → lower penalty

    # Geometric weighting (α) and risk weighting (β) can be tuned.
    α = 0.6
    β = 0.4
    return (α * health * sphericity) + (β * risk_penalty)


def optimize_model_loading(
    breaker: EndpointCircuitBreaker,
    morphology: Morphology,
    force_series: List[float],
    dt: float = 0.01,
) -> Tuple[StrikeState, float]:
    """
    Executes the fused workflow:
      1. Compute burst admission score (fusion of both parent algorithms).
      2. Modulate the physical integration by the admission score – a higher
         score yields a finer time step (more accurate) and a scaling factor
         on the applied forces.
      3. Return both the physical state and the admission metric.
    """
    admission = burst_admission_score(breaker, morphology)

    # Adaptive scaling: stronger admission → more trust in the force series.
    force_scale = 0.5 + admission  # maps [0,1] → [0.5,1.5]
    scaled_forces = [f * force_scale for f in force_series]

    # Adaptive dt: higher admission → smaller dt (more precision)
    adaptive_dt = max(1e-4, dt * (1.0 - admission * 0.5))

    strike_state = integrate_strike(
        force_series=scaled_forces,
        dt=adaptive_dt,
        morphology=morphology,
    )
    return strike_state, admission


# -------------------------- Example Execution ---------------------------
if __name__ == "__main__":
    # Initialise components with realistic parameters
    breaker = EndpointCircuitBreaker(failure_threshold=5, recovery_priority=2.0)
    # Simulate a few failures to showcase risk handling
    for _ in range(2):
        breaker.record_failure()

    morphology = Morphology(length=1.2, width=0.8, height=0.6, mass=3.5)

    # Synthetic force profile (N)
    force_series = np.linspace(0, 10, 200).tolist()

    strike, admission = optimize_model_loading(breaker, morphology, force_series)

    print(f"Admission score : {admission:.4f}")
    print(f"Final velocity  : {strike.final_velocity:.4f} m/s")
    print(f"Peak velocity   : {strike.peak_velocity:.4f} m/s")
    print(f"Total distance   : {strike.total_distance:.4f} m")