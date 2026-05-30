# DARWIN HAMMER — match 3496, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_pheromone_hyb_chelydrid_ambush_m183_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_jepa_e_m705_s0.py (gen5)
# born: 2026-05-29T23:50:28Z

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass(frozen=True)
class StrikeState:
    """Physical state of a strike after integrating a force series."""
    velocity: float
    distance: float
    peak_velocity: float


def compute_phash(values: List[float]) -> int:
    """
    Produce a deterministic integer fingerprint from a list of floats.
    The fingerprint is based on the sign of each value relative to the median.
    All values are used (no arbitrary 64‑value cut‑off) and the result fits
    in a Python int (unlimited precision).
    """
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer fingerprints."""
    return (a ^ b).bit_count()


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1.225,
) -> StrikeState:
    """
    Simple physics integration of a one‑dimensional strike.
    Uses Newton's second law with a quadratic drag term.
    """
    velocity = 0.0
    distance = 0.0
    peak_velocity = 0.0

    for f in force_series:
        # Drag force: 0.5 * rho * Cd * A * v^2 ; we absorb constants into drag_cd
        drag = drag_cd * velocity * abs(velocity)
        acceleration = (f - drag) / m_head
        velocity += acceleration * dt
        distance += velocity * dt
        peak_velocity = max(peak_velocity, abs(velocity))

    return StrikeState(velocity, distance, peak_velocity)


def burst_admission_score(
    values: List[float],
    work_value: float,
    cost_drag: float,
    urgency_force: float,
) -> float:
    """
    Score reflecting the “burst” quality of a signal.
    The exponential term rewards values that exceed the work threshold,
    while a penalty proportional to drag and urgency is subtracted.
    """
    if work_value == 0:
        raise ValueError("work_value must be non‑zero")
    phash = compute_phash(values)
    score = sum(v * (1 - math.exp(-v / work_value)) for v in values)
    # urgency is scaled by the bit‑density of the fingerprint (popcount)
    urgency_penalty = urgency_force * (phash.bit_count() / max(1, len(values)))
    score -= cost_drag + urgency_penalty
    return score


def nlms_update(
    weights: np.ndarray,
    input_signal: np.ndarray,
    error: float,
    mu: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalized Least‑Mean‑Squares weight update.
    w_{n+1} = w_n + mu * e_n * x_n / (||x_n||^2 + eps)
    """
    norm_sq = np.dot(input_signal, input_signal) + eps
    return weights + (mu * error / norm_sq) * input_signal


class HybridChelydridAmbushJepaENLMS:
    """
    Encapsulates the hybrid algorithm, preserving internal weight state
    across successive calls and exposing a single `process` method.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        dt: float = 0.01,
        m_head: float = 1.0,
        drag_cd: float = 0.3,
        fluid_density: float = 1.225,
    ):
        self.mu = learning_rate
        self.dt = dt
        self.m_head = m_head
        self.drag_cd = drag_cd
        self.fluid_density = fluid_density
        self.weights = np.zeros(3)  # [w_vel, w_dist, w_peak]

    def process(
        self,
        values: List[float],
        work_value: float,
        cost_drag: float,
        urgency_force: float,
    ) -> np.ndarray:
        """
        Execute one hybrid step:
        1. Compute a burst admission score.
        2. If the score is positive, integrate the physical strike.
        3. Use the resulting state as the NLMS input and update weights.
        4. Return the current weight vector.
        """
        # 1️⃣ Burst admission
        score = burst_admission_score(values, work_value, cost_drag, urgency_force)

        if score <= 0:
            # No adaptation – keep weights unchanged
            return self.weights.copy()

        # 2️⃣ Physical integration
        strike = integrate_strike(
            force_series=values,
            dt=self.dt,
            m_head=self.m_head,
            drag_cd=self.drag_cd,
            fluid_density=self.fluid_density,
        )

        # 3️⃣ NLMS adaptation
        x = np.array([strike.velocity, strike.distance, strike.peak_velocity])
        # Predicted score from current weights (linear model)
        pred = np.dot(self.weights, x)
        error = score - pred
        self.weights = nlms_update(self.weights, x, error, self.mu)

        # 4️⃣ Return a copy to avoid external mutation
        return self.weights.copy()


if __name__ == "__main__":
    # Simple demonstration
    demo_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    algo = HybridChelydridAmbushJepaENLMS(learning_rate=0.05)

    w = algo.process(
        values=demo_values,
        work_value=1.0,
        cost_drag=0.5,
        urgency_force=1.0,
    )
    print("Updated weights:", w)