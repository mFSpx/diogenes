# DARWIN HAMMER — match 5454, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_capyba_hybrid_regret_engine_m2693_s0.py (gen4)
# born: 2026-05-30T00:02:10Z

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Hashable, Mapping, Iterable, List, Sequence, Tuple, Optional

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class StrikeState:
    """Result of a physics‑based strike simulation."""
    velocity: float
    distance: float
    peak_velocity: float


@dataclass(frozen=True, slots=True)
class MathAction:
    """Elementary decision element used by the hybrid algorithm."""
    id: str
    expected_value: float
    cost: float = 0.0          # monetary / resource cost
    risk: float = 0.0          # probability‑like risk factor


# ----------------------------------------------------------------------
# Helper hash functions (kept from original for potential clustering use)
# ----------------------------------------------------------------------
def compute_dhash(values: List[float]) -> int:
    """Directional hash – 1 if a value is larger than its successor."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    """Population hash – 1 if a value is above the mean (first 64 entries)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Classic Hamming distance for integer bit‑patterns."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Physics core – drag‑augmented strike integration
# ----------------------------------------------------------------------
def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> StrikeState:
    """
    Numerically integrate a 1‑D motion under a time‑varying force series
    while accounting for quadratic drag.

    Parameters
    ----------
    force_series : iterable of float
        External propulsive forces applied at each time step.
    dt : float
        Time step size.
    m_head : float
        Effective mass of the moving head.
    drag_cd, fluid_density, area : float
        Classical drag parameters (coefficient, fluid density, reference area).
    v0 : float
        Initial velocity.

    Returns
    -------
    StrikeState
        Final velocity, travelled distance and peak velocity observed.
    """
    v = v0
    x = 0.0
    peak = v0
    # Pre‑compute constant drag factor for speed
    drag_factor = drag_cd * fluid_density * area / (2.0 * m_head)

    for force in force_series:
        drag = drag_factor * v * abs(v)          # quadratic drag, sign opposite to v
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)               # no backward motion in this model
        x += v * dt
        if v > peak:
            peak = v
    return StrikeState(v, x, peak)


def pulse_force(peak_force: float, steps: int) -> np.ndarray:
    """
    Generate a symmetric triangular pulse of length ``steps`` whose apex
    equals ``peak_force``.
    """
    if steps <= 0:
        return np.array([], dtype=float)
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    idx = np.arange(steps, dtype=float)
    shape = np.maximum(0.0, 1.0 - np.abs(idx - mid) / denom)
    return peak_force * shape


def burst_admission_score(
    work_value: float,
    cost_drag: float,
    urgency_force: float,
    steps: int = 12,
    *,
    dt: float = 1.0,
    m_head: float = 1.0,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
) -> float:
    """
    Compute a performance score for a single action by simulating a
    short “burst” using the physics engine and then weighting the
    resulting velocity by the action's intrinsic value while subtracting
    the explicit drag‑related cost.
    """
    force_series = pulse_force(max(0.0, urgency_force), steps)
    state = integrate_strike(
        force_series,
        dt,
        m_head,
        drag_cd=drag_cd,
        fluid_density=fluid_density,
        area=area,
    )
    return state.velocity * work_value - cost_drag


# ----------------------------------------------------------------------
# Social interaction core – PSO‑like update with deterministic randomness
# ----------------------------------------------------------------------
def social_interaction(
    x: Sequence[float],
    g_best: Sequence[float],
    k: int = 1,
    r1: Optional[float] = None,
    seed: Optional[int | str] = None,
    w: float = 0.5,
) -> np.ndarray:
    """
    A lightweight particle‑swarm style move. ``w`` is an inertia weight,
    ``k`` toggles between attraction (k=1) and repulsion (k=2).
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must have the same dimensionality")
    if k not in (1, 2):
        raise ValueError("k must be 1 (attraction) or 2 (repulsion)")

    rng = random.Random(seed)
    r = rng.random() if r1 is None else float(r1)
    if not (0.0 <= r <= 1.0):
        raise ValueError("r1 must be within [0, 1]")

    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)

    return w * x_arr + r * (g_arr - k * x_arr)


# ----------------------------------------------------------------------
# Regret‑weighted decision core – robust handling of zero‑regret cases
# ----------------------------------------------------------------------
def regret_weighted_strategy(
    actions: Sequence[MathAction],
    regret_coefficient: float = 1.0,
) -> np.ndarray:
    """
    Produce a weight for each action that favours high expected value
    while penalising risk. If the total regret is zero (all risks are
    zero), the function falls back to a simple expected‑value normalisation.
    """
    risks = np.array([a.risk for a in actions], dtype=float)
    ev = np.array([a.expected_value for a in actions], dtype=float)

    total_regret = np.sum(risks * regret_coefficient)
    if total_regret == 0.0:
        # Uniform risk → weight proportional to expected value only
        denom = ev.sum() if ev.sum() != 0 else 1.0
        return ev / denom
    # Classic regret‑adjusted weighting
    denominator = 1.0 + (risks * regret_coefficient) / total_regret
    return ev / denominator


# ----------------------------------------------------------------------
# Deeply integrated hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    actions: Sequence[MathAction],
    x: Sequence[float],
    g_best: Sequence[float],
    regret_coefficient: float = 1.0,
    k: int = 1,
    r1: Optional[float] = None,
    seed: Optional[int | str] = None,
    *,
    dt: float = 1.0,
    steps: int = 12,
    m_head: float = 1.0,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
) -> List[float]:
    """
    Core hybrid routine.

    1. Compute regret‑adjusted weights.
    2. Run a physics‑driven burst for each action, scaling the input
       force by the regret weight (deeper coupling between the two
       subsystems).
    3. Produce a PSO‑style social move.
    4. Blend the physics‑derived burst influence with the social move.

    Returns a list of updated positions (or decision variables).
    """
    # 1. Regret weights (vectorised)
    regret_weights = regret_weighted_strategy(actions, regret_coefficient)

    # 2. Physics‑driven burst scores, now *modulated* by regret weight
    burst_scores = np.empty(len(actions), dtype=float)
    for i, (action, rw) in enumerate(zip(actions, regret_weights)):
        # The urgency force is interpreted as expected_value scaled by regret weight
        urgency = action.expected_value * rw
        burst_scores[i] = burst_admission_score(
            work_value=action.expected_value,
            cost_drag=action.cost,
            urgency_force=urgency,
            steps=steps,
            dt=dt,
            m_head=m_head,
            drag_cd=drag_cd,
            fluid_density=fluid_density,
            area=area,
        ) * rw

    # 3. Social interaction vector
    social_vec = social_interaction(x, g_best, k=k, r1=r1, seed=seed)

    # 4. Blend: move each dimension toward the social target,
    #    but bias the step size by the normalized burst score.
    burst_norm = burst_scores / (np.max(np.abs(burst_scores)) + 1e-12)  # scale to [-1,1]
    result = np.asarray(social_vec, dtype=float) + burst_norm * (np.asarray(g_best, dtype=float) - np.asarray(social_vec, dtype=float))

    return result.tolist()


# ----------------------------------------------------------------------
# Simple demonstration when executed as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_actions = [
        MathAction("action1", expected_value=10.0, cost=2.0, risk=0.5),
        MathAction("action2", expected_value=20.0, cost=3.0, risk=0.7),
        MathAction("action3", expected_value=5.0, cost=1.0, risk=0.0),
    ]
    x0 = [1.0, 2.0, 0.5]
    g_best = [3.0, 4.0, 2.0]

    updated = hybrid_operation(
        demo_actions,
        x0,
        g_best,
        regret_coefficient=1.2,
        k=1,
        seed=42,
    )
    print("Updated state:", updated)