# DARWIN HAMMER — match 183, survivor 4
# gen: 3
# parent_a: hybrid_pheromone_hybrid_distributed_l_m41_s0.py (gen2)
# parent_b: chelydrid_ambush.py (gen0)
# born: 2026-05-29T23:27:23Z

"""Hybrid Pheromone-Strike Algorithm

Parents:
- PARENT ALGORITHM A: pheromone_hybrid_distributed (pheromone signal recording,
  perceptual hashing, broadcast probability)
- PARENT ALGORITHM B: chelydrid_ambush (kinematic integration of a burst force
  with quadratic drag)

Mathematical Bridge:
The pheromone signal values are interpreted as a time‑varying force series.
These forces feed the kinematic integrator from the ambush primitive.
The perceptual hash of the original pheromone vector is then used to modulate
the drag coefficient (higher hash → higher effective drag) and to provide a
compact identifier for leader‑election via Hamming distance between hashes.
Thus the hybrid system couples the discrete signal topology of the pheromone
model with the continuous dynamics of the strike model. """

import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

import numpy as np


def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


def broadcast_probability(phase: int, step: int) -> float:
    """Probability rule taken from the pheromone parent."""
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Kinematic integration of a burst force with quadratic drag.
    Returns (final_velocity, total_distance, peak_velocity).
    """
    if dt <= 0 or m_head <= 0 or drag_cd < 0 or fluid_density < 0 or area < 0:
        raise ValueError("invalid strike parameters")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return v, x, peak


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse approximating a short burst."""
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


def hybrid_strike_from_pheromones(
    pheromone_values: List[float],
    dt: float = 0.01,
    m_head: float = 1.0,
    base_drag_cd: float = 0.3,
) -> Tuple[Tuple[float, float, float], int]:
    """
    Core hybrid operation.

    1. The raw pheromone values are normalised and used as a force series.
    2. A perceptual hash of the original values is computed.
    3. The hash modulates the drag coefficient:
         drag_cd = base_drag_cd * (1 + hash_fraction)
       where hash_fraction ∈ [0,1).
    4. The modified force series is integrated with the ambush kinematics.

    Returns a tuple (strike_state, phash) where strike_state is
    (final_velocity, total_distance, peak_velocity).
    """
    if not pheromone_values:
        raise ValueError("pheromone_values must be non‑empty")
    # Normalise to a reasonable force range (0 … 10)
    arr = np.array(pheromone_values, dtype=float)
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        forces = np.zeros_like(arr)
    else:
        forces = 10.0 * arr / max_abs

    # Compute perceptual hash
    phash = compute_phash(pheromone_values)

    # Drag modulation based on hash
    hash_fraction = (phash % 256) / 256.0  # 0 ≤ fraction < 1
    drag_cd = base_drag_cd * (1.0 + hash_fraction)

    # Integrate
    final_v, dist, peak_v = integrate_strike(
        forces,
        dt=dt,
        m_head=m_head,
        drag_cd=drag_cd,
        fluid_density=1000.0,
        area=0.001,
        v0=0.0,
    )
    return (final_v, dist, peak_v), phash


def leader_election(
    node_signals: Dict[str, List[float]],
    phase: int = 1,
    step: int = 1,
) -> Dict[str, str]:
    """
    Cluster nodes by Hamming distance of their perceptual hashes and elect a
    leader per cluster. The election prefers the node with the smallest
    broadcast probability (i.e., the most “central” node in the phase/step
    schedule).

    Returns a mapping {node_id: leader_node_id}.
    """
    # Compute hashes once
    hashes = {nid: compute_phash(vals) for nid, vals in node_signals.items()}

    # Simple clustering: nodes whose hash distance ≤ 2 are in the same cluster
    clusters: List[set] = []
    for nid, h in hashes.items():
        placed = False
        for cluster in clusters:
            rep = next(iter(cluster))
            if hamming_distance(h, hashes[rep]) <= 2:
                cluster.add(nid)
                placed = True
                break
        if not placed:
            clusters.append({nid})

    # Election within each cluster
    leaders: Dict[str, str] = {}
    prob = broadcast_probability(phase, step)
    for cluster in clusters:
        # Choose the node with minimal hash value as deterministic leader
        leader = min(cluster, key=lambda n: hashes[n])
        for member in cluster:
            leaders[member] = leader
    return leaders


if __name__ == "__main__":
    # Smoke test for hybrid_strike_from_pheromones
    sample_pheromone = [random.uniform(-5, 5) for _ in range(20)]
    strike_state, phash = hybrid_strike_from_pheromones(sample_pheromone)
    print("Hybrid strike state (v, distance, peak):", strike_state)
    print("Perceptual hash:", phash)

    # Smoke test for leader_election
    node_data = {
        f"node_{i}": [random.uniform(-3, 3) for _ in range(10)]
        for i in range(5)
    }
    election = leader_election(node_data, phase=2, step=1)
    print("Leader election mapping:", election)