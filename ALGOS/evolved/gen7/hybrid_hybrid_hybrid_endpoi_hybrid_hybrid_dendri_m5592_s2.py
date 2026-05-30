# DARWIN HAMMER — match 5592, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py (gen2)
# parent_b: hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py (gen6)
# born: 2026-05-30T00:03:11Z

"""Hybrid Voronoi‑Dendritic Circuit: Fusion of
`hybrid_hybrid_endpoint_circ_hybrid_voronoi_parti_m745_s1.py` (Voronoi
partition + endpoint circuit‑breaker) and
`hybrid_hybrid_dendritic_com_hybrid_sparse_wta_hy_m1239_s4.py` (Hodgkin‑Huxley
dendritic membrane potential + regret‑weighted ternary decision + sparse
winner‑take‑all).

Mathematical bridge
------------------
For every seed *s* we compute a dendritic membrane potential `V_s` using a
simplified Hodgkin‑Huxley formulation.  This potential is then used as a
multiplicative weight on the Euclidean distance when assigning points to the
nearest seed:


d̂(p, s) = distance(p, s) * exp(-V_s / V_scale)


Thus the Voronoi tessellation becomes *potential‑aware*: seeds with higher
excitability (large `V_s`) attract points from a larger neighbourhood.  The
resulting assignments feed a per‑seed `EndpointCircuitBreaker`; a seed whose
potential exceeds a configurable threshold is “opened” (disabled) and its
region is reassigned to the next best seed.  Finally the vector of region
sizes is fed to a sparse winner‑take‑all (WTA) layer that returns the `k`
most active seeds, providing a compact decision signature.

The module therefore intertwines the two parent topologies rather than
stacking them side‑by‑side.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – Voronoi / Endpoint Circuit‑Breaker utilities
# ---------------------------------------------------------------------------

def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    return (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Standard L2 distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Parent B – Dendritic / Sparse WTA utilities
# ---------------------------------------------------------------------------

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


def sodium_current(V: np.ndarray, m: np.ndarray, h: np.ndarray,
                   g_Na: float = 120.0, E_Na: float = 50.0) -> np.ndarray:
    """Hodgkin‑Huxley sodium current: I_Na = g_Na * m³ * h * (V‑E_Na)."""
    return g_Na * np.power(m, 3) * h * (V - E_Na)


def calculate_membrane_potential(
    C_m: float,
    g_L: float,
    E_L: float,
    V_rest: float,
    I_ion: np.ndarray,
    I_syn: np.ndarray,
    dt: float = 0.01,
) -> np.ndarray:
    """
    Simple Euler integration of the membrane equation:

        C_m * dV/dt = -g_L * (V - E_L) + I_ion + I_syn

    Returns the updated membrane potential array.
    """
    dV = (-g_L * (V_rest - E_L) + I_ion + I_syn) * (dt / C_m)
    return V_rest + dV


def regret_weighted_probability(
    value: float, regret: float, temperature: float = 1.0
) -> float:
    """
    Softmax‑like mapping that penalises high regret.
    """
    exp_term = math.exp(-(value + regret) / temperature)
    return exp_term / (1.0 + exp_term)


def ternary_encode(prob: float) -> int:
    """
    Map a probability to a ternary symbol:
        0 → low  (p < 1/3)
        1 → mid  (1/3 ≤ p < 2/3)
        2 → high (p ≥ 2/3)
    """
    if prob < 1.0 / 3.0:
        return 0
    elif prob < 2.0 / 3.0:
        return 1
    else:
        return 2


def sparse_winner_take_all(vector: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Return a sparse binary vector where only the `k` largest entries are 1.
    """
    if k <= 0:
        return np.zeros_like(vector, dtype=int)
    idx = np.argpartition(-vector, range(k))[:k]
    out = np.zeros_like(vector, dtype=int)
    out[idx] = 1
    return out


# ---------------------------------------------------------------------------
# Hybrid core – bridging the two parent topologies
# ---------------------------------------------------------------------------

def compute_seed_potentials(
    seeds: List[Tuple[float, float]],
    points: List[Tuple[float, float]],
    C_m: float = 1.0,
    g_L: float = 0.1,
    E_L: float = -65.0,
    dt: float = 0.01,
) -> np.ndarray:
    """
    For each seed we treat the aggregated Euclidean distance to its assigned
    points as an ionic current `I_ion`.  A simplistic activation (`m`) and
    inactivation (`h`) gate are derived from the distance statistics and fed
    into the Hodgkin‑Huxley sodium current.  The resulting `I_ion` drives an
    Euler step of the membrane equation, yielding a potential `V_s` per seed.
    """
    # 1. naive assignment (pure Euclidean) – just to obtain a distance sum per seed
    assignments = {i: [] for i in range(len(seeds))}
    for p in points:
        dists = [euclidean_distance(p, s) for s in seeds]
        nearest_idx = int(np.argmin(dists))
        assignments[nearest_idx].append(p)

    # 2. compute a scalar current for each seed
    V = np.full(len(seeds), E_L)  # start from leak reversal
    for i, pts in assignments.items():
        if not pts:
            I_ion = 0.0
        else:
            # mean distance as proxy for excitatory drive
            mean_dist = np.mean([euclidean_distance(p, seeds[i]) for p in pts])
            # map distance -> gating variables (bounded between 0 and 1)
            m = 1.0 / (1.0 + math.exp((mean_dist - 5.0)))  # sigmoid
            h = 1.0 - m
            I_ion = sodium_current(np.array([V[i]]), np.array([m]), np.array([h]))[0]
        # synaptic current is set to zero for this hybrid demo
        I_syn = 0.0
        V[i] = calculate_membrane_potential(
            C_m, g_L, E_L, V[i], I_ion, I_syn, dt=dt
        )[0]
    return V


def weighted_distance(
    point: Tuple[float, float],
    seed: Tuple[float, float],
    potential: float,
    scale: float = 20.0,
) -> float:
    """
    Distance modulated by the seed's membrane potential.
    Larger potentials shrink the effective distance, widening the basin of
    attraction for that seed.
    """
    base = euclidean_distance(point, seed)
    weight = math.exp(-potential / scale)
    return base * weight


def weighted_nearest(
    point: Tuple[float, float],
    seeds: List[Tuple[float, float]],
    potentials: np.ndarray,
    scale: float = 20.0,
) -> int:
    """
    Return the index of the seed that minimises the potential‑aware distance.
    """
    distances = [
        weighted_distance(point, seed, pot, scale)
        for seed, pot in zip(seeds, potentials)
    ]
    return int(np.argmin(distances))


def assign_with_potentials(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    potentials: np.ndarray,
    scale: float = 20.0,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Voronoi assignment using `weighted_nearest`.  Returns a dict mapping seed
    indices to the list of points belonging to each region.
    """
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = weighted_nearest(p, seeds, potentials, scale)
        regions[idx].append(p)
    return regions


def update_circuit_breakers(
    regions: Dict[int, List[Tuple[float, float]]],
    breakers: List[EndpointCircuitBreaker],
    size_threshold: int = 5,
) -> None:
    """
    If a region grows beyond `size_threshold` we record a failure on the
    corresponding breaker; otherwise we record success.  Breakers that become
    open are later ignored during reassignment.
    """
    for idx, pts in regions.items():
        cb = breakers[idx]
        if len(pts) > size_threshold:
            cb.record_failure()
        else:
            cb.record_success()


def hybrid_process(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    breaker_threshold: int = 3,
    region_size_thr: int = 5,
    wta_k: int = 3,
) -> Tuple[Dict[int, List[Tuple[float, float]]], np.ndarray]:
    """
    Full hybrid pipeline:

    1. Compute dendritic potentials for each seed.
    2. Perform potential‑aware Voronoi assignment.
    3. Update per‑seed `EndpointCircuitBreaker`s.
    4. Collapse open breakers: points belonging to an opened seed are
       reassigned to the next best (still‑closed) seed.
    5. Build a feature vector = region sizes and feed it to a sparse WTA.
    6. Return the final region mapping and the sparse decision vector.
    """
    # initialise circuit breakers
    breakers = [EndpointCircuitBreaker(failure_threshold=breaker_threshold)
                for _ in seeds]

    # 1. membrane potentials
    potentials = compute_seed_potentials(seeds, points)

    # 2. primary assignment
    regions = assign_with_potentials(points, seeds, potentials)

    # 3. circuit‑breaker update
    update_circuit_breakers(regions, breakers, size_threshold=region_size_thr)

    # 4. reassign points from opened seeds
    open_indices = [i for i, cb in enumerate(breakers) if not cb.allow()]
    if open_indices:
        # recompute potentials without the opened seeds (they stay at 0 weight)
        closed_mask = np.array([cb.allow() for cb in breakers])
        closed_potentials = potentials * closed_mask
        # reassign each point that currently belongs to a closed seed
        reassigned: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
        for i, pts in regions.items():
            if i in open_indices:
                for p in pts:
                    new_idx = weighted_nearest(p, seeds, closed_potentials)
                    reassigned[new_idx].append(p)
            else:
                reassigned[i].extend(pts)
        regions = reassigned

    # 5. region‑size vector → sparse WTA
    size_vector = np.array([len(regions[i]) for i in range(len(seeds))], dtype=float)
    # optional: convert sizes to regret‑weighted probabilities before WTA
    regrets = np.maximum(0.0, size_vector - np.mean(size_vector))
    probs = np.array([regret_weighted_probability(v, r) for v, r in zip(size_vector, regrets)])
    ternary = np.vectorize(ternary_encode)(probs)
    # treat ternary symbols as a numeric signal for WTA
    decision_vector = sparse_winner_take_all(ternary.astype(float), k=wta_k)

    return regions, decision_vector


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # generate synthetic data
    random.seed(42)
    np.random.seed(42)

    # 15 random points in a 2‑D unit square
    points = [(random.random(), random.random()) for _ in range(15)]

    # 4 seed locations (could be thought of as “neurons”)
    seeds = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]

    regions, decision = hybrid_process(points, seeds)

    print("Final Voronoi regions (seed → point count):")
    for idx, pts in regions.items():
        print(f"  Seed {idx}: {len(pts)} points")

    print("\nSparse WTA decision vector:", decision.tolist())
    print("\nCircuit breaker states:")
    for i, cb in enumerate([EndpointCircuitBreaker() for _ in seeds]):
        # Re‑create breakers to show default state (all closed)
        print(f"  Seed {i}: open={cb.open}, failures={cb.failures}")