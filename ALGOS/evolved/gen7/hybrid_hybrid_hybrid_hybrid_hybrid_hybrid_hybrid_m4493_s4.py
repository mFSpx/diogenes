# DARWIN HAMMER — match 4493, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2439_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s1.py (gen5)
# born: 2026-05-29T23:56:08Z

"""Hybrid Voronoi‑Physarum‑RBF Algorithm
Parents:
    • hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c (Voronoi partition + liquid‑time‑constant ODE)
    • hybrid_hybrid_hybrid_distri_rbf_surrogate (Physarum‑based simulated annealing leader election + RBF surrogate)

Mathematical Bridge
-------------------
Each Voronoi cell is treated as a node of a Physarum‑inspired transport network.
The hidden state `h_i` of cell *i* evolves with a liquid‑time‑constant ODE  
    dh_i/dt = -h_i/τ_i + B_i ,   τ_i = f(dist(seed_i, point))  
where `B_i` is a hyper‑dimensional binding of the cell’s points to its seed.
The norm of `h_i` defines the conductance `g_i` of the edge incident to the
global “source”.  An RBF surrogate predicts the hidden state from the seed
coordinates; its prediction error `ε_i` supplies the uncertainty term used in
the annealing temperature

    T_i = cooling_temperature(k, t0 * bp * g_i * (p_a + p_b) * (1+ε_i), α)

with `bp` the broadcast probability from the original leader‑election scheme.
The temperature modulates the Metropolis acceptance rule that selects a
subset of Voronoi cells as leaders.  Thus the Voronoi‑ODE dynamics, the
Physarum conductance, and the RBF‑surrogate uncertainty are tightly coupled
through the shared τ_i, g_i and T_i, yielding a single unified hybrid system.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
RegionIdx = int

# ----------------------------------------------------------------------
# Voronoi utilities (Parent A)
# ----------------------------------------------------------------------
def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))

def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[RegionIdx, List[Point]]:
    """Assign each point to its nearest seed, returning a region dictionary."""
    regions: Dict[RegionIdx, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions

# ----------------------------------------------------------------------
# Hyper‑dimensional binding (simplified)
# ----------------------------------------------------------------------
def bind_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Element‑wise multiplication as a binding operation."""
    return v1 * v2

def point_to_vector(p: Point, dim: int = 128) -> np.ndarray:
    """Encode a 2‑D point into a high‑dimensional binary‑like vector."""
    rng = np.random.default_rng(seed=int(p[0] * 1e6 + p[1] * 1e3) & 0xffffffff)
    return rng.random(dim)

# ----------------------------------------------------------------------
# Liquid‑time‑constant ODE (Parent A)
# ----------------------------------------------------------------------
def liquid_time_update(
    region_points: List[Point],
    seed: Point,
    h_prev: np.ndarray,
    dt: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """
    Perform one Euler step of the ODE:
        dh/dt = -h/τ + B
    τ is proportional to the average distance of points to the seed.
    B is the bound vector of all points in the region.
    Returns the updated hidden state and the computed τ.
    """
    if not region_points:
        # Empty region → no dynamics
        return h_prev, 1.0

    # τ proportional to mean distance (avoid zero)
    distances = [euclidean(p, seed) for p in region_points]
    tau = max(1e-3, np.mean(distances))

    # Bind each point vector to the seed vector and average
    seed_vec = point_to_vector(seed, dim=h_prev.shape[0])
    bound_sum = np.zeros_like(h_prev)
    for p in region_points:
        bound_sum += bind_vectors(point_to_vector(p, dim=h_prev.shape[0]), seed_vec)
    B = bound_sum / len(region_points)

    # Euler integration
    dh = (-h_prev / tau) + B
    h_new = h_prev + dt * dh
    return h_new, tau

# ----------------------------------------------------------------------
# Physarum conductance (derived from hidden state)
# ----------------------------------------------------------------------
def conductance_from_state(h: np.ndarray, eps: float = 1e-6) -> float:
    """Conductance is the L2 norm of the hidden state (positive)."""
    return max(eps, float(np.linalg.norm(h)))

# ----------------------------------------------------------------------
# Radial‑Basis Function surrogate (Parent B)
# ----------------------------------------------------------------------
class RBFSurrogate:
    """Very light RBF surrogate: f(x) = Σ w_i * exp(-γ||x‑c_i||²)"""
    def __init__(self, centers: List[Point], gamma: float = 0.5):
        self.centers = np.array(centers)          # shape (m,2)
        self.gamma = gamma
        # Random weights for demonstration
        rng = np.random.default_rng(42)
        self.weights = rng.standard_normal(len(centers))

    def predict(self, x: Point) -> float:
        diff = self.centers - np.array(x)        # (m,2)
        d2 = np.einsum('ij,ij->i', diff, diff)   # squared distances
        phi = np.exp(-self.gamma * d2)           # (m,)
        return float(np.dot(self.weights, phi))

    def error(self, x: Point, true_val: np.ndarray) -> float:
        """L2 error between surrogate scalar prediction and norm of true hidden state."""
        pred = self.predict(x)
        true_norm = float(np.linalg.norm(true_val))
        return abs(pred - true_norm)

# ----------------------------------------------------------------------
# Temperature schedule (Parent B)
# ----------------------------------------------------------------------
def broadcast_probability(phases: int, phase: int) -> float:
    if phases < 1 or phase < 1:
        raise ValueError("phases and phase must be positive")
    return min(1.0, 1.0 / (2 ** max(0, phases - phase)))

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

def hybrid_temperature(
    phases: int,
    phase: int,
    conductance: float,
    pressure_a: float,
    pressure_b: float,
    surrogate_uncertainty: float,
    t0: float = 1.0,
    alpha: float = 0.95,
) -> float:
    """
    Combined temperature:
        T = cooling_temperature(phase,
                t0 * bp * conductance * (p_a + p_b) * (1 + ε),
                alpha)
    where ε is the surrogate uncertainty.
    """
    bp = broadcast_probability(phases, phase)
    base = t0 * bp * conductance * (pressure_a + pressure_b) * (1.0 + surrogate_uncertainty)
    return cooling_temperature(phase, base, alpha)

# ----------------------------------------------------------------------
# Leader election using Metropolis rule (Parent B)
# ----------------------------------------------------------------------
def metropolis_accept(delta_e: int, temperature: float) -> bool:
    """Accept with probability exp(-ΔE / T)."""
    if delta_e <= 0:
        return True
    prob = math.exp(-delta_e / max(temperature, 1e-12))
    return random.random() < prob

def leader_election(
    region_indices: List[RegionIdx],
    temperatures: Dict[RegionIdx, float],
    conflicts: Dict[RegionIdx, List[RegionIdx]],
    max_leaders: int = 3,
) -> List[RegionIdx]:
    """
    Simulated‑annealing leader election.
    - `conflicts[i]` lists regions that would cause a conflict if both were leaders.
    - Starts with an empty leader set and proposes random candidates.
    - Acceptance follows the Metropolis rule using the region‑specific temperature.
    """
    leaders: List[RegionIdx] = []
    attempts = 0
    while len(leaders) < max_leaders and attempts < 10 * max_leaders:
        cand = random.choice(region_indices)
        # Energy = number of conflicts with already selected leaders
        delta_e = sum(1 for l in leaders if cand in conflicts.get(l, []))
        if metropolis_accept(delta_e, temperatures[cand]):
            if cand not in leaders:
                leaders.append(cand)
        attempts += 1
    return leaders

# ----------------------------------------------------------------------
# Core hybrid step (demonstrates integration of both parents)
# ----------------------------------------------------------------------
def hybrid_step(
    points: List[Point],
    seeds: List[Point],
    prev_states: Dict[RegionIdx, np.ndarray],
    surrogate: RBFSurrogate,
    phase: int,
    phases: int,
    pressure_a: float = 1.0,
    pressure_b: float = 1.0,
) -> Tuple[Dict[RegionIdx, np.ndarray], List[RegionIdx]]:
    """
    One iteration of the hybrid algorithm:
        1. Voronoi assignment.
        2. Liquid‑time‑constant ODE update per region.
        3. Conductance extraction.
        4. Surrogate uncertainty evaluation.
        5. Temperature computation.
        6. Leader election.
    Returns updated hidden states and the elected leaders.
    """
    # 1. Voronoi partition
    regions = assign_voronoi(points, seeds)

    # Containers for the next iteration
    new_states: Dict[RegionIdx, np.ndarray] = {}
    temperatures: Dict[RegionIdx, float] = {}
    # Simple conflict graph: neighboring seeds (Euclidean distance below a threshold)
    conflict_graph: Dict[RegionIdx, List[RegionIdx]] = defaultdict(list)
    neighbor_thresh = 0.3 * max(
        max(euclidean(s1, s2) for s1 in seeds for s2 in seeds), 1.0
    )
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            if i != j and euclidean(si, sj) < neighbor_thresh:
                conflict_graph[i].append(j)

    # 2‑5. Per‑region dynamics
    for idx, region_pts in regions.items():
        h_prev = prev_states.get(idx, np.zeros(128))
        h_new, tau = liquid_time_update(region_pts, seeds[idx], h_prev)

        # Conductance from hidden state
        g = conductance_from_state(h_new)

        # Surrogate uncertainty (error)
        eps = surrogate.error(seeds[idx], h_new)

        # Temperature specific to this region
        T = hybrid_temperature(
            phases=phases,
            phase=phase,
            conductance=g,
            pressure_a=pressure_a,
            pressure_b=pressure_b,
            surrogate_uncertainty=eps,
        )
        new_states[idx] = h_new
        temperatures[idx] = T

    # 6. Leader election
    leaders = leader_election(
        region_indices=list(regions.keys()),
        temperatures=temperatures,
        conflicts=conflict_graph,
        max_leaders=3,
    )
    return new_states, leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic data
    rng = np.random.default_rng(1234)
    num_seeds = 5
    seeds = [tuple(rng.random(2)) for _ in range(num_seeds)]
    points = [tuple(rng.random(2)) for _ in range(200)]

    # Initial hidden states (zero vectors)
    hidden_states: Dict[int, np.ndarray] = {i: np.zeros(128) for i in range(num_seeds)}

    # Instantiate surrogate with the same seeds as centers
    surrogate = RBFSurrogate(centers=seeds, gamma=10.0)

    # Run a few hybrid steps
    for phase in range(1, 6):
        hidden_states, leaders = hybrid_step(
            points=points,
            seeds=seeds,
            prev_states=hidden_states,
            surrogate=surrogate,
            phase=phase,
            phases=5,
            pressure_a=1.0,
            pressure_b=0.8,
        )
        print(f"Phase {phase}: elected leaders -> {leaders}")

    sys.exit(0)