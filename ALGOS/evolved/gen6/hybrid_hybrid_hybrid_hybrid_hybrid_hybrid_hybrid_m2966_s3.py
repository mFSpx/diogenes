# DARWIN HAMMER — match 2966, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_diffus_hybrid_hybrid_worksh_m2528_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2530_s0.py (gen5)
# born: 2026-05-29T23:46:57Z

"""Hybrid Algorithm Fusion of Diffusion Forcing & Ternary Routing

Parents:
- hybrid_hybrid_hybrid_diffus_hybrid_hybrid_worksh_m2528_s0.py (Diffusion Forcing + Weekday Workshare)
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2530_s0.py (Ternary Router + Morphology similarity)

Mathematical Bridge:
Both parents manipulate vectors and weights.  The diffusion parent defines a *cumulative noise
schedule* 𝛼ₜ ∈ [0,1] that perturbs a feature vector at timestep t.  The workshare parent defines a
*weekday weight vector* w ∈ ℝ⁷ that allocates resources across the seven days of a week.
The ternary‑router parent builds Euclidean distances between *morphology feature vectors*
f = (ℓ, w, h, m).  

The hybrid algorithm therefore:
1. Perturbs each morphology vector with the diffusion schedule → f̃ₜ = √αₜ·f.
2. Computes a weighted edge cost between two nodes i and j at a given weekday d:
   
   c₍i,j₎(t,d) = ‖f̃ₜ(i) − f̃ₜ(j)‖₂ · w_d · (1 + αₜ)

   The term (1+αₜ) injects the cumulative diffusion “force” into the routing cost.
3. Uses the above cost in a ternary‑branching Dijkstra‑like search to obtain a minimum‑cost
   path (the *ternary router* component).
4. Simultaneously allocates a total budget of units across the week using the same cumulative
   schedule, thus re‑using the diffusion mathematics for the workshare component.

The resulting system is a single, mathematically coherent hybrid that fuses diffusion,
weekday allocation and ternary routing through shared vector‑based operations.
"""

import sys
import pathlib
import math
import random
import numpy as np
from collections import deque
from datetime import date

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
class Morphology:
    """Stores the morphology of a physical object."""
    __slots__ = ("length", "width", "height", "mass")
    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive.")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    def as_vector(self) -> np.ndarray:
        """Return the 4‑dimensional feature vector."""
        return np.array([self.length, self.width, self.height, self.mass], dtype=float)


# ----------------------------------------------------------------------
# Diffusion schedule utilities (from Parent A)
# ----------------------------------------------------------------------
def cosine_noise_schedule(num_steps: int, s: float = 0.008) -> np.ndarray:
    """
    Produce a cosine‑based cumulative noise schedule αₜ ∈ [0,1].
    The schedule follows the same shape used in many diffusion models.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    steps = np.arange(num_steps + 1, dtype=float)
    alphas_cum = np.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]  # normalize α₀ = 1
    return alphas_cum


def apply_noise_to_vector(vec: np.ndarray, alpha_t: float) -> np.ndarray:
    """
    Perturb a feature vector with the diffusion factor √αₜ.
    """
    if not (0.0 <= alpha_t <= 1.0):
        raise ValueError("alpha_t must be in [0,1]")
    return math.sqrt(alpha_t) * vec


# ----------------------------------------------------------------------
# Weekday workshare utilities (from Parent A)
# ----------------------------------------------------------------------
def weekday_weight_vector(seed: int = 42) -> np.ndarray:
    """
    Deterministic weekday weights w_d for d ∈ {0,…,6}.
    The vector is normalized to sum to 1.
    """
    rng = random.Random(seed)
    raw = np.array([rng.random() for _ in range(7)], dtype=float)
    return raw / raw.sum()


# ----------------------------------------------------------------------
# Ternary routing utilities (from Parent B)
# ----------------------------------------------------------------------
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Standard L2 distance."""
    return float(np.linalg.norm(v1 - v2))


def hybrid_edge_cost(
    mor_i: Morphology,
    mor_j: Morphology,
    alpha_t: float,
    weekday_idx: int,
    weekday_weights: np.ndarray,
) -> float:
    """
    Compute the hybrid edge cost c₍i,j₎(t,d) = ‖f̃ₜ(i) − f̃ₜ(j)‖₂ · w_d · (1 + αₜ).

    Parameters
    ----------
    mor_i, mor_j : Morphology
        Nodes whose similarity is evaluated.
    alpha_t : float
        Cumulative diffusion factor at timestep t.
    weekday_idx : int
        Index of the current weekday (0 = Monday … 6 = Sunday).
    weekday_weights : np.ndarray
        Normalized weekday weight vector of shape (7,).

    Returns
    -------
    float
        Weighted edge cost.
    """
    vec_i = apply_noise_to_vector(mor_i.as_vector(), alpha_t)
    vec_j = apply_noise_to_vector(mor_j.as_vector(), alpha_t)
    base_dist = euclidean_distance(vec_i, vec_j)
    weight = weekday_weights[weekday_idx]
    return base_dist * weight * (1.0 + alpha_t)


def ternary_dijkstra(
    graph: dict[int, list[int]],
    morphologies: dict[int, Morphology],
    src: int,
    dst: int,
    alpha_t: float,
    weekday_idx: int,
    weekday_weights: np.ndarray,
) -> list[int]:
    """
    Dijkstra‑like shortest‑path search that, at each expansion step,
    only considers the three lowest‑cost outgoing edges (the “ternary” constraint).

    Returns the list of node IDs forming the path from src to dst,
    or an empty list if no path exists.
    """
    num_nodes = len(morphologies)
    INF = float('inf')
    dist = np.full(num_nodes, INF, dtype=float)
    prev = np.full(num_nodes, -1, dtype=int)

    dist[src] = 0.0
    visited = np.full(num_nodes, False, dtype=bool)

    while True:
        # Select the unvisited node with smallest tentative distance
        unvisited_idxs = np.where(~visited)[0]
        if unvisited_idxs.size == 0:
            break
        u = unvisited_idxs[np.argmin(dist[unvisited_idxs])]
        if dist[u] == INF or u == dst:
            break
        visited[u] = True

        # Compute costs to neighbours, keep only three smallest
        neighbours = graph.get(u, [])
        costs = []
        for v in neighbours:
            if visited[v]:
                continue
            c = hybrid_edge_cost(
                morphologies[u],
                morphologies[v],
                alpha_t,
                weekday_idx,
                weekday_weights,
            )
            costs.append((c, v))
        # Ternary pruning
        costs.sort(key=lambda x: x[0])
        for c, v in costs[:3]:
            alt = dist[u] + c
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u

    # Reconstruct path
    if dist[dst] == INF:
        return []
    path = deque()
    cur = dst
    while cur != -1:
        path.appendleft(cur)
        cur = prev[cur]
    return list(path)


# ----------------------------------------------------------------------
# Diffusion‑driven weekday allocation (from Parent A)
# ----------------------------------------------------------------------
def diffusion_allocation(
    total_units: int,
    schedule: np.ndarray,
    weekday_weights: np.ndarray,
) -> np.ndarray:
    """
    Allocate `total_units` across the seven weekdays using the cumulative
    diffusion schedule.  The allocation for day d is:

        a_d = round( total_units * w_d * (1 + α_T) / Z )

    where α_T is the final schedule value and Z normalizes the sum to total_units.
    """
    if total_units < 0:
        raise ValueError("total_units must be non‑negative")
    if schedule.ndim != 1 or weekday_weights.shape != (7,):
        raise ValueError("Invalid shape for schedule or weekday_weights")
    alpha_T = schedule[-1]  # final cumulative noise factor
    raw = weekday_weights * (1.0 + alpha_T)
    raw = raw / raw.sum()  # ensure it sums to 1
    allocation = np.rint(raw * total_units).astype(int)

    # Fix rounding errors to guarantee exact total
    diff = total_units - allocation.sum()
    if diff != 0:
        # Distribute the remainder to the days with largest fractional parts
        fractions = raw * total_units - allocation
        order = np.argsort(-fractions)
        for i in range(abs(diff)):
            idx = order[i % 7]
            allocation[idx] += int(math.copysign(1, diff))
    return allocation


# ----------------------------------------------------------------------
# High‑level hybrid operation exposing at least three functions
# ----------------------------------------------------------------------
def build_sample_graph() -> tuple[dict[int, list[int]], dict[int, Morphology]]:
    """
    Construct a tiny undirected graph with 6 nodes and random morphology.
    Returns (adjacency_dict, morphologies_dict).
    """
    rng = random.Random(123)
    morphologies = {}
    for i in range(6):
        morphologies[i] = Morphology(
            length=rng.uniform(0.5, 3.0),
            width=rng.uniform(0.5, 3.0),
            height=rng.uniform(0.5, 3.0),
            mass=rng.uniform(1.0, 10.0),
        )
    # Simple ring topology + a chord
    graph = {
        0: [1, 5],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 5],
        5: [4, 0, 2],  # extra chord to increase connectivity
    }
    return graph, morphologies


def hybrid_route_and_allocate(
    src: int,
    dst: int,
    total_units: int,
    num_timesteps: int = 10,
    weekday_idx: int = date.today().weekday(),
) -> tuple[list[int], np.ndarray]:
    """
    Perform a hybrid operation:
    1. Generate a cosine noise schedule.
    2. Compute a ternary‑router path using the schedule at the final timestep.
    3. Allocate `total_units` across weekdays using the same schedule.

    Returns (path, allocation_vector).
    """
    if not (0 <= weekday_idx < 7):
        raise ValueError("weekday_idx must be in 0..6")
    schedule = cosine_noise_schedule(num_timesteps)
    weekday_weights = weekday_weight_vector()
    graph, morphologies = build_sample_graph()

    # Use the final cumulative alpha for routing (most diffused state)
    alpha_T = schedule[-1]
    path = ternary_dijkstra(
        graph,
        morphologies,
        src,
        dst,
        alpha_T,
        weekday_idx,
        weekday_weights,
    )
    allocation = diffusion_allocation(total_units, schedule, weekday_weights)
    return path, allocation


def evaluate_hybrid_cost(
    path: list[int],
    graph: dict[int, list[int]],
    morphologies: dict[int, Morphology],
    schedule: np.ndarray,
    weekday_idx: int,
) -> float:
    """
    Given a path, compute the total hybrid cost by summing the edge costs
    at each timestep of the schedule and averaging over timesteps.
    """
    if not path:
        return float('inf')
    weekday_weights = weekday_weight_vector()
    total = 0.0
    for t, alpha_t in enumerate(schedule):
        step_cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            step_cost += hybrid_edge_cost(
                morphologies[a],
                morphologies[b],
                alpha_t,
                weekday_idx,
                weekday_weights,
            )
        total += step_cost
    return total / len(schedule)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters for the demo
    SOURCE_NODE = 0
    TARGET_NODE = 4
    TOTAL_UNITS = 100
    TIMESTEPS = 12

    # Run the hybrid routine
    path, allocation = hybrid_route_and_allocate(
        src=SOURCE_NODE,
        dst=TARGET_NODE,
        total_units=TOTAL_UNITS,
        num_timesteps=TIMESTEPS,
        weekday_idx=date.today().weekday(),
    )

    # Simple verification output
    print("Hybrid path (node IDs):", path)
    print("Weekday allocation (Mon→Sun):", allocation.tolist())
    print("Allocation sum check:", allocation.sum() == TOTAL_UNITS)

    # Additional cost evaluation (optional)
    graph, morphologies = build_sample_graph()
    schedule = cosine_noise_schedule(TIMESTEPS)
    cost = evaluate_hybrid_cost(
        path,
        graph,
        morphologies,
        schedule,
        weekday_idx=date.today().weekday(),
    )
    print(f"Average hybrid cost over {TIMESTEPS} timesteps: {cost:.4f}")