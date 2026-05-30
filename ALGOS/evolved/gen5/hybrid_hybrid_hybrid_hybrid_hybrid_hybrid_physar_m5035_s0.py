# DARWIN HAMMER — match 5035, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hoeffding_tre_m16_s3.py (gen3)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_voronoi_parti_m2081_s0.py (gen4)
# born: 2026-05-29T23:59:26Z

"""Hybrid Physarum‑Voronoi‑Tropical Algorithm
================================================

This module fuses the two parent algorithms:

* **Parent A** – tropical algebra utilities (max‑plus addition `t_add`,
  max‑plus multiplication `t_mul`, tropical matrix multiplication
  `t_matmul`), Hoeffding‑bound split decision and sketching utilities.
* **Parent B** – Physarum‑inspired flux dynamics on a weighted graph,
  Voronoi partitioning of the node set and conductance update rules.

**Mathematical bridge**

A Physarum network is represented by a conductance matrix **C**.
In the tropical world the “addition’’ is `max` and the “multiplication’’ is
ordinary `+`.  Consequently the pressure potentials at the nodes can be
computed as a tropical matrix‑vector product


p = C ⊗ 1          (⊗ = tropical matrix‑multiplication)


where the vector of ones is the tropical neutral element (0).  The
Voronoi partition supplies a region‑wise scaling factor `α_r`.  The
conductance update of an edge `(i,j)` uses the Physarum flux
`q_ij = g_ij / ℓ_ij * (p_i - p_j)` and the Hoeffding‑bound split decision
to decide whether the Voronoi cell containing the edge should be split.
All three worlds therefore interact through the same numeric objects:
the conductance matrix (tropical), the flux (physics) and the split
decision (statistics).

The code below implements this hybrid system with three public functions
that demonstrate the combined behaviour:

1. `tropical_matmul_conductance` – tropical matrix multiplication that also
   returns a conductance‑aware matrix.
2. `physarum_voronoi_step` – a full update step: Voronoi assignment,
   pressure computation via tropical algebra, flux calculation and
   conductance update.
3. `should_split_region` – Hoeffding‑bound based region split decision
   using the gain (increase of total conductance) observed in the step.

A tiny smoke test runs the algorithm on a 4‑node square graph."""

from __future__ import annotations
import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Tropical algebra (Parent A)
# ----------------------------------------------------------------------
def t_add(x, y):
    """Tropical addition – element‑wise maximum."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication – element‑wise sum."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical (max‑plus) matrix multiplication.
    (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Tropical polynomial evaluation.
    coeffs[k] + k * x, then max over k.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)


# ----------------------------------------------------------------------
# Hoeffding bound split decision (Parent A)
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str


def should_split(best_gain: float, second_best_gain: float,
                 r: float, delta: float, n: int,
                 tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    split = gap > eps or eps < tie_threshold
    reason = ("gap_exceeds_bound" if gap > eps
              else ("tie_threshold" if eps < tie_threshold else "wait"))
    return SplitDecision(split, eps, gap, reason)


# ----------------------------------------------------------------------
# Physarum primitives (Parent B)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       gain: float, decay: float,
                       dt: float) -> float:
    """Euler update of conductance with growth proportional to |q|."""
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Voronoi utilities (Parent B)
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_seed(point: Point, seeds: List[Point]) -> int:
    """Return index of the nearest seed to `point`."""
    if not seeds:
        raise ValueError('At least one seed required')
    distances = [euclidean_distance(point, s) for s in seeds]
    return int(np.argmin(distances))


def assign_voronoi(nodes: Dict[int, Point],
                   seeds: List[Point]) -> Dict[int, int]:
    """
    Assign each node to the index of its nearest Voronoi seed.
    Returns a mapping node_id -> seed_index.
    """
    return {nid: nearest_seed(coord, seeds) for nid, coord in nodes.items()}


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class Edge:
    u: int
    v: int
    length: float
    conductance: float = 1.0  # initial conductance


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def tropical_matmul_conductance(C: np.ndarray) -> np.ndarray:
    """
    Perform a tropical matrix multiplication of the conductance matrix
    with a vector of zeros (the tropical neutral element).  The result
    is a column vector of node potentials `p = C ⊗ 0`.
    """
    # 0 in tropical algebra is the additive identity (−inf).  Adding 0
    # leaves the matrix unchanged, so we can simply take the max over rows.
    # Using t_matmul with a column of zeros yields the same.
    zero_vec = np.zeros((C.shape[1], 1))
    potentials = t_matmul(C, zero_vec)  # shape (n, 1)
    return potentials.squeeze()


def compute_pressures(nodes: Dict[int, Point],
                      seeds: List[Point],
                      coeffs: List[float]) -> Dict[int, float]:
    """
    Tropical pressure at a node is the max over seeds of
    coeff_i + distance(node, seed_i).  This mirrors a tropical
    polynomial where the seed index plays the role of the exponent.
    """
    pressures = {}
    for nid, coord in nodes.items():
        dists = np.array([euclidean_distance(coord, s) for s in seeds])
        # tropical polynomial: coeff_i + d_i
        pressures[nid] = np.max(np.array(coeffs) + dists)
    return pressures


def physarum_voronoi_step(nodes: Dict[int, Point],
                          edges: List[Edge],
                          seeds: List[Point],
                          coeffs: List[float],
                          params: Dict[str, float]) -> Tuple[List[Edge], List[Point]]:
    """
    One hybrid iteration:

    1. Voronoi assignment of nodes.
    2. Tropical pressure computation per node.
    3. Flux calculation on each edge.
    4. Conductance update using Physarum dynamics.
    5. Optional Voronoi region split decision based on conductance gain.

    Returns the updated edge list and possibly an extended seed list.
    """
    # 1. Voronoi assignment (used later for region‑wise scaling)
    region_of = assign_voronoi(nodes, seeds)

    # 2. Tropical pressures
    pressures = compute_pressures(nodes, seeds, coeffs)

    # 3‑4. Flux & conductance update
    total_gain_per_region = defaultdict(float)
    for e in edges:
        p_u = pressures[e.u]
        p_v = pressures[e.v]
        q = flux(e.conductance, e.length, p_u, p_v)

        # region‑wise scaling factor α_r = 1 + size_of_region (simple heuristic)
        region = region_of[e.u]  # assume both ends belong to same region or ignore mismatch
        region_size = sum(1 for r in region_of.values() if r == region)
        alpha = 1.0 + region_size * 0.01  # modest boost for larger regions

        # Update conductance with scaled gain
        new_g = update_conductance(
            e.conductance,
            q,
            gain=params.get('gain', 1.0) * alpha,
            decay=params.get('decay', 0.1),
            dt=params.get('dt', 0.01)
        )
        total_gain_per_region[region] += (new_g - e.conductance)
        e.conductance = new_g

    # 5. Split decision per region (if any region shows statistically
    # significant gain over the next best region)
    if total_gain_per_region:
        best_region, best_gain = max(total_gain_per_region.items(),
                                     key=lambda kv: kv[1])
        # second best (or 0 if only one region)
        second_best_gain = max(
            (g for r, g in total_gain_per_region.items() if r != best_region),
            default=0.0
        )
        decision = should_split(best_gain, second_best_gain,
                                r=params.get('r', 1.0),
                                delta=params.get('delta', 0.05),
                                n=params.get('n', 100))
        if decision.should_split:
            # create a new seed near the centroid of the best region
            region_nodes = [nid for nid, reg in region_of.items() if reg == best_region]
            if region_nodes:
                xs = [nodes[n][0] for n in region_nodes]
                ys = [nodes[n][1] for n in region_nodes]
                new_seed = (float(np.mean(xs)), float(np.mean(ys)))
                seeds.append(new_seed)
                # extend coefficients list with a small random perturbation
                coeffs.append(coeffs[best_region] + random.uniform(-0.1, 0.1))
    return edges, seeds


def count_min_sketch_edges(edges: List[Edge],
                          width: int = 64,
                          depth: int = 4) -> List[List[int]]:
    """
    Simple count‑min sketch of edge identifiers (u,v) for monitoring.
    """
    table = [[0] * width for _ in range(depth)]
    for e in edges:
        item = f'{e.u}:{e.v}'
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny square graph
    nodes = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0)
    }
    edges = [
        Edge(0, 1, length=1.0),
        Edge(1, 2, length=1.0),
        Edge(2, 3, length=1.0),
        Edge(3, 0, length=1.0),
        Edge(0, 2, length=math.sqrt(2)),
        Edge(1, 3, length=math.sqrt(2))
    ]

    # Initial Voronoi seeds (two arbitrary points)
    seeds = [(0.2, 0.2), (0.8, 0.8)]
    # Coefficients for tropical pressure (one per seed)
    coeffs = [0.5, 0.7]

    # Hyper‑parameters
    params = {
        'gain': 1.0,
        'decay': 0.05,
        'dt': 0.02,
        'r': 1.0,
        'delta': 0.05,
        'n': 200  # number of observations for Hoeffding bound
    }

    # Run a few hybrid steps
    for step in range(5):
        edges, seeds = physarum_voronoi_step(nodes, edges, seeds, coeffs, params)
        # Build conductance matrix for inspection
        n = len(nodes)
        C = np.full((n, n), -np.inf)  # tropical -inf = neutral element for max
        for e in edges:
            C[e.u, e.v] = e.conductance
            C[e.v, e.u] = e.conductance  # undirected
        potentials = tropical_matmul_conductance(C)
        print(f"Step {step + 1}")
        print(f"  Seeds ({len(seeds)}): {seeds}")