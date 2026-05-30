# DARWIN HAMMER — match 3871, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_sheaf__hybrid_hybrid_hybrid_m2030_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1209_s7.py (gen6)
# born: 2026-05-29T23:52:08Z

"""Hybrid algorithm merging:
- Parent A: regret‑weighted strategy with epistemic certainty applied to sheaf restriction maps.
- Parent B: geometric Voronoi partitioning, Ollivier‑Ricci curvature, pheromone‑entropy, and reconstruction‑risk‑adjusted causal effect.

Mathematical bridge:
The sheaf’s restriction maps live on graph edges.  Ollivier‑Ricci curvature 𝜅(e) provides a
similarity weight for each edge based on the geometry of the underlying point cloud.
We therefore weight the regret‑weighted edge cost by (1+𝜅(e)), and propagate the
epistemic certainty of the associated action through the sheaf coboundary.
Conversely, each Voronoi region i inherits a “regional sheaf’’ whose
restriction‑map statistics (average regret weight) are used as a modifier of the
reconstruction‑risk‑adjusted average treatment effect (ATE_i).  The final hybrid
objective combines the edge‑wise regret‑curvature term with the region‑wise
risk‑curvature‑ATE term in a single normalized scalar.

The implementation below provides three core functions:
1. `regret_weighted_strategy` – computes edge‑wise regret weights using cost,
   probability and epistemic certainty.
2. `curvature_weighted_ate` – builds Voronoi regions, estimates mean curvature per
   region, and returns a risk‑curvature‑adjusted WATE.
3. `hybrid_metric` – fuses the two quantities into a unified score.
"""

import math
import random
import sys
import pathlib
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (sheaf & regret weighting)
# ----------------------------------------------------------------------

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_MAP: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.0,
    "SURE_MAYBE": 0.3,
}

@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class Action:
    """Action with cost, success probability and epistemic certainty."""
    cost: float
    probability: float
    epistemic_certainty: str

    def epistemic_factor(self) -> float:
        return _EPISTEMIC_MAP.get(self.epistemic_certainty.upper(), 0.0)

class Sheaf:
    """Cellular sheaf over a simple undirected graph.

    Each node carries a 1‑dimensional stalk; each edge carries a linear restriction map.
    """

    def __init__(self, node_dims: Dict[Any, int], edge_dims: Dict[Tuple[Any, Any], int]):
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.restriction_maps: Dict[Tuple[Any, Any], np.ndarray] = {}

    def add_restriction_map(self, edge: Tuple[Any, Any], matrix: np.ndarray) -> None:
        """Store a restriction map for an edge."""
        if matrix.shape != (self.edge_dims[edge], self.node_dims[edge[0]]):
            raise ValueError("Matrix shape does not match declared dimensions")
        self.restriction_maps[edge] = matrix

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Probability of pruning an edge after time `t`."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))

def regret_weighted_strategy(actions: List[Action]) -> List[float]:
    """Return a regret weight for each action.

    The regret weight is defined as:
        w = cost * (1 - probability) * epistemic_factor
    """
    weights = []
    for a in actions:
        factor = a.epistemic_factor()
        w = a.cost * (1.0 - a.probability) * factor
        weights.append(w)
    return weights

# ----------------------------------------------------------------------
# Parent B components (geometry, curvature, risk, ATE)
# ----------------------------------------------------------------------

Point = Tuple[float, float]

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: List[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to its nearest seed → Voronoi regions."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def approximate_curvature(seed_i: Point, seed_j: Point) -> float:
    """Very rough Ollivier‑Ricci curvature proxy.

    Using an exponential decay of Euclidean distance:
        κ = exp( - d(i,j) )
    """
    d = distance(seed_i, seed_j)
    return math.exp(-d)

def region_mean_curvature(i: int, seeds: List[Point], adjacency: Dict[int, List[int]]) -> float:
    """Mean curvature of region i with its Voronoi neighbours."""
    neigh = adjacency.get(i, [])
    if not neigh:
        return 0.0
    curvatures = [approximate_curvature(seeds[i], seeds[j]) for j in neigh]
    return sum(curvatures) / len(curvatures)

def build_adjacency(seeds: List[Point]) -> Dict[int, List[int]]:
    """Construct a simple adjacency graph among seeds using k‑nearest (k=3)."""
    k = 3
    adj: Dict[int, List[int]] = {i: [] for i in range(len(seeds))}
    for i, si in enumerate(seeds):
        dists = [(j, distance(si, sj)) for j, sj in enumerate(seeds) if j != i]
        dists.sort(key=lambda x: x[1])
        for j, _ in dists[:k]:
            adj[i].append(j)
    return adj

def curvature_weighted_ate(
    points: List[Point],
    seeds: List[Point],
    ate_per_region: List[float],
    risk_per_region: List[float],
) -> float:
    """Compute the risk‑curvature‑adjusted weighted average treatment effect (WATE)."""
    regions = assign(points, seeds)
    adjacency = build_adjacency(seeds)

    numer = 0.0
    denom = 0.0
    for i in range(len(seeds)):
        κ_i = region_mean_curvature(i, seeds, adjacency)
        ρ_i = risk_per_region[i]
        ATE_i = ate_per_region[i]
        weight = ρ_i * (1.0 + κ_i)
        numer += weight * ATE_i
        denom += weight
    return numer / denom if denom != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid core: fuse regret‑curvature edge term with region‑wise WATE
# ----------------------------------------------------------------------

def hybrid_metric(
    graph_edges: List[Tuple[Any, Any]],
    sheaf: Sheaf,
    actions_per_edge: Dict[Tuple[Any, Any], List[Action]],
    points: List[Point],
    seeds: List[Point],
    ate_per_region: List[float],
    risk_per_region: List[float],
) -> float:
    """
    Compute a unified hybrid score.

    1. For each edge e = (u, v):
         - Compute regret weight w_e from the actions attached to e.
         - Estimate curvature κ_e between the Voronoi seeds that contain u and v.
         - Edge contribution = w_e * (1 + κ_e).

    2. Compute region‑wise risk‑curvature‑adjusted WATE (from Parent B).

    3. Return a normalized combination:
         hybrid = ( Σ_e contribution ) / |E|  +  WATE
    """
    # --- Edge‑wise part -------------------------------------------------
    # Map each node to the index of the Voronoi seed that contains it.
    node_to_region = {}
    node_regions = assign(list(sheaf.node_dims.keys()), seeds)
    for region_idx, nodes in node_regions.items():
        for n in nodes:
            node_to_region[n] = region_idx

    edge_contrib_sum = 0.0
    for e in graph_edges:
        # regret weight for edge e
        actions = actions_per_edge.get(e, [])
        if not actions:
            continue
        regret_weights = regret_weighted_strategy(actions)
        w_e = sum(regret_weights) / len(regret_weights)  # average per edge

        # curvature between the regions of its endpoints
        r_u = node_to_region.get(e[0])
        r_v = node_to_region.get(e[1])
        if r_u is None or r_v is None:
            κ_e = 0.0
        else:
            κ_e = approximate_curvature(seeds[r_u], seeds[r_v])

        edge_contrib_sum += w_e * (1.0 + κ_e)

    edge_term = edge_contrib_sum / max(1, len(graph_edges))

    # --- Region‑wise part -----------------------------------------------
    wate = curvature_weighted_ate(points, seeds, ate_per_region, risk_per_region)

    # --- Fusion ---------------------------------------------------------
    hybrid_score = edge_term + wate
    return hybrid_score

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph
    nodes = ["A", "B", "C", "D"]
    node_dims = {n: 1 for n in nodes}
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    edge_dims = {e: 1 for e in edges}
    sheaf = Sheaf(node_dims, edge_dims)

    # Dummy restriction maps (identity)
    for e in edges:
        sheaf.add_restriction_map(e, np.identity(1))

    # Actions per edge
    actions_per_edge = {
        ("A", "B"): [Action(cost=3.0, probability=0.7, epistemic_certainty="FACT")],
        ("B", "C"): [Action(cost=2.0, probability=0.4, epistemic_certainty="PROBABLE")],
        ("C", "D"): [Action(cost=5.0, probability=0.2, epistemic_certainty="POSSIBLE")],
        ("D", "A"): [Action(cost=1.0, probability=0.9, epistemic_certainty="SURE_MAYBE")],
    }

    # Geometry for Parent B
    points = [(random.random(), random.random()) for _ in range(50)]
    seeds = [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]
    ate_per_region = [0.5, -0.1, 0.3]
    risk_per_region = [0.9, 0.6, 0.8]

    score = hybrid_metric(
        graph_edges=edges,
        sheaf=sheaf,
        actions_per_edge=actions_per_edge,
        points=points,
        seeds=seeds,
        ate_per_region=ate_per_region,
        risk_per_region=risk_per_region,
    )
    print(f"Hybrid score: {score:.4f}")