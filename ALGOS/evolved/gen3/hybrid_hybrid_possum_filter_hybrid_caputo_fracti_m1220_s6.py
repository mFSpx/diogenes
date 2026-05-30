# DARWIN HAMMER — match 1220, survivor 6
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""Hybrid Spatial‑Privacy Fractional Tree Model
================================================

This module fuses the two parent algorithms:

* **Parent A – ``hybrid_possum_filter_hybrid_privacy_model_m53_s0.py``**  
  Provides a *spatial‑aware privacy risk vector* for a collection of entities.
  The risk for each entity depends on the number of similar entities within a
  distance ``delta_m``.

* **Parent B – ``hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py``**  
  Supplies a *Caputo fractional‑memory weighting* and a routine for computing a
  minimum‑cost tree cost where the path‑weight term is replaced by a
  fractional‑weighted sum of root‑to‑node distances.

**Mathematical bridge**

We treat the set of entities as nodes in a geometric graph whose edge length is
the haversine distance between the two locations.  A minimum‑spanning tree (MST)
is built over this graph – this re‑uses the *tree* structure of Parent B while
the edge lengths are exactly the spatial distances used by Parent A.

The privacy risk vector ``r`` (size ``N``) is then used to *modulate* the
fractional path‑weight term.  Concretely, for a fractional order ``α ∈ (0,1]``
and a scalar ``λ`` we define the hybrid cost


C_hybrid = Σ_{e∈MST}  length(e)                         # material cost
           + λ · Σ_{k=1}^{N}  w_k(α) · d_k · r_k      # fractional privacy cost


where

* ``d_k`` is the distance from the root to node ``k`` along the tree,
* ``w_k(α)`` are the normalized Caputo kernel weights
  ``φ(t‑k;α) = (t‑k)^{α‑1} / Γ(α)`` with ``t = N``,
* ``r_k`` is the privacy risk for node ``k`` obtained from Parent A.

The three core functions below implement:

1. ``spatial_privacy_risk_vector`` – Parent A logic.
2. ``caputo_weights`` – the fractional kernel from Parent B.
3. ``hybrid_fractional_privacy_tree_cost`` – builds the MST, computes root
   distances, and evaluates the fused cost expression.

A small smoke test at the bottom demonstrates the end‑to‑end workflow on a
synthetic set of entities.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
    """Geographical entity with an optional categorical signature."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _signature(e: Entity) -> str:
    """Canonical signature used for similarity comparison."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple normalized reconstruction risk."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """
    Compute the privacy risk for each entity based on the number of *similar*
    entities (same signature) that lie within ``delta_m`` metres.
    """
    n = len(entities)
    risks = np.empty(n, dtype=float)
    for i, entity in enumerate(entities):
        similar = [
            e
            for j, e in enumerate(entities)
            if i != j
            and _signature(entity) == _signature(e)
            and haversine_m((entity.lat, entity.lon), (e.lat, e.lon)) <= delta_m
        ]
        unique_ids = {e.id for e in similar}
        risks[i] = reconstruction_risk_score(len(unique_ids), n)
    return risks


# ----------------------------------------------------------------------
# Fractional‑memory utilities (Parent B)
# ----------------------------------------------------------------------


_LANCZOS_G = 7
_LANCZOS_COEFFS = np.array(
    [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
)


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real ``z`` > 0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_COEFFS[0]
    for i in range(1, len(_LANCZOS_COEFFS)):
        x += _LANCZOS_COEFFS[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_weights(alpha: float, history_len: int) -> np.ndarray:
    """
    Normalized Caputo kernel weights for a history of length ``history_len``.

    w_k(α) = φ(t‑k;α) / Σ_{j=1}^{t} φ(t‑j;α)   with  t = history_len
    φ(τ;α) = τ^{α‑1} / Γ(α)   (τ ≥ 1)
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    t = history_len
    taus = np.arange(1, t + 1)  # τ = t‑k where k runs 1..t
    phi = taus ** (alpha - 1) / gamma_lanczos(alpha)
    weights = phi / phi.sum()
    return weights


def fractional_weighted_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """Apply Caputo weights to a 1‑D array of values."""
    if values.shape != weights.shape:
        raise ValueError("Shapes of values and weights must match")
    return float(np.dot(values, weights))


# ----------------------------------------------------------------------
# Hybrid tree construction and cost evaluation
# ----------------------------------------------------------------------


def _mst_prim(dist_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Simple Prim's algorithm returning a list of edges (u, v, length) that form
    a minimum‑spanning tree of the fully‑connected graph described by
    ``dist_matrix`` (symmetric, zero diagonal).
    """
    n = dist_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    edges = []
    while len(edges) < n - 1:
        # Mask to consider only edges from visited -> unvisited
        mask = np.logical_not(visited)
        candidate_rows = np.where(visited)[0][:, None]
        candidate_cols = np.where(mask)[0][None, :]
        submatrix = dist_matrix[candidate_rows, candidate_cols]
        min_idx = np.argmin(submatrix)
        i_rel, j_rel = divmod(min_idx, submatrix.shape[1])
        u = candidate_rows[i_rel]
        v = candidate_cols[j_rel]
        edges.append((u, v, dist_matrix[u, v]))
        visited[v] = True
    return edges


def _root_distances_from_mst(
    n: int, edges: List[Tuple[int, int, float]], root: int
) -> np.ndarray:
    """
    Given ``n`` nodes and a list of undirected MST edges, compute the distance
    from ``root`` to every node by a BFS/DFS traversal.
    """
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    distances = np.full(n, np.inf, dtype=float)
    distances[root] = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for neigh, w in adj[node]:
            if distances[neigh] == np.inf:
                distances[neigh] = distances[node] + w
                stack.append(neigh)
    return distances


def hybrid_fractional_privacy_tree_cost(
    entities: List[Entity],
    root_id: str,
    delta_m: float,
    alpha: float,
    lam: float,
) -> float:
    """
    Compute the hybrid cost:

        C = Σ_edge length  +  λ * Σ_k w_k(α) * d_k * r_k

    where
        * edges form the MST over haversine distances,
        * d_k is the root‑to‑node distance,
        * r_k is the spatial privacy risk,
        * w_k(α) are Caputo fractional weights.
    """
    if not entities:
        return 0.0

    # ------------------------------------------------------------------
    # 1. Build distance matrix
    # ------------------------------------------------------------------
    n = len(entities)
    coords = [(e.lat, e.lon) for e in entities]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(coords[i], coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # ------------------------------------------------------------------
    # 2. Minimum‑spanning tree (material cost)
    # ------------------------------------------------------------------
    mst_edges = _mst_prim(dist_matrix)
    material_cost = sum(w for _, _, w in mst_edges)

    # ------------------------------------------------------------------
    # 3. Root distances
    # ------------------------------------------------------------------
    try:
        root_index = next(i for i, e in enumerate(entities) if e.id == root_id)
    except StopIteration:
        raise ValueError(f"root_id '{root_id}' not found among entities")
    root_dist = _root_distances_from_mst(n, mst_edges, root_index)

    # ------------------------------------------------------------------
    # 4. Privacy risk vector (from Parent A)
    # ------------------------------------------------------------------
    risk_vec = spatial_privacy_risk_vector(entities, delta_m)

    # ------------------------------------------------------------------
    # 5. Fractional weights (from Parent B)
    # ------------------------------------------------------------------
    weights = caputo_weights(alpha, n)

    # ------------------------------------------------------------------
    # 6. Fractional privacy term
    # ------------------------------------------------------------------
    privacy_term = lam * np.dot(weights, root_dist * risk_vec)

    return material_cost + privacy_term


# ----------------------------------------------------------------------
# Additional demonstration functions (required ≥3)
# ----------------------------------------------------------------------


def generate_random_entities(
    count: int,
    lat_range: Tuple[float, float] = (35.0, 36.0),
    lon_range: Tuple[float, float] = (-120.0, -119.0),
    categories: Iterable[str] = ("A", "B", "C"),
) -> List[Entity]:
    """Create ``count`` synthetic entities with random coordinates and categories."""
    ents = []
    for i in range(count):
        lat = random.uniform(*lat_range)
        lon = random.uniform(*lon_range)
        cat = random.choice(list(categories))
        ents.append(Entity(id=f"e{i}", lat=lat, lon=lon, category=cat))
    return ents


def demo_hybrid_cost():
    """Run a quick demonstration of the hybrid cost on synthetic data."""
    random.seed(0)
    ents = generate_random_entities(12)
    root = ents[0].id
    delta = 5000.0          # 5 km neighbourhood for privacy risk
    alpha = 0.7             # fractional order
    lam = 0.3               # path‑weight scaling
    cost = hybrid_fractional_privacy_tree_cost(
        entities=ents,
        root_id=root,
        delta_m=delta,
        alpha=alpha,
        lam=lam,
    )
    print(f"Hybrid cost (root={root}, α={alpha}, λ={lam}): {cost:.3f}")


def compare_fractional_vs_plain(
    entities: List[Entity],
    root_id: str,
    delta_m: float,
    lam: float = 0.3,
) -> Tuple[float, float]:
    """
    Compute both the hybrid fractional cost and the plain (un‑weighted) tree
    cost for the same MST, returning a tuple ``(plain, fractional)``.
    """
    # Plain cost: material + λ * Σ d_k (no fractional weighting, no risk)
    n = len(entities)
    # distance matrix and MST as in the hybrid routine
    coords = [(e.lat, e.lon) for e in entities]
    dist_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(coords[i], coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    mst_edges = _mst_prim(dist_matrix)
    material = sum(w for _, _, w in mst_edges)
    root_index = next(i for i, e in enumerate(entities) if e.id == root_id)
    root_dist = _root_distances_from_mst(n, mst_edges, root_index)
    plain = material + lam * root_dist.sum()

    # Fractional hybrid cost
    fractional = hybrid_fractional_privacy_tree_cost(
        entities, root_id, delta_m, alpha=0.7, lam=lam
    )
    return plain, fractional


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_cost()
    # Additional sanity check
    ents = generate_random_entities(8)
    p, f = compare_fractional_vs_plain(ents, ents[0].id, delta_m=4000.0)
    print(f"Plain cost = {p:.3f} | Hybrid fractional cost = {f:.3f}")