# DARWIN HAMMER — match 1220, survivor 5
# gen: 3
# parent_a: hybrid_possum_filter_hybrid_privacy_model_m53_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py (gen1)
# born: 2026-05-29T23:34:28Z

"""
Hybrid Spatial‑Privacy / Fractional‑Memory Tree Model

Parents:
* **hybrid_possum_filter_hybrid_privacy_model_m53_s0.py** – provides
  `Entity`, haversine distance, and a spatial‑aware privacy risk vector.
* **hybrid_caputo_fractional_minimum_cost_tree_m35_s6.py** – provides a Lanczos
  Gamma approximation, Caputo fractional kernel weights, and a fractional‑memory
  tree cost formulation.

Mathematical Bridge:
Both parents rely on a *summation over history*.  The privacy model yields a
risk scalar `r_i ∈ [0,1]` for each entity `i`.  The fractional‑memory tree replaces a
plain sum of root‑to‑node distances by a Caputo‑weighted sum.  By multiplying the
distance of each node by its privacy risk we obtain a **risk‑aware fractional
memory term**:

    C_hybrid = Σ_edge length
               + λ · Σ_{k=1}^{T} w_k(α) · d_k · r_k

where `d_k` is the distance from the root to the `k`‑th inserted node,
`r_k` its privacy risk, and `w_k(α)` the normalized Caputo kernel
`φ(t‑k;α) = (t‑k)^{α‑1}/Γ(α)`.  This module implements the combined
computation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Spatial entities and privacy risk
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Entity:
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
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _signature(e: Entity) -> str:
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(entities: List[Entity], delta_m: float) -> np.ndarray:
    """
    For each entity i compute a privacy risk based on other entities that share the
    same signature and lie within `delta_m` metres.
    """
    n = len(entities)
    risks = np.empty(n, dtype=float)
    for i, e_i in enumerate(entities):
        similar = [
            e_j
            for j, e_j in enumerate(entities)
            if i != j
            and _signature(e_i) == _signature(e_j)
            and haversine_m((e_i.lat, e_i.lon), (e_j.lat, e_j.lon)) <= delta_m
        ]
        uniq = len({e.id for e in similar})
        risks[i] = reconstruction_risk_score(uniq, n)
    return risks


# ----------------------------------------------------------------------
# Parent B – Caputo fractional utilities
# ----------------------------------------------------------------------


_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real `z > 0`."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_weights(T: int, alpha: float) -> np.ndarray:
    """
    Normalized Caputo kernel weights for a history of length `T`.
    w_k(α) = (T‑k)^{α‑1} / Γ(α)   for k = 1…T,
    then normalized so Σ w_k = 1.
    """
    if T <= 0:
        return np.array([], dtype=float)
    ks = np.arange(1, T + 1)
    raw = (T - ks + 1) ** (alpha - 1) / gamma_lanczos(alpha)
    total = raw.sum()
    if total == 0:
        return np.full(T, 1.0 / T, dtype=float)
    return raw / total


# ----------------------------------------------------------------------
# Hybrid core – tree construction and fractional‑privacy cost
# ----------------------------------------------------------------------


def _build_prim_tree(entities: List[Entity]) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float], List[int]]:
    """
    Construct a minimum‑spanning tree (MST) using Prim's algorithm on the haversine
    distances between entities.

    Returns:
        parent: dict mapping child index -> parent index (root has parent -1)
        edge_len: dict mapping (parent, child) -> distance
        order: list of node indices in the order they were added to the tree
    """
    n = len(entities)
    if n == 0:
        return {}, {}, []

    in_tree = [False] * n
    parent = {0: -1}
    edge_len = {}
    order = [0]  # start with node 0 as root
    in_tree[0] = True

    # min distance to the current tree for each outside node
    min_dist = [float("inf")] * n
    nearest = [-1] * n
    for i in range(1, n):
        d = haversine_m((entities[0].lat, entities[0].lon), (entities[i].lat, entities[i].lon))
        min_dist[i] = d
        nearest[i] = 0

    while len(order) < n:
        # pick the outside node with smallest distance to the tree
        cand = min((i for i in range(n) if not in_tree[i]), key=lambda i: min_dist[i])
        p = nearest[cand]
        parent[cand] = p
        edge_len[(p, cand)] = min_dist[cand]
        in_tree[cand] = True
        order.append(cand)

        # update distances for remaining nodes
        for i in range(n):
            if not in_tree[i]:
                d = haversine_m((entities[cand].lat, entities[cand].lon),
                                (entities[i].lat, entities[i].lon))
                if d < min_dist[i]:
                    min_dist[i] = d
                    nearest[i] = cand

    return parent, edge_len, order


def _root_to_node_distances(parent: Dict[int, int], edge_len: Dict[Tuple[int, int], float]) -> np.ndarray:
    """
    Compute distance from the root (node 0) to every node following the tree edges.
    """
    n = max(parent.keys()) + 1
    dist = np.zeros(n, dtype=float)
    for node in range(1, n):
        p = parent[node]
        dist[node] = dist[p] + edge_len[(p, node)]
    return dist


def hybrid_fractional_privacy_tree_cost(
    entities: List[Entity],
    alpha: float,
    lam: float,
    delta_m: float,
) -> float:
    """
    Compute the hybrid cost:
        C = Σ_edge length  +  λ· Σ_{k} w_k(α)· d_k· r_k

    Parameters
    ----------
    entities : List[Entity]
        The dataset (order is irrelevant; the algorithm builds its own tree).
    alpha : float
        Fractional order in (0,1] for the Caputo kernel.
    lam : float
        Path‑weight scaling factor.
    delta_m : float
        Spatial neighbourhood radius for privacy risk computation.

    Returns
    -------
    float
        The hybrid cost value.
    """
    if not entities:
        return 0.0

    # 1️⃣ Privacy risk per entity
    risk_vec = spatial_aware_privacy_risk_vector(entities, delta_m)

    # 2️⃣ Build MST and obtain insertion order
    parent, edge_len, order = _build_prim_tree(entities)

    # 3️⃣ Material cost = sum of edge lengths
    material_cost = sum(edge_len.values())

    # 4️⃣ Distances from root for each node (aligned with original indices)
    root_dist = _root_to_node_distances(parent, edge_len)

    # 5️⃣ Align distances and risks with insertion order
    d_ordered = np.array([root_dist[i] for i in order], dtype=float)
    r_ordered = np.array([risk_vec[i] for i in order], dtype=float)

    # 6️⃣ Fractional Caputo weights (length = number of nodes)
    w = caputo_weights(len(order), alpha)

    # 7️⃣ Weighted privacy‑aware path term
    path_term = lam * float(np.sum(w * d_ordered * r_ordered))

    return material_cost + path_term


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------


def generate_random_entities(n: int, seed: int = 42) -> List[Entity]:
    """Create `n` random entities spread over a rough lat/lon box."""
    random.seed(seed)
    entities = []
    for i in range(n):
        lat = random.uniform(-45.0, 45.0)
        lon = random.uniform(-90.0, 90.0)
        cat = random.choice(["A", "B", "C"])
        entities.append(Entity(id=str(i), lat=lat, lon=lon, category=cat))
    return entities


def demo_hybrid_cost():
    """Run a small demo and print the hybrid cost."""
    ents = generate_random_entities(15)
    alpha = 0.7          # fractional order
    lam = 0.5            # scaling of path term
    delta_m = 5_000.0    # 5 km neighbourhood for privacy risk
    cost = hybrid_fractional_privacy_tree_cost(ents, alpha, lam, delta_m)
    print(f"Hybrid fractional‑privacy tree cost (α={alpha}, λ={lam}): {cost:.3f}")


if __name__ == "__main__":
    demo_hybrid_cost()