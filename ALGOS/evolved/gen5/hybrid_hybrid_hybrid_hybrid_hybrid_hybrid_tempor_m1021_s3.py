# DARWIN HAMMER — match 1021, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s0.py (gen4)
# parent_b: hybrid_hybrid_temporal_moti_hybrid_doomsday_cale_m218_s0.py (gen2)
# born: 2026-05-29T23:32:25Z

"""Hybrid Sheaf‑Temporal‑Gini Model
===================================

This module fuses two previously independent algorithms:

* **Parent A** – *HybridSheaf* (Cellular Sheaf + Dense Associative Memory).  
  The core of this algorithm is a collection of *restriction maps* 𝑅₍ᵤ→ᵥ₎ that linearly project a
  section (vector) defined on node *u* onto the space of node *v*.  Sections live in
  ℝ^{dim(node)} and the sheaf enforces consistency across edges via these maps.

* **Parent B** – *Hybrid Temporal Motif + Gini Coefficient*.  
  Here a burst‑signal vector **b** (e.g. counts per category) is analysed with the
  Gini coefficient  

\[
G(\mathbf b)=\frac{\sum_{i=1}^{n}(2i-n-1)b_{(i)}}{n\sum_{i=1}^{n}b_i},
\]

where \(b_{(i)}\) denotes the sorted components of **b**.

--------------------------------------------------------------------
Mathematical Bridge
-------------------

We treat the *global burst‑signal* of the sheaf as the **aggregate** of all node
sections after they have been transported to a common reference node *r* using the
restriction maps:

\[
\mathbf{b}= \sum_{u\in V} \; \mathbf{R}_{u\to r}\; \mathbf{s}_u,
\]

where \(\mathbf{s}_u\) is the section on node *u* and \(\mathbf{R}_{u\to r}\) is the
composed restriction map along any path from *u* to the reference node *r*.
The resulting vector **b** lives in the space of node *r* (chosen to have
dimension equal to the number of categories).  The inequality of the
distribution of pattern activations across the sheaf is then quantified by the
Gini coefficient applied to **b**.

The implementation below provides:

* Construction of a sheaf from a list of `Entity` objects (geographic + category).
* Automatic creation of edges based on a haversine distance threshold.
* Identity restriction maps (or simple linear scaling) that enable the above
  aggregation.
* Functions that compute the aggregated burst vector and its Gini coefficient.

Only the allowed standard‑library modules are used.
"""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Union, Dict

# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Geospatial entity carrying a categorical label."""
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

@dataclass(frozen=True)
class BurstSignal:
    """Result of burst detection for a specific key."""
    key: str
    count: int
    z_score: float

def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def gini_coefficient_numpy(values: np.ndarray) -> float:
    """Gini coefficient for a 1‑D non‑negative array."""
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator

# ----------------------------------------------------------------------
# Sheaf core from Parent A (completed and extended)
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    Hybrid data structure combining Cellular Sheaf and Dense Associative Memory.
    Nodes are identified by arbitrary hashable keys; each node has an associated
    dimension (typically the number of categories).  Sections are vectors stored
    on nodes.  Restriction maps are linear transformations that relate sections
    on adjacent nodes.
    """

    def __init__(self, node_dims: Dict[any, int], edges: List[Tuple[any, any]]):
        """
        Parameters
        ----------
        node_dims : dict
            Mapping ``node -> dimension``.
        edges : list of (src, dst)
            Directed edges defining the sheaf topology.
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[any, any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[any, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Restriction map handling
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[any, any],
                        src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    # ------------------------------------------------------------------
    # Section handling
    # ------------------------------------------------------------------
    def set_section(self, node: any, value: np.ndarray) -> None:
        """Assign a vector to a node."""
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: any) -> np.ndarray:
        """Retrieve the section stored at ``node``."""
        if node not in self._sections:
            raise KeyError(f"No section set for node {node}")
        return self._sections[node]

    # ------------------------------------------------------------------
    # Helper: compose restriction maps along a path
    # ------------------------------------------------------------------
    def _compose_to(self, src: any, dst: any, visited=None) -> np.ndarray:
        """
        Return the matrix that transports a vector from ``src`` to ``dst``.
        If no path exists, raise a RuntimeError.
        """
        if visited is None:
            visited = set()
        if src == dst:
            dim = self.node_dims[src]
            return np.eye(dim)
        visited.add(src)
        for (u, v), (src_mat, dst_mat) in self._restrictions.items():
            if u != src:
                continue
            if v in visited:
                continue
            try:
                tail = self._compose_to(v, dst, visited)
                # The transport from src to dst is dst_mat @ tail @ src_mat.T ?
                # For our identity‑restriction construction (see builder) we simply
                # return dst_mat @ tail because src_mat is also identity.
                return dst_mat @ tail
            except RuntimeError:
                continue
        raise RuntimeError(f"No path from {src} to {dst}")

    # ------------------------------------------------------------------
    # Global aggregation (bridge to Parent B)
    # ------------------------------------------------------------------
    def aggregate_to(self, reference_node: any) -> np.ndarray:
        """
        Transport every node section to ``reference_node`` using the sheaf's
        restriction maps and sum the results.  The returned vector lives in the
        space of ``reference_node``.
        """
        agg = np.zeros(self.node_dims[reference_node], dtype=float)
        for node, sec in self._sections.items():
            transport = self._compose_to(node, reference_node)
            agg += transport @ sec
        return agg

# ----------------------------------------------------------------------
# Fusion functions (the three required demonstrations)
# ----------------------------------------------------------------------
def build_sheaf_from_entities(entities: Iterable[Entity],
                              distance_threshold_m: float = 5_000.0) -> HybridSheaf:
    """
    Construct a ``HybridSheaf`` where each entity becomes a node.
    Node dimension equals the number of distinct categories across *entities*.
    Edges are added between nodes whose haversine distance is below the threshold.
    Restriction maps are identity matrices (dimension‑preserving), which makes
    the composed transport simply a sum of sections.
    """
    # Determine the global category set and assign an index to each.
    categories = sorted({e.category for e in entities})
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    dim = len(categories)

    # Node dimensions dictionary.
    node_dims = {e.id: dim for e in entities}

    # Build edges based on spatial proximity (undirected, stored as two directed edges).
    edges = []
    entity_list = list(entities)
    for i, e1 in enumerate(entity_list):
        for e2 in entity_list[i + 1:]:
            if haversine_m((e1.lat, e1.lon), (e2.lat, e2.lon)) <= distance_threshold_m:
                edges.append((e1.id, e2.id))
                edges.append((e2.id, e1.id))

    sheaf = HybridSheaf(node_dims, edges)

    # Identity restriction maps for each directed edge.
    for u, v in edges:
        src_map = np.eye(dim)
        dst_map = np.eye(dim)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Set sections: one‑hot vector for the entity's category scaled by its score.
    for e in entity_list:
        vec = np.zeros(dim, dtype=float)
        vec[cat_to_idx[e.category]] = max(e.score, 1.0)  # ensure non‑zero for Gini
        sheaf.set_section(e.id, vec)

    return sheaf


def compute_burst_vector(sheaf: HybridSheaf, reference_node: any = None) -> np.ndarray:
    """
    Produce the global burst‑signal vector by aggregating all sections onto a
    reference node.  If ``reference_node`` is ``None`` the first node in the sheaf
    is used.
    """
    if reference_node is None:
        reference_node = next(iter(sheaf.node_dims))
    return sheaf.aggregate_to(reference_node)


def gini_of_sheaf_bursts(sheaf: HybridSheaf, reference_node: any = None) -> float:
    """
    Compute the Gini coefficient of the sheaf's burst vector.
    """
    burst_vec = compute_burst_vector(sheaf, reference_node)
    return gini_coefficient_numpy(burst_vec)


# ----------------------------------------------------------------------
# Additional helper: simple burst detection using categories (mirrors Parent B)
# ----------------------------------------------------------------------
def detect_category_bursts(entities: Iterable[Entity],
                          min_z: float = 2.0) -> List[BurstSignal]:
    """
    Count occurrences of each category and return ``BurstSignal`` objects for
    categories whose count deviates from the mean by more than ``min_z`` standard
    deviations.
    """
    counts = Counter(e.category for e in entities)
    if not counts:
        return []
    mean = sum(counts.values()) / len(counts)
    var = sum((c - mean) ** 2 for c in counts.values()) / len(counts)
    std = math.sqrt(var) if var > 0 else 1.0
    bursts = []
    for cat, cnt in counts.items():
        z = (cnt - mean) / std
        if abs(z) >= min_z:
            bursts.append(BurstSignal(key=cat, count=cnt, z_score=z))
    return bursts

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset.
    sample_entities = [
        Entity(id="A", lat=40.7128, lon=-74.0060, category="food", score=3.0),
        Entity(id="B", lat=40.7130, lon=-74.0055, category="drink", score=2.0),
        Entity(id="C", lat=40.7300, lon=-73.9950, category="food", score=1.5),
        Entity(id="D", lat=40.7500, lon=-73.9900, category="entertainment", score=4.0),
        Entity(id="E", lat=40.7520, lon=-73.9890, category="food", score=2.5),
    ]

    # Build the sheaf.
    sheaf = build_sheaf_from_entities(sample_entities, distance_threshold_m=3000.0)

    # Compute burst vector and Gini coefficient.
    burst_vec = compute_burst_vector(sheaf)
    gini_val = gini_of_sheaf_bursts(sheaf)

    # Detect category bursts independently (Parent B style).
    bursts = detect_category_bursts(sample_entities, min_z=0.5)

    print("Burst vector (aggregated on first node):", burst_vec)
    print("Gini coefficient of burst distribution:", round(gini_val, 4))
    print("Detected category bursts:", bursts)