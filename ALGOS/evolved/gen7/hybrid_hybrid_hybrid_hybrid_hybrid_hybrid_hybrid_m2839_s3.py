# DARWIN HAMMER — match 2839, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m217_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s5.py (gen6)
# born: 2026-05-29T23:46:12Z

"""Hybrid Sheaf‑Stylometry‑Voronoi Model
------------------------------------
Parent A: `hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py` – provides the
`HybridSheaf` data structure, linear restriction maps and pattern storage.

Parent B: `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1528_s5.py` – extracts a
stylometric feature vector **f**, maps it to polar seed coordinates **S(f)**, builds a
Voronoi partition, computes a temperature **T(f)** and a developmental rate
`r(T)` (Schoolfield model). The weekday‑dependent sinusoid `σ(d)` is also supplied.

Mathematical Bridge
-------------------
The radius component of each polar seed `r_i = f_i` is used as a scalar
restriction factor that projects the developmental‑rate pattern onto the node
associated with cell *i*.  For an edge *(i→j)* the source‑restriction matrix is
`R_{i→j}^{src}= [[r_i]]` and the destination‑restriction matrix is
`R_{i→j}^{dst}= [[r_j]]`.  Applying these maps to the pattern vector **p**
(yielding the rate for each cell) generates node sections that fuse the
geometric‑Voronoi description with the associative‑memory encoding of the
sheaf.

The resulting hybrid system therefore consists of:
1. A feature‑driven seed set **S(f)** → Voronoi cells → sheaf nodes.
2. A temperature‑driven pattern **p = r(T(f))·σ(d)** stored in the sheaf.
3. Linear restriction maps derived from the seed radii that project **p**
   onto node sections, completing the mathematical fusion.
"""

import sys
import math
import random
import pathlib
import re
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Iterable, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – HybridSheaf (trimmed / corrected)
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    Hybrid data structure merging Cellular Sheaf and Dense Associative Memory.
    Nodes are identified by integers 0..N‑1.
    """

    def __init__(self, node_dims: Dict[int, int], edges: List[Tuple[int, int]], patterns: np.ndarray):
        """
        node_dims : dict mapping node id → dimension (int)
        edges     : list of directed edges (u, v)
        patterns  : ndarray of shape (N, D) – one pattern row per node
        """
        self.node_dims = node_dims
        self.edges = edges
        self.patterns = np.asarray(patterns, dtype=float)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: np.ndarray, dst_map: np.ndarray) -> None:
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

    def set_section(self, node: int, value: np.ndarray) -> None:
        """Assign a vector to a node."""
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        """Retrieve the section vector for a node."""
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def apply_restrictions(self) -> None:
        """
        Propagate patterns across edges using the registered restriction maps.
        For each edge (u→v) we compute:
            sec_v += R_dst @ (R_src @ pattern_u)
        The sections are accumulated (sum) and finally stored.
        """
        # initialise empty sections
        for node, dim in self.node_dims.items():
            self._sections[node] = np.zeros(dim)

        for (u, v), (R_src, R_dst) in self._restrictions.items():
            # pattern row for node u (shape (dim_u,))
            pat_u = self.patterns[u].reshape(-1, 1)  # column vector
            # project through src map
            tmp = R_src @ pat_u                     # shape (k,1)
            # project to destination dimension
            sec = R_dst @ tmp                       # shape (dim_v,1)
            # accumulate
            self._sections[v] += sec.ravel()

# ----------------------------------------------------------------------
# Parent B – Stylometry → Voronoi → Developmental Rate
# ----------------------------------------------------------------------
# Function‑word categories (full list)
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set("no not never none nobody".split()),
}

def extract_features(text: str) -> np.ndarray:
    """
    Count occurrences of each function‑word category, normalize to sum‑to‑1.
    Returns a column vector f ∈ ℝⁿ where n = number of categories.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    counts = []
    for cat in FUNCTION_CATS.values():
        cnt = sum(1 for w in words if w in cat)
        counts.append(cnt)
    total = sum(counts) if sum(counts) > 0 else 1
    f = np.array(counts, dtype=float) / total
    return f.reshape(-1, 1)   # column vector

def polar_seeds_from_features(f: np.ndarray) -> np.ndarray:
    """
    Map each component f_i to a polar coordinate (r_i,θ_i) where
        r_i = f_i (radius)
        θ_i = 2π·i/n (angle)
    Returns an (n,2) array of Cartesian seed points.
    """
    n = f.shape[0]
    angles = 2 * math.pi * np.arange(n) / n
    radii = f.ravel()
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    return np.column_stack((xs, ys))

def nearest_seed(point: np.ndarray, seeds: np.ndarray) -> int:
    """Return index of the seed closest to `point`."""
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))

def partition_points(seeds: np.ndarray, points: np.ndarray) -> List[List[int]]:
    """
    Simple Voronoi‑like assignment: each point is assigned to the nearest seed.
    Returns a list of lists; cell i contains indices of points belonging to seed i.
    """
    cells = [[] for _ in range(len(seeds))]
    for idx, pt in enumerate(points):
        cell = nearest_seed(pt, seeds)
        cells[cell].append(idx)
    return cells

def temperature_from_features(f: np.ndarray) -> float:
    """Linear temperature model: T = 273.15 + 30·∑f_i."""
    return 273.15 + 30.0 * float(f.sum())

def schoolfield_rate(T: float) -> float:
    """
    Simplified Schoolfield (high‑temperature) model:
        r(T) = (E * T) / (1 + exp((T - T_opt)/Δ))
    where we fix E=0.01, T_opt=298 K, Δ=10 K.
    """
    E = 0.01
    T_opt = 298.0
    Δ = 10.0
    return (E * T) / (1.0 + math.exp((T - T_opt) / Δ))

def weekday_sinusoid() -> float:
    """Sinusoidal modifier based on current weekday (Monday=0)."""
    wd = datetime.utcnow().weekday()
    return 0.5 * (1 + math.sin(2 * math.pi * wd / 7))

# ----------------------------------------------------------------------
# Hybrid Functions Demonstrating the Fusion
# ----------------------------------------------------------------------
def generate_hybrid_sheaf(text: str, random_points: np.ndarray) -> HybridSheaf:
    """
    1. Extract stylometric feature vector f.
    2. Build polar seeds S(f) and Voronoi cells for `random_points`.
    3. Compute temperature T(f) and developmental rate r = r(T)·σ(weekday).
    4. Construct a HybridSheaf where each Voronoi cell is a node.
    5. Use seed radii as scalar restriction factors linking neighboring cells.
    Returns the fully built and restriction‑applied HybridSheaf.
    """
    # --- Step 1: features ---
    f = extract_features(text)                         # (n,1)

    # --- Step 2: seeds & cells ---
    seeds = polar_seeds_from_features(f)               # (n,2)
    cells = partition_points(seeds, random_points)    # list of point‑index lists
    n_cells = len(seeds)

    # --- Step 3: pattern (developmental rate) ---
    T = temperature_from_features(f)
    base_rate = schoolfield_rate(T)
    rate = base_rate * weekday_sinusoid()              # scalar
    pattern_matrix = np.full((n_cells, 1), rate)       # each node stores same rate

    # --- Step 4: sheaf construction ---
    node_dims = {i: 1 for i in range(n_cells)}         # each node 1‑dimensional
    # Simple adjacency: connect cells whose seeds are within 1.5×max radius distance
    max_dist = np.max(np.linalg.norm(seeds - seeds.mean(axis=0), axis=1)) * 1.5
    edges = []
    for i in range(n_cells):
        for j in range(n_cells):
            if i != j:
                if np.linalg.norm(seeds[i] - seeds[j]) <= max_dist:
                    edges.append((i, j))

    sheaf = HybridSheaf(node_dims=node_dims, edges=edges, patterns=pattern_matrix)

    # --- Step 5: restrictions based on seed radii ---
    radii = np.linalg.norm(seeds, axis=1)               # scalar radius per node
    for (u, v) in edges:
        src_map = np.array([[radii[u]]])                # shape (1,1)
        dst_map = np.array([[radii[v]]])                # shape (1,1)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Propagate patterns through restriction maps
    sheaf.apply_restrictions()
    return sheaf

def get_node_sections(sheaf: HybridSheaf) -> Dict[int, float]:
    """
    Retrieve the scalar section value for each node after restriction propagation.
    Returns a dict {node_id: section_value}.
    """
    sections = {}
    for node in sheaf.node_dims:
        sec_vec = sheaf.get_section(node)
        # sections are 1‑dimensional; extract scalar
        sections[node] = float(sec_vec.squeeze())
    return sections

def compute_hybrid_output(sheaf: HybridSheaf) -> np.ndarray:
    """
    Assemble a matrix whose rows correspond to nodes and columns to:
        [seed_x, seed_y, section_value]
    This matrix showcases the fused geometric (seed), sheaf (section) and
    developmental‑rate information.
    """
    # Re‑derive seeds from stored pattern dimensions (they are not stored inside sheaf,
    # so we recompute them from the pattern rows count)
    n = len(sheaf.node_dims)
    # Recover the original feature vector magnitude via average radius (heuristic)
    # Here we simply use the radii stored in the restriction maps of outgoing edges.
    # For isolated nodes we fall back to 1.0.
    radii = np.ones(n)
    for (u, v), (src_map, _) in sheaf._restrictions.items():
        radii[u] = src_map.item()  # scalar
    angles = 2 * math.pi * np.arange(n) / n
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)

    sections = np.array([sheaf.get_section(i).item() for i in range(n)])
    return np.column_stack((xs, ys, sections))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I think that the quick brown fox jumps over the lazy dog, and it does so "
        "because it is clever. However, no one knows why the fox chose that particular "
        "day."
    )
    # Generate 200 random 2‑D points inside the unit square
    rng = np.random.default_rng(42)
    points = rng.random((200, 2))

    sheaf = generate_hybrid_sheaf(sample_text, points)
    sections = get_node_sections(sheaf)
    hybrid_matrix = compute_hybrid_output(sheaf)

    print("Node sections (scalar values):")
    for nid, val in sections.items():
        print(f"  Node {nid}: {val:.6f}")

    print("\nHybrid output matrix (first 5 rows):")
    print(hybrid_matrix[:5])
    print("\nAll operations completed successfully.")