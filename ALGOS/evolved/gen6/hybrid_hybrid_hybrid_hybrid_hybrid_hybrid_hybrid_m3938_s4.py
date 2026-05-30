# DARWIN HAMMER — match 3938, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s1.py (gen4)
# born: 2026-05-29T23:52:39Z

"""Hybrid Stylometry‑Sheaf Hyperdimensional Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_doomsd_m857_s2.py (stylometry + HDC + Gini)
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s1.py (sheaf cohomology + semantic similarity + Bayesian update)

Mathematical Bridge:
Both parents manipulate high‑dimensional representations.  This fusion treats each *sheaf node* as a
stylometric hypervector (built from symbol frequencies, bound to a Gini‑derived hypervector).  Edge
restrictions become probability vectors that are updated by a Bayesian rule whose likelihood is the
semantic similarity (dot‑product) of the incident node hypervectors.  The final global representation
is obtained by bundling all node hypervectors and binding the bundle with an aggregated Gini
hypervector, thus unifying stylometry, temporal inequality, sheaf topology, semantic similarity,
and Bayesian inference in a single hyperdimensional system.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent A)
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 1000, seed: int | str | None = None) -> Vector:
    """Deterministic bipolar (+1 / -1) hypervector."""
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 1000) -> Vector:
    """Hash a symbol to a deterministic seed and generate its hypervector."""
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise multiplication (binding)."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: Iterable[Vector]) -> Vector:
    """Superposition (addition) followed by bipolar thresholding."""
    summed = np.sum(np.array(list(vectors)), axis=0)
    return [1 if v >= 0 else -1 for v in summed]

# ----------------------------------------------------------------------
# Sheaf structures (from Parent B)
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)                # node_id -> dimension (unused but kept)
        self.edges = list(edge_list)                    # list of (u, v)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: Iterable[float], dst_map: Iterable[float]) -> None:
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float), np.array(dst_map, dtype=float))

    def set_section(self, node: int, value: Iterable[int]) -> None:
        self._sections[node] = np.array(value, dtype=int)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections[node]

    def get_restriction(self, edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

# ----------------------------------------------------------------------
# Utility functions shared by both parents
# ----------------------------------------------------------------------
def gini_coefficient(values: List[int]) -> float:
    """Compute Gini coefficient for a list of non‑negative numbers."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(values)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    gini = (2 * cumulative) / (n * total) - (n + 1) / n
    return gini

def weekday_distribution(start: datetime, days: int = 30) -> List[int]:
    """Count occurrences of each weekday in a sliding window."""
    counts = [0] * 7
    for i in range(days):
        d = start + timedelta(days=i)
        counts[d.weekday()] += 1
    return counts

def stylometric_proportions(text: str) -> Dict[str, float]:
    """Simple token‑frequency proportions for alphanumeric tokens."""
    tokens = [tok for tok in text.split() if tok.isalnum()]
    total = len(tokens) or 1
    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return {tok: cnt / total for tok, cnt in freq.items()}

def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two hypervectors (range -1..1)."""
    if np.all(vec_a == 0) or np.all(vec_b == 0):
        return 0.0
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Bayesian posterior proportional to prior * likelihood, normalized to sum 1."""
    posterior = prior * likelihood
    total = posterior.sum()
    if total == 0:
        return prior  # fallback to prior if likelihood kills everything
    return posterior / total

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_hyper_sheaf(node_dims: Dict[int, int],
                     edge_list: List[Tuple[int, int]],
                     text: str,
                     reference_date: datetime = None) -> Sheaf:
    """
    Construct a Sheaf where each node section is a stylometric hypervector.
    The hypervector is built by:
      1. weighting symbol hypervectors by their token proportion,
      2. bundling them,
      3. binding the bundle with a Gini‑derived hypervector based on weekday distribution.
    Edge restrictions are initialized as uniform probability vectors.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc)

    # 1. Stylometric weighting
    proportions = stylometric_proportions(text)
    symbol_hvs = [bind(symbol_vector(sym), random_vector(dim=1000, seed=int(prop * 1e6)))
                  for sym, prop in proportions.items()
                  for _ in range(1)]  # each symbol contributes once weighted by its proportion via seed

    base_hv = bundle(symbol_hvs) if symbol_hvs else random_vector(dim=1000)

    # 2. Gini hypervector from weekday distribution
    weekdays = weekday_distribution(reference_date, days=30)
    gini = gini_coefficient(weekdays)
    gini_hv = random_vector(dim=1000, seed=int(gini * 1e9))

    # 3. Final node hypervector (bind base with Gini)
    node_hv = bind(base_hv, gini_hv)

    sheaf = Sheaf(node_dims, edge_list)

    # Populate node sections
    for node in node_dims.keys():
        sheaf.set_section(node, node_hv)

    # Initialize edge restrictions with uniform priors (simple probability vectors)
    uniform = np.full(1000, 1.0 / 1000)
    for edge in edge_list:
        sheaf.set_restriction(edge, uniform, uniform)

    return sheaf

def update_sheaf_edges(sheaf: Sheaf) -> None:
    """
    For each edge (u, v):
      - compute semantic similarity of the incident node hypervectors,
      - treat similarity (scaled to [0,1]) as likelihood,
      - perform Bayesian update on both source and destination restriction maps.
    """
    for edge in sheaf.edges:
        u, v = edge
        hv_u = sheaf.get_section(u)
        hv_v = sheaf.get_section(v)

        sim = semantic_similarity(hv_u, hv_v)          # -1 .. 1
        likelihood = np.clip((sim + 1) / 2, 0, 1)      # map to 0 .. 1

        src_map, dst_map = sheaf.get_restriction(edge)

        # Use the same likelihood vector for all dimensions (simple model)
        likelihood_vec = np.full_like(src_map, likelihood)

        new_src = bayes_update(src_map, likelihood_vec)
        new_dst = bayes_update(dst_map, likelihood_vec)

        sheaf.set_restriction(edge, new_src, new_dst)

def fuse_sheaf_hypervector(sheaf: Sheaf) -> Vector:
    """
    Produce a global hypervector by:
      1. bundling all node sections,
      2. computing an aggregated Gini coefficient from all edge restriction priors,
      3. binding the bundle with a Gini‑derived hypervector.
    """
    node_vectors = [list(sheaf.get_section(node)) for node in sheaf.node_dims.keys()]
    bundled = bundle(node_vectors)

    # Aggregate Gini from edge priors (treat each restriction map as a distribution)
    all_priors = []
    for edge in sheaf.edges:
        src, dst = sheaf.get_restriction(edge)
        all_priors.append(src)
        all_priors.append(dst)

    if all_priors:
        # flatten and convert to counts for Gini
        flat = np.concatenate(all_priors)
        # scale to integer counts for Gini (multiply by 1e6)
        counts = (flat * 1e6).astype(int).tolist()
        agg_gini = gini_coefficient(counts)
    else:
        agg_gini = 0.0

    gini_hv = random_vector(dim=len(bundled), seed=int(agg_gini * 1e9))
    return bind(bundled, gini_hv)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple example with three nodes and two edges
    node_dims = {0: 1000, 1: 1000, 2: 1000}
    edges = [(0, 1), (1, 2)]

    sample_text = "The quick brown fox jumps over the lazy dog. The quick fox is quick."

    sheaf = build_hyper_sheaf(node_dims, edges, sample_text)

    # Update edge restrictions based on semantic similarity
    update_sheaf_edges(sheaf)

    # Fuse everything into a single hypervector
    final_hv = fuse_sheaf_hypervector(sheaf)

    # Verify dimensions and print a short checksum
    assert len(final_hv) == 1000
    checksum = sum(final_hv)  # will be between -1000 and 1000
    print(f"Final hypervector length: {len(final_hv)}, checksum: {checksum}")