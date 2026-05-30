# DARWIN HAMMER — match 2176, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_distributed_l_hybrid_hybrid_geomet_m641_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_model_pool_hy_m707_s1.py (gen5)
# born: 2026-05-29T23:41:13Z

import numpy as np
import random
import math
from collections.abc import Mapping, Set, Hashable
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Optional

Node = int
Graph = Dict[Node, Set[Node]]

# ----------------------------------------------------------------------
# Hash utilities
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 64‑bit binary string based on median threshold."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers interpreted as bit‑vectors."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def build_graph(elements: List[List[float]], max_hamming: int = 4) -> Graph:
    """Build an undirected similarity graph where edges join elements whose
    perceptual hashes differ by at most ``max_hamming`` bits."""
    hashes = [compute_phash(e) for e in elements]
    graph: Graph = {i: set() for i in range(len(elements))}
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if hamming_distance(hashes[i], hashes[j]) <= max_hamming:
                graph[i].add(j)
                graph[j].add(i)
    return graph


# ----------------------------------------------------------------------
# Spectral curvature (proxy)
# ----------------------------------------------------------------------
def graph_curvature(graph: Graph) -> float:
    """Return a simple curvature proxy: the spectral radius of the normalized
    Laplacian. Larger values indicate a more “curved” (irregular) graph."""
    n = len(graph)
    if n == 0:
        return 0.0
    # Build adjacency matrix in dense form (small graphs only)
    A = np.zeros((n, n), dtype=float)
    for i, neigh in graph.items():
        for j in neigh:
            A[i, j] = 1.0
    deg = A.sum(axis=1)
    # Avoid division by zero
    deg[deg == 0] = 1.0
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    L_norm = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    eigenvalues = np.linalg.eigvalsh(L_norm)
    # Spectral radius (largest eigenvalue) as curvature measure
    return float(eigenvalues.max())


# ----------------------------------------------------------------------
# Maximal Independent Set (greedy, degree ordering)
# ----------------------------------------------------------------------
def maximal_independent_set(
    graph: Graph,
    rng: Optional[np.random.Generator] = None,
) -> Set[Node]:
    """Return a maximal independent set using a degree‑aware greedy strategy."""
    if rng is None:
        rng = np.random.default_rng()
    # Nodes sorted by increasing degree (low degree first)
    nodes_by_degree = sorted(graph.keys(), key=lambda n: len(graph[n]))
    independent: Set[Node] = set()
    excluded: Set[Node] = set()
    for node in nodes_by_degree:
        if node in excluded:
            continue
        independent.add(node)
        # Exclude the node and all its neighbours from future selection
        excluded.add(node)
        excluded.update(graph[node])
    # Random tie‑breaking for nodes with equal degree
    # (shuffle the order before the greedy pass)
    rng.shuffle(nodes_by_degree)
    return independent


# ----------------------------------------------------------------------
# MinHash utilities (deterministic, seeded)
# ----------------------------------------------------------------------
def minhash(text: str, dim: int = 256, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return a binary MinHash sketch of ``text``.  Uses a deterministic RNG
    seeded from the text to keep sketches reproducible."""
    if rng is None:
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


# ----------------------------------------------------------------------
# Model tier / pool (unchanged but with dataclass)
# ----------------------------------------------------------------------
@dataclass
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


@dataclass
class ModelPool:
    ram_ceiling_mb: int = 6000
    loaded: Dict[str, ModelTier] = field(default_factory=dict)

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model respecting tier exclusivity and RAM ceiling."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict least‑recently‑added models until enough RAM is free."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # pop arbitrary (FIFO) entry
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)


# ----------------------------------------------------------------------
# Hybrid operation – deeper integration
# ----------------------------------------------------------------------
def hybrid_operation(
    graph: Graph,
    text: str,
    dim: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Combine graph topology with a MinHash sketch of ``text``.

    1. Compute a maximal independent set (MIS) of the graph.
    2. Derive a curvature scalar from the graph Laplacian.
    3. Produce a MinHash sketch of the text.
    4. Weight each entry of the sketch by the MIS membership proportion
       and the curvature, yielding a richer representation.
    """
    if rng is None:
        rng = np.random.default_rng()
    mis = maximal_independent_set(graph, rng)

    # Proportion of nodes that belong to the MIS (0..1)
    mis_ratio = len(mis) / max(1, len(graph))

    # Curvature factor (scaled to a reasonable range)
    curvature = graph_curvature(graph)
    curvature_factor = 1.0 + curvature  # ensure >1

    # Deterministic MinHash sketch
    sketch = minhash(text, dim, rng)

    # Final embedding: scale sketch by both factors
    weighted = sketch.astype(float) * mis_ratio * curvature_factor
    return weighted


# ----------------------------------------------------------------------
# Example driver (kept minimal for reproducibility)
# ----------------------------------------------------------------------
def _example_usage() -> None:
    rng = np.random.default_rng(42)

    # Synthetic numeric elements
    elements = [rng.random(10).tolist() for _ in range(30)]
    graph = build_graph(elements)

    text = "Example text for hybrid minhash‑graph integration."
    hybrid_vec = hybrid_operation(graph, text, dim=256, rng=rng)

    # Model pool demonstration
    pool = ModelPool()
    pool.load(ModelTier(name="small_model", ram_mb=1024, tier="T1"))
    # Attempt to load a conflicting T3 model (will raise if T2 present)
    pool.load_with_eviction(ModelTier(name="large_model", ram_mb=4000, tier="T3"))

    print("Hybrid vector (first 10 values):", hybrid_vec[:10])
    print("Loaded models:", list(pool.loaded.keys()))


if __name__ == "__main__":
    _example_usage()