# DARWIN HAMMER — match 1368, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s2.py (gen2)
# parent_b: hybrid_physarum_network_hybrid_hybrid_hdc_hy_m683_s2.py (gen5)
# born: 2026-05-29T23:35:36Z

"""Hybrid Sketch–Sheaf Physarum Module.

Parents:
- hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s2.py (sketch → sheaf → RLCT)
- hybrid_physarum_network_hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m683_s2.py
  (physarum flux‑based conductance updates + hyperdimensional bind)

Mathematical Bridge:
The Count‑Min sketch matrix is interpreted as a cellular sheaf:
each hash bucket is a vertex, each sketch depth is a local vector‑space dimension,
and edges connect consecutive buckets.  The coboundary δ(s) (difference of
adjacent vertex vectors) yields a sheaf Laplacian energy that quantifies
information distortion.  Node‑wise vector norms are treated as “pressures” in a
physarum network; edge fluxes computed from these pressures drive conductance
updates exactly as in the original physarum model.  Conductance values, in turn,
scale the restriction maps (identity up to a scalar) and are bound to
hyperdimensional symbol vectors, providing a unified representation that
captures both combinatorial sketch loss and adaptive transport dynamics.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Hyperdimensional utilities (from parent B)
# ---------------------------------------------------------------------------

def random_vector(dim: int = 10000, seed: int | str | None = None) -> List[int]:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:8], "big")
    return random_vector(dim, seed)

def bind(a: List[int], b: List[int]) -> List[int]:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[List[int]]) -> List[int]:
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if s >= 0 else -1 for s in sums]

# ---------------------------------------------------------------------------
# Physarum primitives (from parent B)
# ---------------------------------------------------------------------------

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ---------------------------------------------------------------------------
# Sheaf representation (core of parent A)
# ---------------------------------------------------------------------------

class Sheaf:
    """Cellular sheaf built from a Count‑Min sketch.

    Attributes
    ----------
    node_vectors : np.ndarray, shape (num_nodes, dim)
        The local sections (sketch counts) at each vertex.
    edges : List[Tuple[int, int]]
        Directed edges connecting vertices; we use a simple chain topology.
    """

    def __init__(self, node_vectors: np.ndarray, edges: List[Tuple[int, int]]):
        self.node_vectors = node_vectors.astype(float)  # shape (N, d)
        self.edges = edges

    def coboundary_residuals(self) -> np.ndarray:
        """Return δ(s) for each edge as a (num_edges, dim) array."""
        res = []
        for u, v in self.edges:
            res.append(self.node_vectors[u] - self.node_vectors[v])
        return np.stack(res) if res else np.empty((0, self.node_vectors.shape[1]))

    def laplacian_energy(self) -> float:
        """Sum of squared ℓ₂‑norms of coboundary residuals."""
        r = self.coboundary_residuals()
        return float(np.sum(np.linalg.norm(r, axis=1) ** 2))

# ---------------------------------------------------------------------------
# Hybrid construction utilities
# ---------------------------------------------------------------------------

def count_min_sketch(data: List[int], width: int, depth: int, seed: int = 0) -> np.ndarray:
    """Build a simple Count‑Min sketch matrix (width × depth)."""
    rng = np.random.default_rng(seed)
    # Random hash functions simulated by rng integers
    hash_params = rng.integers(1, 2 ** 31 - 1, size=(depth, 2), dtype=np.int64)

    sketch = np.zeros((width, depth), dtype=np.int64)

    for x in data:
        for d in range(depth):
            a, b = hash_params[d]
            idx = (a * x + b) % width
            sketch[idx, d] += 1
    return sketch

def count_min_sheaf(data: List[int], width: int, depth: int) -> Tuple[Sheaf, np.ndarray]:
    """Construct a Sheaf from a Count‑Min sketch.

    Returns
    -------
    sheaf : Sheaf
        Nodes correspond to hash buckets; edges connect consecutive buckets.
    sketch : np.ndarray
        The raw sketch matrix (width × depth).
    """
    sketch = count_min_sketch(data, width, depth)
    # Each bucket becomes a node; the local vector is the depth‑dimensional count column.
    node_vectors = sketch.astype(float)  # shape (width, depth)
    edges = [(i, i + 1) for i in range(width - 1)]  # simple chain
    sheaf = Sheaf(node_vectors, edges)
    return sheaf, sketch

def hybrid_rlct_via_sheaf(sheaf: Sheaf) -> float:
    """Estimate an RLCT from sheaf coboundary residual magnitudes.

    The method performs a log‑log linear regression between edge‑wise
    residual norms and the corresponding summed node counts (as a proxy for
    sample size).  The slope (negative) is the RLCT estimate.
    """
    residuals = sheaf.coboundary_residuals()
    if residuals.size == 0:
        return 0.0

    # Edge‑wise residual norm
    r_norm = np.linalg.norm(residuals, axis=1) + 1e-12

    # Edge‑wise “sample size”: sum of counts of its two incident nodes
    sample_sizes = []
    for u, v in sheaf.edges:
        size = np.sum(sheaf.node_vectors[u]) + np.sum(sheaf.node_vectors[v])
        sample_sizes.append(size + 1e-12)
    sample_sizes = np.array(sample_sizes)

    log_r = np.log(r_norm)
    log_n = np.log(sample_sizes)

    # Linear regression: log_r = a * log_n + b  →  RLCT = -a
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    slope, _ = np.linalg.lstsq(A, log_r, rcond=None)[0]
    rlct = -slope
    return float(rlct)

def init_conductances(sheaf: Sheaf, base: float = 1.0) -> Dict[Tuple[int, int], float]:
    """Create an initial conductance dictionary for the sheaf edges."""
    rng = random.Random(42)
    conductances = {}
    for e in sheaf.edges:
        conductances[e] = base * (0.5 + rng.random())  # random positive conductance
    return conductances

def physarum_sheaf_step(
    sheaf: Sheaf,
    conductances: Dict[Tuple[int, int], float],
    dt: float = 1.0,
    gain: float = 1.0,
    decay: float = 0.05,
) -> Dict[Tuple[int, int], float]:
    """One physarum update step driven by sheaf node pressures.

    Node pressure = ℓ₂ norm of its local vector.  Flux along each edge updates the
    edge conductance via the classic physarum rule.
    """
    pressures = np.linalg.norm(sheaf.node_vectors, axis=1)
    edge_lengths = {e: 1.0 for e in sheaf.edges}  # unit length for simplicity

    new_conductances = {}
    for (u, v) in sheaf.edges:
        g = conductances[(u, v)]
        q = flux(g, edge_lengths[(u, v)], pressures[u], pressures[v])
        new_conductances[(u, v)] = update_conductance(g, q, dt=dt, gain=gain, decay=decay)
    return new_conductances

def hybrid_info_loss(sheaf: Sheaf, conductances: Dict[Tuple[int, int], float]) -> float:
    """Blend sheaf Laplacian energy with physarum conductance energy.

    Returns a normalized scalar in [0, 1] where larger values indicate higher
    information loss.
    """
    lap_energy = sheaf.laplacian_energy()
    conduct_energy = sum(g ** 2 for g in conductances.values())

    # Normalization by simple additive constant to avoid division by zero
    norm = lap_energy + conduct_energy + 1e-12
    return float((lap_energy + conduct_energy) / norm)

def hybrid_bind_representation(sheaf: Sheaf, conductances: Dict[Tuple[int, int], float], dim: int = 1000) -> List[int]:
    """Create a hyperdimensional representation of the hybrid state.

    For each node we generate a symbol vector from its index, bind it with a
    binary encoding of the incident conductances, and finally bundle all bound
    vectors.
    """
    node_vectors = []
    for node in range(sheaf.node_vectors.shape[0]):
        sym = symbol_vector(f"node_{node}", dim)
        # Encode incident conductances as a binary hypervector (threshold at median)
        incident = [conductances[(node, node + 1)] for (u, v) in sheaf.edges if u == node] + \
                   [conductances[(node - 1, node)] for (u, v) in sheaf.edges if v == node]
        if incident:
            median = np.median(incident)
            bin_vec = [1 if c >= median else -1 for c in incident]
            # Pad / repeat to match dimension
            if len(bin_vec) < dim:
                repeats = (dim + len(bin_vec) - 1) // len(bin_vec)
                bin_vec = (bin_vec * repeats)[:dim]
            else:
                bin_vec = bin_vec[:dim]
            bound = bind(sym, bin_vec)
        else:
            bound = sym  # isolated node
        node_vectors.append(bound)

    return bundle(node_vectors)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic data
    random.seed(0)
    data = [random.randint(0, 1000) for _ in range(500)]

    width, depth = 12, 6
    sheaf, sketch = count_min_sheaf(data, width, depth)

    rlct = hybrid_rlct_via_sheaf(sheaf)
    print(f"RLCT estimate: {rlct:.4f}")

    conductances = init_conductances(sheaf)
    for step in range(5):
        conductances = physarum_sheaf_step(sheaf, conductances, dt=0.5)
        loss = hybrid_info_loss(sheaf, conductances)
        print(f"Step {step+1}, info loss: {loss:.4f}")

    hd_rep = hybrid_bind_representation(sheaf, conductances, dim=1024)
    print(f"Hybrid hyperdimensional representation length: {len(hd_rep)}")
    print(f"First 10 bits: {hd_rep[:10]}")