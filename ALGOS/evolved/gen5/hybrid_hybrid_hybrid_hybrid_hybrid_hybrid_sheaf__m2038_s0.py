# DARWIN HAMMER — match 2038, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m869_s1.py (gen4)
# parent_b: hybrid_hybrid_sheaf_cohomol_hybrid_liquid_time_c_m83_s0.py (gen2)
# born: 2026-05-29T23:40:31Z

"""
Hybrid Sheaf‑Bayesian Gradient (HSBG) algorithm.

Parents:
- hybrid_hybrid_hybrid_minimu_temporal_motifs_m280_s1 (Algorithm A)
- hybrid_hybrid_sheaf_cohomol_hybrid_liquid_time_c_m83_s0 (Algorithm B)

Mathematical bridge:
Algorithm A produces a Bayesian‑marginalised weight matrix **W** that encodes
prior knowledge about edge interactions.  Algorithm B builds a sheaf over a
directed graph, where each edge carries a pair of linear restriction maps.
In HSBG the Bayesian posterior is used to initialise (and periodically
re‑initialise) the restriction maps of the sheaf, turning the probabilistic
information into concrete linear operators.  The sheaf’s coboundary operator
provides a residual **r = δ(section)** that is fed to a simple gradient‑descent
step updating **W**.  Finally, MinHash signatures of the node sections are
computed to obtain a compact, time‑invariant representation of the evolving
state, enabling fast consistency checks across time steps.

The module therefore intertwines Bayesian marginalisation, sheaf cohomology,
gradient‑based matrix adaptation and MinHash hashing into a single unified
pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, List, Iterable, Sequence
import numpy as np
import hashlib
from datetime import datetime

# ----------------------------------------------------------------------
# Types from Parent A
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Types from Parent B
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

# ----------------------------------------------------------------------
# Sheaf definition (adapted from Parent B)
class Sheaf:
    """
    A sheaf over a directed graph.
    - node_dims: mapping node → dimension of the local vector space.
    - edges: list of (source, target) tuples.
    - _restrictions[(u,v)] = (src_map, dst_map) where src_map maps ℝ^{dim(u)} → ℝ^{k},
      dst_map maps ℝ^{dim(v)} → ℝ^{k}.  The integer k may differ per edge.
    - _sections[node] holds the current vector at the node.
    """
    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]]):
        self.node_dims: Dict[str, int] = dict(node_dims)
        self.edges: List[Tuple[str, str]] = list(edge_list)
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[str, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[str, str], src_map: Sequence[Sequence[float]], dst_map: Sequence[Sequence[float]]) -> None:
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: str, value: Sequence[float]) -> None:
        dim = self.node_dims.get(node)
        if dim is None:
            raise KeyError(f"Node {node} not defined in node_dims.")
        arr = np.array(value, dtype=float)
        if arr.shape[0] != dim:
            raise ValueError(f"Section dimension mismatch for node {node}: expected {dim}, got {arr.shape[0]}")
        self._sections[node] = arr

    def get_section(self, node: str) -> np.ndarray:
        return self._sections[node]

    def coboundary(self) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Compute the sheaf coboundary δ(section) for each edge.
        For edge (u,v):
            δ_e = dst_map @ s_v - src_map @ s_u
        Returns a dict edge → residual vector.
        """
        residuals: Dict[Tuple[str, str], np.ndarray] = {}
        for (u, v) in self.edges:
            if (u, v) not in self._restrictions:
                raise KeyError(f"Restriction maps not set for edge ({u},{v})")
            src_map, dst_map = self._restrictions[(u, v)]
            s_u = self.get_section(u)
            s_v = self.get_section(v)
            residual = dst_map @ s_v - src_map @ s_u
            residuals[(u, v)] = residual
        return residuals

# ----------------------------------------------------------------------
# Bayesian marginalisation utilities (from Parent A)
def bayesian_posterior(prior_counts: np.ndarray, observation_counts: np.ndarray) -> np.ndarray:
    """
    Compute a Dirichlet‑multinomial posterior.
    posterior = (α + n) / Σ(α + n)
    Returns a probability vector summing to 1.
    """
    if prior_counts.shape != observation_counts.shape:
        raise ValueError("Shape mismatch between prior and observations.")
    posterior = prior_counts + observation_counts
    total = posterior.sum()
    if total == 0:
        raise ValueError("Zero total count in posterior.")
    return posterior / total

def initialise_restriction_maps_from_posterior(sheaf: Sheaf, posterior: np.ndarray) -> None:
    """
    Map a posterior probability vector onto the restriction matrices of the sheaf.
    For each edge we create a low‑rank map whose singular values are proportional
    to the corresponding posterior entry.  This gives a deterministic yet
    probabilistically informed linear operator.
    """
    edges = sheaf.edges
    if len(edges) == 0:
        return
    # Normalise posterior to have the same number of edges (repeat if needed)
    probs = np.resize(posterior, len(edges))
    for idx, (u, v) in enumerate(edges):
        dim_u = sheaf.node_dims[u]
        dim_v = sheaf.node_dims[v]
        # Choose a small intermediate dimension k = 1 for simplicity
        k = 1
        # Build rank‑1 matrices: src_map = sqrt(p) * aᵀ, dst_map = sqrt(p) * bᵀ
        a = np.random.randn(dim_u, 1)
        b = np.random.randn(dim_v, 1)
        scale = math.sqrt(probs[idx])
        src_map = (scale * a.T)  # shape (1, dim_u)
        dst_map = (scale * b.T)  # shape (1, dim_v)
        sheaf.set_restriction((u, v), src_map, dst_map)

# ----------------------------------------------------------------------
# Gradient‑descent weight update (from Parent A)
def gradient_descent_step(W: np.ndarray, grad: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """Simple SGD update."""
    return W - lr * grad

def compute_weight_gradient(W: np.ndarray, residuals: Dict[Tuple[str, str], np.ndarray],
                            sheaf: Sheaf) -> np.ndarray:
    """
    Assemble a gradient for W based on coboundary residuals.
    For each edge we compute:
        g_e = src_map.T @ residual_e
    and sum over edges.
    """
    grad = np.zeros_like(W)
    for (u, v), resid in residuals.items():
        src_map, _ = sheaf._restrictions[(u, v)]
        # src_map shape (k, dim_u); resid shape (k,)
        contrib = src_map.T @ resid  # shape (dim_u,)
        # Assume W maps node‑wise sections to a global hidden vector;
        # we distribute the contribution to the rows corresponding to node u.
        start = sheaf._node_offset(u)
        end = start + sheaf.node_dims[u]
        grad[start:end, :] += np.outer(contrib, np.ones(W.shape[1]))
    return grad

# Helper to compute node offsets inside the global weight matrix
def _compute_node_offsets(node_dims: Dict[str, int]) -> Dict[str, int]:
    offsets: Dict[str, int] = {}
    pos = 0
    for n, d in node_dims.items():
        offsets[n] = pos
        pos += d
    return offsets

# Monkey‑patch the helper into Sheaf (keeps the public API tidy)
Sheaf._node_offset = lambda self, node: _compute_node_offsets(self.node_dims)[node]

# ----------------------------------------------------------------------
# MinHash utilities (from Parent B)
def minhash_vector(vec: np.ndarray, num_perm: int = 64) -> int:
    """
    Compute a simple MinHash signature for a numeric vector.
    For each permutation we hash the bytes of (seed || element) and keep the minimum.
    """
    min_hash = (1 << 64) - 1
    for seed in range(num_perm):
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(seed.to_bytes(4, byteorder="little"))
        hasher.update(vec.tobytes())
        h = int.from_bytes(hasher.digest(), byteorder="little")
        if h < min_hash:
            min_hash = h
    return min_hash

def sections_minhash(sheaf: Sheaf, num_perm: int = 64) -> Dict[str, int]:
    """
    Compute a MinHash signature for each node's section.
    Returns a mapping node → signature.
    """
    signatures: Dict[str, int] = {}
    for node, sec in sheaf._sections.items():
        signatures[node] = minhash_vector(sec, num_perm)
    return signatures

# ----------------------------------------------------------------------
# Core hybrid operation -------------------------------------------------
def hybrid_initialize(sheaf: Sheaf,
                     prior_counts: np.ndarray,
                     observation_counts: np.ndarray,
                     hidden_dim: int) -> np.ndarray:
    """
    Initialise the weight matrix W and the sheaf's restriction maps.
    - Bayesian posterior derived from prior and observations.
    - Restriction maps are seeded from the posterior.
    - W is initialised with small random values of shape (total_node_dim, hidden_dim).
    Returns the weight matrix W.
    """
    posterior = bayesian_posterior(prior_counts, observation_counts)
    initialise_restriction_maps_from_posterior(sheaf, posterior)
    total_dim = sum(sheaf.node_dims.values())
    rng = np.random.default_rng()
    W = rng.normal(loc=0.0, scale=0.1, size=(total_dim, hidden_dim))
    return W

def hybrid_step(sheaf: Sheaf,
                W: np.ndarray,
                hidden_state: np.ndarray,
                lr: float = 0.01) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray], Dict[str, int]]:
    """
    Perform one hybrid iteration:
    1. Compute the sheaf coboundary residuals δ(section).
    2. Propagate residuals through W to obtain a gradient.
    3. Update W via SGD.
    4. Produce MinHash signatures of the current sections.
    Returns the updated W, residuals, and MinHash signatures.
    """
    residuals = sheaf.coboundary()
    grad = compute_weight_gradient(W, residuals, sheaf)
    W_updated = gradient_descent_step(W, grad, lr)
    signatures = sections_minhash(sheaf)
    return W_updated, residuals, signatures

# ----------------------------------------------------------------------
# Smoke test ------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny directed graph with two nodes.
    node_dims = {"A": 3, "B": 2}
    edges = [("A", "B")]

    # Create sheaf instance.
    sheaf = Sheaf(node_dims, edges)

    # Random initial sections.
    rng = np.random.default_rng(42)
    sheaf.set_section("A", rng.normal(size=node_dims["A"]))
    sheaf.set_section("B", rng.normal(size=node_dims["B"]))

    # Prior and observation counts for Bayesian update (simple 2‑class example).
    prior = np.array([1.0, 1.0, 1.0, 1.0])          # length will be resized to edges
    observations = np.array([5, 3, 2, 0])

    # Initialise weight matrix.
    hidden_dim = 4
    W = hybrid_initialize(sheaf, prior, observations, hidden_dim)

    # Dummy hidden state (not used directly in this minimal example).
    hidden_state = rng.normal(size=hidden_dim)

    # Run a few hybrid steps.
    for step in range(3):
        W, residuals, signatures = hybrid_step(sheaf, W, hidden_state, lr=0.05)
        print(f"Step {step+1}:")
        for e, r in residuals.items():
            print(f"  Residual {e}: {r}")
        for n, sig in signatures.items():
            print(f"  MinHash {n}: {sig:#018x}")
        print("-" * 40)
    sys.exit(0)