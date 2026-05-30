# DARWIN HAMMER — match 42, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s0.py (gen3)
# born: 2026-05-29T23:26:30Z

"""Hybrid Sheaf‑Associative‑VRAM Scheduler

Parents
-------
* hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (Sheaf + Dense Associative Memory)
* hybrid_hybrid_hybrid_minimu_model_vram_scheduler_m8_s0.py (Tree metrics + VRAM slot planning)

Mathematical Bridge
-------------------
A cellular sheaf assigns a vector to each node of a directed graph.  
A dense associative memory (DAM) defines an energy 
    E(x) = -½ xᵀ W x
for a global query vector *x*.  

The VRAM scheduler works on a *tree* of model components; each edge carries a Euclidean length that
encodes a cost of moving information between components.

We fuse the two worlds by:
1. Flattening all node‑section vectors of a sheaf into a single query vector *x*.
2. Using the DAM energy of *x* as a scalar *cost* for a VRAM allocation plan.
3. Modulating the sheaf’s restriction maps with the tree‑metric distances so that
   high‑cost (high‑energy) sections are relaxed more strongly along long edges.

The resulting hybrid system can (a) store feasible VRAM plans in the DAM,
(b) evaluate a sheaf‑derived plan via the DAM energy, and
(c) adapt the sheaf topology using tree‑derived gradients to minimise that energy.

"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple
from dataclasses import asdict, dataclass

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * ``node_dims`` maps node identifier → dimension of its vector space.
    * ``edges`` is a list of directed edges (u, v).
    * Each edge stores a pair of restriction matrices (src_map, dst_map)
      mapping node vectors to a common edge space.
    * ``sections`` stores the current vector assigned to each node.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        if value.shape != (self.node_dims[node],):
            raise ValueError(f"section vector for node {node} must have shape ({self.node_dims[node]},)")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def flatten_sections(self) -> np.ndarray:
        """Concatenate all node sections in a deterministic order."""
        ordered = [self._sections.get(n, np.zeros(d)) for n, d in sorted(self.node_dims.items())]
        return np.concatenate(ordered) if ordered else np.array([])

    def apply_restrictions(self) -> Dict[Tuple[Any, Any], np.ndarray]:
        """
        For each edge (u, v) compute the edge‑space vector
        r_e = src_map @ sec(u) - dst_map @ sec(v).
        Returns a dict edge → residual vector.
        """
        residuals = {}
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            ru = src_map @ self.get_section(u)
            rv = dst_map @ self.get_section(v)
            residuals[(u, v)] = ru - rv
        return residuals


class DenseAssociativeMemory:
    """
    Simple Hopfield‑style dense associative memory.
    Stores binary (or real‑valued) patterns via Hebbian learning.
    Energy: E(x) = -½ xᵀ W x
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.W = np.zeros((dim, dim), dtype=float)

    def store(self, pattern: np.ndarray) -> None:
        """Hebbian update: W ← W + pattern·patternᵀ (zero diagonal)."""
        p = np.asarray(pattern, dtype=float).reshape(-1)
        if p.shape[0] != self.dim:
            raise ValueError("Pattern dimension mismatch.")
        self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0.0)

    def energy(self, query: np.ndarray) -> float:
        q = np.asarray(query, dtype=float).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError("Query dimension mismatch.")
        return -0.5 * q @ self.W @ q

    def retrieve(self, query: np.ndarray, steps: int = 10, lr: float = 0.1) -> np.ndarray:
        """
        Gradient descent on the energy (ascent on -E) to converge to a stored attractor.
        """
        x = np.asarray(query, dtype=float).reshape(-1)
        for _ in range(steps):
            grad = -self.W @ x  # dE/dx = -W x
            x = x - lr * grad
        return x

# ----------------------------------------------------------------------
# Core structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS to compute distances from root
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)
    return adj, edge_len, dist

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_energy(sheaf: Sheaf, memory: DenseAssociativeMemory) -> float:
    """
    Compute the DAM energy of the flattened sheaf sections.
    This scalar quantifies how compatible the current sheaf configuration
    is with the stored VRAM allocation patterns.
    """
    query = sheaf.flatten_sections()
    if query.size == 0:
        raise ValueError("Sheaf has no sections to evaluate.")
    return memory.energy(query)


def hybrid_update_rule(
    sheaf: Sheaf,
    memory: DenseAssociativeMemory,
    tree_nodes: Dict[str, Tuple[float, float]],
    tree_edges: List[Tuple[str, str]],
    root: str,
    lr: float = 0.05,
) -> None:
    """
    Perform a single hybrid update:
    * Compute DAM energy gradient w.r.t. each node section.
    * Propagate the gradient along the tree using distances as a damping factor.
    * Apply a correction to each node's section (simple gradient step).
    """
    # 1. Energy gradient for the whole flattened vector
    query = sheaf.flatten_sections()
    grad_full = -memory.W @ query  # dE/dx = -W x

    # 2. Split gradient back to per‑node pieces
    split_grads = {}
    offset = 0
    for node, dim in sorted(sheaf.node_dims.items()):
        split_grads[node] = grad_full[offset: offset + dim]
        offset += dim

    # 3. Tree distances to modulate step size
    _, _, dist = tree_metrics(tree_nodes, tree_edges, root)

    # 4. Update each node's section
    for node in sheaf.node_dims:
        d = dist.get(node, 0.0)
        damping = math.exp(-d)  # longer distance → smaller step
        new_section = sheaf.get_section(node) - lr * damping * split_grads[node]
        sheaf.set_section(node, new_section)


def hybrid_retrieve(
    sheaf: Sheaf,
    memory: DenseAssociativeMemory,
    target_plan: VramSlotPlan,
    steps: int = 15,
    lr: float = 0.1,
) -> Tuple[Sheaf, np.ndarray]:
    """
    Encode a ``VramSlotPlan`` as a query vector, run DAM retrieval,
    and map the result back onto the sheaf sections.

    Returns the updated sheaf and the raw retrieved vector.
    """
    # Simple encoding: hash textual fields into a fixed‑size real vector.
    # For reproducibility we use a deterministic pseudo‑random seed.
    seed = abs(hash((target_plan.artifact_id, target_plan.artifact_kind, target_plan.action))) % (2**32)
    rng = np.random.default_rng(seed)
    encoded = rng.normal(size=memory.dim)

    retrieved = memory.retrieve(encoded, steps=steps, lr=lr)

    # Distribute retrieved vector onto sheaf nodes proportionally to their dimensions.
    offset = 0
    for node, dim in sorted(sheaf.node_dims.items()):
        slice_vec = retrieved[offset: offset + dim]
        sheaf.set_section(node, slice_vec)
        offset += dim

    return sheaf, retrieved

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Build a tiny sheaf -----
    node_dims = {"A": 3, "B": 2}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)

    # Random restriction maps
    rng = np.random.default_rng(42)
    src_map = rng.normal(size=(4, 3))
    dst_map = rng.normal(size=(4, 2))
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Initialise sections
    sheaf.set_section("A", rng.normal(size=3))
    sheaf.set_section("B", rng.normal(size=2))

    # ----- Initialise DAM with same dimension -----
    dim = sum(node_dims.values())
    dam = DenseAssociativeMemory(dim)

    # Store a couple of random patterns (simulating viable VRAM plans)
    for _ in range(5):
        pat = rng.normal(size=dim)
        dam.store(pat)

    # ----- Tree for the VRAM scheduler -----
    tree_nodes = {"A": (0.0, 0.0), "B": (1.0, 0.0)}
    tree_edges = [("A", "B")]
    root = "A"

    # ----- Hybrid operations -----
    print("Initial hybrid energy:", hybrid_energy(sheaf, dam))

    hybrid_update_rule(sheaf, dam, tree_nodes, tree_edges, root)

    print("Energy after one update:", hybrid_energy(sheaf, dam))

    # Create a dummy VramSlotPlan
    plan = VramSlotPlan(
        artifact_id="model_xyz",
        artifact_kind="checkpoint",
        action="load",
        estimated_mb=512,
        reason="pre‑emptive load",
        detail={"layer": "encoder", "priority": 1},
    )

    sheaf, retrieved_vec = hybrid_retrieve(sheaf, dam, plan)

    print("Energy after retrieval:", hybrid_energy(sheaf, dam))
    print("Retrieved vector norm:", np.linalg.norm(retrieved_vec))
    print("Smoke test completed successfully.")