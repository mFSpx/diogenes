# DARWIN HAMMER — match 23, survivor 2
# gen: 2
# parent_a: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
# parent_b: hybrid_ternary_lens_audit_decreasing_pruning_m15_s0.py (gen1)
# born: 2026-05-29T23:22:53Z

"""Hybrid sheaf‑cohomology & ternary‑lens pruning algorithm.

Parents:
- **hybrid_sheaf_cohomology_percyphon_m2_s1.py** – defines a cellular sheaf,
  its stalk dimensions, restriction maps and the coboundary matrix Δ.
- **hybrid_ternary_lens_audit_decreasing_pruning_m15_s0.py** – defines a
  decreasing‑exponential pruning probability `p(t)=λ·exp(-α·t)` and a random
  pruning step applied to a list of candidates.

Mathematical bridge:
Both modules manipulate linear objects.  The sheaf’s restriction maps are
represented by matrices; the pruning algorithm supplies a scalar probability
that can be used to *randomly discard* some of those matrices before the
coboundary operator is assembled.  By interpreting each vendor entry from a
manifest as a node (dimension 1) and each edge’s restriction map as a scalar
derived from a hash, we obtain a concrete sheaf.  The pruning step removes a
fraction of edges according to `p(t)`, yielding a *pruned coboundary matrix*.
Subsequent linear‑algebraic analysis (e.g. null‑space computation) therefore
reflects both the topological sheaf structure and the stochastic pruning
policy.

The module provides three high‑level functions that showcase the hybrid
behaviour:
1. `build_sheaf_from_manifest` – constructs a Sheaf from a vendor manifest.
2. `prune_sheaf_edges` – removes edges with probability `p(t)`.
3. `sheaf_nullspace_dimension` – computes the dimension of the kernel of the
   (pruned) coboundary matrix.

A small smoke test demonstrates end‑to‑end execution without external data.
"""

import json
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Core sheaf implementation (adapted from parent A)
# ----------------------------------------------------------------------


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


class Sheaf:
    """Cellular sheaf over a graph with 1‑dimensional stalks per node."""

    def __init__(self, node_dims: dict[Any, int], edge_list: Sequence[tuple[Any, Any]]):
        self.node_dims = dict(node_dims)               # node_id → dim (here always 1)
        self.edges = list(edge_list)                  # list of (u, v)
        self._restrictions: dict[tuple[Any, Any], tuple[np.ndarray, np.ndarray]] = {}
        self._sections: dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Assign linear restriction maps for an oriented edge."""
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: Any, value: Sequence[float]) -> None:
        self._sections[node] = np.asarray(value, dtype=float)

    def _edge_dim(self, u: Any, v: Any) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _c1_layout(self):
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos

    def coboundary_operator(self) -> np.ndarray:
        """Construct the full coboundary matrix Δ."""
        nodes, c0_off, c0_dim = self._c0_layout()
        c1_off, c1_dim = self._c1_layout()
        delta = np.zeros((c1_dim, c0_dim), dtype=float)

        for u, v in self.edges:
            row_start, d_e = c1_off[(u, v)]

            if (u, v) in self._restrictions:
                src_map, dst_map = self._restrictions[(u, v)]
            else:
                dst_map, src_map = self._restrictions[(v, u)]

            col_u = c0_off[u]
            col_v = c0_off[v]
            dim_u = self.node_dims[u]
            dim_v = self.node_dims[v]

            delta[row_start:row_start + d_e, col_u:col_u + dim_u] -= src_map
            delta[row_start:row_start + d_e, col_v:col_v + dim_v] += dst_map

        return delta


# ----------------------------------------------------------------------
# Pruning utilities (adapted from parent B)
# ----------------------------------------------------------------------


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential decreasing pruning probability."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam and alpha must be non‑negative")
    return min(1.0, lam * math.exp(-alpha * t))


def prune_edges(edges: list[tuple[Any, Any]],
                t: float,
                lam: float = 1.0,
                alpha: float = 0.2,
                seed: int | str | None = None) -> list[tuple[Any, Any]]:
    """Randomly discard edges according to the pruning probability."""
    rng = random.Random(seed)
    p = prune_probability(t, lam, alpha)
    return [e for e in edges if rng.random() >= p]


# ----------------------------------------------------------------------
# Hybrid construction helpers
# ----------------------------------------------------------------------


def _scalar_from_hash(text: str) -> float:
    """Deterministic pseudo‑random scalar in [-1, 1] derived from SHA‑256."""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Use first 8 bytes as a 64‑bit unsigned integer
    val = int.from_bytes(h[:8], "big")
    # Map to [-1, 1]
    return (val / (2**64 - 1)) * 2.0 - 1.0


def build_sheaf_from_manifest(manifest: dict[str, Any]) -> Sheaf:
    """
    Translate a vendor manifest into a Sheaf.

    * Each vendor entry becomes a node with dimension 1.
    * Edges are created between successive nodes (i, i+1).
    * Restriction maps are 1×1 scalars derived from the hash of the two node keys.
    """
    vendors = manifest.get("vendors", [])
    if not vendors:
        raise ValueError("Manifest must contain a non‑empty 'vendors' list")

    node_ids = [v.get("candidate_key", f"node_{i}") for i, v in enumerate(vendors)]
    node_dims = {nid: 1 for nid in node_ids}
    edges = [(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1)]

    sheaf = Sheaf(node_dims, edges)

    for u, v in edges:
        # Create deterministic scalars for the two directions
        src_scalar = _scalar_from_hash(f"{u}->{v}")
        dst_scalar = _scalar_from_hash(f"{v}->{u}")
        src_map = np.array([[src_scalar]])   # shape (1,1)
        dst_map = np.array([[dst_scalar]])   # shape (1,1)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Optional: set a trivial section (all ones) for each node
    for nid in node_ids:
        sheaf.set_section(nid, [1.0])

    return sheaf


def prune_sheaf_edges(sheaf: Sheaf,
                      t: float,
                      lam: float = 1.0,
                      alpha: float = 0.2,
                      seed: int | str | None = None) -> Sheaf:
    """
    Return a new Sheaf where a subset of edges has been removed according
    to the pruning schedule.
    """
    kept_edges = prune_edges(sheaf.edges, t, lam, alpha, seed)
    pruned_sheaf = Sheaf(sheaf.node_dims, kept_edges)

    # Preserve restriction maps only for kept edges
    for edge in kept_edges:
        if edge in sheaf._restrictions:
            src_map, dst_map = sheaf._restrictions[edge]
        else:
            # Edge may have been stored with opposite orientation
            src_map, dst_map = sheaf._restrictions[(edge[1], edge[0])]
            # Swap orientation to match the new edge direction
            src_map, dst_map = dst_map, src_map
        pruned_sheaf.set_restriction(edge, src_map, dst_map)

    # Copy sections (they are node‑wise and unchanged)
    for node, sec in sheaf._sections.items():
        pruned_sheaf.set_section(node, sec)

    return pruned_sheaf


def sheaf_nullspace_dimension(sheaf: Sheaf, tol: float = 1e-10) -> int:
    """
    Compute the dimension of the kernel of the coboundary matrix Δ of the given sheaf.
    """
    delta = sheaf.coboundary_operator()
    u, s, vh = np.linalg.svd(delta, full_matrices=True)
    null_mask = s <= tol
    null_dim = np.sum(null_mask)
    return int(null_dim)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


def _dummy_manifest() -> dict[str, Any]:
    """Create a minimal manifest with a few synthetic vendor entries."""
    return {
        "vendors": [
            {
                "candidate_key": "vendor_alpha",
                "classification": "usable_now",
                "fast_path_compatible": True,
                "benchmark_required": False,
                "notes": "",
            },
            {
                "candidate_key": "vendor_beta",
                "classification": "needs_conversion",
                "fast_path_compatible": False,
                "benchmark_required": True,
                "notes": "requires conversion",
            },
            {
                "candidate_key": "vendor_gamma",
                "classification": "research_only",
                "fast_path_compatible": False,
                "benchmark_required": False,
                "notes": "",
            },
        ]
    }


if __name__ == "__main__":
    # Build sheaf from a dummy manifest
    manifest = _dummy_manifest()
    sheaf = build_sheaf_from_manifest(manifest)

    # Show original coboundary shape
    delta_orig = sheaf.coboundary_operator()
    print("Original coboundary shape:", delta_orig.shape)

    # Prune edges with t=2.0 (moderate pruning)
    pruned_sheaf = prune_sheaf_edges(sheaf, t=2.0, lam=1.0, alpha=0.3, seed=42)

    # Show pruned coboundary shape
    delta_pruned = pruned_sheaf.coboundary_operator()
    print("Pruned coboundary shape:", delta_pruned.shape)

    # Compute nullspace dimensions
    null_dim_orig = sheaf_nullspace_dimension(sheaf)
    null_dim_pruned = sheaf_nullspace_dimension(pruned_sheaf)
    print("Nullspace dimension (original):", null_dim_orig)
    print("Nullspace dimension (pruned):", null_dim_pruned)

    # Exit cleanly
    sys.exit(0)