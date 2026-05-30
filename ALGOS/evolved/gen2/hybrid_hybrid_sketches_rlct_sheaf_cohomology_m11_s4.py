# DARWIN HAMMER — match 11, survivor 4
# gen: 2
# parent_a: hybrid_sketches_rlct_grokking_m5_s0.py (gen1)
# parent_b: sheaf_cohomology.py (gen0)
# born: 2026-05-29T23:22:48Z

import hashlib
import random
import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = [
    "Sheaf",
    "count_min_sketch",
    "count_min_sheaf",
    "hybrid_rlct_via_sheaf",
    "hybrid_info_loss",
]

# ---------------------------------------------------------------------------
# Sheaf class (cellular sheaf over a simple directed graph)
# ---------------------------------------------------------------------------


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a linear restriction map
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and
      ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    # -----------------------------------------------------------------------
    # Restriction maps
    # -----------------------------------------------------------------------
    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    # -----------------------------------------------------------------------
    # Sections
    # -----------------------------------------------------------------------
    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Assign a vector to a node."""
        v = np.asarray(value, dtype=float).reshape(-1)
        if v.size != self.node_dims[node]:
            raise ValueError(f"Section size {v.size} does not match node dimension {self.node_dims[node]}")
        self._sections[node] = v

    # -----------------------------------------------------------------------
    # Layout helpers (compute offsets for block matrices)
    # -----------------------------------------------------------------------
    def _c0_layout(self) -> Tuple[List[Any], Dict[Any, int], int]:
        nodes = list(self.node_dims.keys())
        offsets: Dict[Any, int] = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _edge_dim(self, u: Any, v: Any) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c1_layout(self) -> Tuple[Dict[Tuple[Any, Any], Tuple[int, int]], int]:
        offsets: Dict[Tuple[Any, Any], Tuple[int, int]] = {}
        pos = 0
        for e in self.edges:
            d = self._edge_dim(*e)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos

    # -----------------------------------------------------------------------
    # Linear operators
    # -----------------------------------------------------------------------
    def coboundary_operator(self) -> np.ndarray:
        """Construct the coboundary matrix δ : C⁰ → C¹."""
        nodes, c0_off, c0_dim = self._c0_layout()
        c1_off, c1_dim = self._c1_layout()
        delta = np.zeros((c1_dim, c0_dim), dtype=float)

        for (u, v) in self.edges:
            row_start, d_e = c1_off[(u, v)]

            if (u, v) in self._restrictions:
                src_map, dst_map = self._restrictions[(u, v)]
            else:
                # reversed orientation stored
                dst_map, src_map = self._restrictions[(v, u)]

            col_u = c0_off[u]
            col_v = c0_off[v]
            dim_u = self.node_dims[u]
            dim_v = self.node_dims[v]

            delta[row_start : row_start + d_e, col_u : col_u + dim_u] -= src_map
            delta[row_start : row_start + d_e, col_v : col_v + dim_v] += dst_map

        return delta

    def consistency_residual(self) -> np.ndarray:
        """Compute δ(s) for the current node sections s."""
        nodes, c0_off, c0_dim = self._c0_layout()
        s = np.zeros(c0_dim, dtype=float)
        for n in nodes:
            if n in self._sections:
                off = c0_off[n]
                dim = self.node_dims[n]
                s[off : off + dim] = self._sections[n]
        delta = self.coboundary_operator()
        return delta @ s

    def global_inconsistency(self) -> float:
        """Return the squared ℓ₂‑norm of the coboundary residual."""
        r = self.consistency_residual()
        return float(np.dot(r, r))

    def laplacian(self) -> np.ndarray:
        """Sheaf Laplacian L = δᵀ δ."""
        delta = self.coboundary_operator()
        return delta.T @ delta


# ---------------------------------------------------------------------------
# Count‑Min sketch utilities
# ---------------------------------------------------------------------------


def _hash_bucket(depth_idx: int, item: Any, width: int) -> int:
    """Deterministic hash to a bucket index for a given depth."""
    h = hashlib.sha256(f"{depth_idx}:{item}".encode()).hexdigest()
    return int(h, 16) % width


def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> np.ndarray:
    """
    Build a Count‑Min sketch as a ``depth × width`` integer NumPy array.
    """
    table = np.zeros((depth, width), dtype=int)
    for item in items:
        for d in range(depth):
            idx = _hash_bucket(d, item, width)
            table[d, idx] += 1
    return table


# ---------------------------------------------------------------------------
# Hybrid construction utilities
# ---------------------------------------------------------------------------


def _hash_scale(u: Tuple[int, int], v: Tuple[int, int]) -> float:
    """
    Produce a deterministic scaling factor in (0.9, 1.1) from the ordered
    pair of node identifiers.  This replaces the ad‑hoc random scaling
    used in the original implementation and ties the restriction maps
    directly to the hash functions that generated the sketch.
    """
    h = hashlib.sha256(f"{u}-{v}".encode()).hexdigest()
    base = int(h, 16) % 200  # 0 … 199
    return 0.9 + (base / 1000.0)  # 0.9 … 1.099


def count_min_sheaf(items: List[Any], width: int = 64, depth: int = 4) -> Sheaf:
    """
    Convert a Count‑Min sketch into a cellular sheaf.

    * Each vertex corresponds to a pair ``(d, b)`` where ``d`` is the depth
      (hash function) and ``b`` is the bucket index.
    * Vertices are 1‑dimensional (the stored count).
    * Directed edges connect consecutive depths for the same bucket,
      i.e. ``(d, b) → (d+1, b)``.
    * Restriction maps are 1×1 scalar matrices equal to a deterministic
      hash‑derived scale factor, ensuring full rank while preserving the
      algebraic relationship between the two hash functions.
    """
    sketch = count_min_sketch(items, width, depth)

    # Node dimensions: every vertex carries a scalar.
    node_dims = {(d, b): 1 for d in range(depth) for b in range(width)}

    # Edges: connect (d, b) → (d+1, b) for all admissible depths.
    edges = [((d, b), (d + 1, b)) for d in range(depth - 1) for b in range(width)]

    sheaf = Sheaf(node_dims, edges)

    # Set restriction maps (scalar 1×1 matrices).
    for (u, v) in edges:
        scale = _hash_scale(u, v)
        src_map = np.array([[scale]], dtype=float)  # shape (1,1)
        dst_map = np.array([[scale]], dtype=float)  # shape (1,1)
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Populate sections with the sketch counts.
    for d in range(depth):
        for b in range(width):
            sheaf.set_section((d, b), np.array([sketch[d, b]], dtype=float))

    return sheaf


# ---------------------------------------------------------------------------
# Hybrid RLCT estimation
# ---------------------------------------------------------------------------


def _log_log_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform ordinary least‑squares linear regression on ``log(x)`` vs ``log(y)``.
    Returns the intercept and slope.
    """
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Regression inputs must be strictly positive.")
    X = np.log(x)
    Y = np.log(y)
    A = np.vstack([np.ones_like(X), X]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    intercept, slope = coeffs
    return float(intercept), float(slope)


def hybrid_rlct_via_sheaf(sheaf: Sheaf) -> float:
    """
    Estimate a Real Log‑Canonical Threshold (RLCT) from sheaf inconsistency.

    Procedure:
    1. Compute the absolute coboundary residuals ``r_i``.
    2. Sort them increasingly and treat the sorted values as ``ε_i``.
    3. Perform a log‑log regression of ``ε_i`` against the index ``i``.
    4. The RLCT is defined as ``-slope`` (the negative of the regression slope).

    The method mirrors the original RLCT estimation but grounds the
    regression in the sheaf‑theoretic residuals.
    """
    residuals = np.abs(sheaf.consistency_residual())
    if residuals.size == 0:
        raise ValueError("Sheaf has no edges; cannot compute RLCT.")
    # Avoid zero entries which would break the log.
    eps = 1e-12
    residuals = np.maximum(residuals, eps)

    sorted_res = np.sort(residuals)  # ε₁ ≤ ε₂ ≤ … ≤ ε_n
    indices = np.arange(1, sorted_res.size + 1, dtype=float)

    _, slope = _log_log_regression(indices, sorted_res)
    rlct = -slope
    return max(rlct, 0.0)  # RLCT is non‑negative by definition


# ---------------------------------------------------------------------------
# Hybrid information‑loss measure
# ---------------------------------------------------------------------------


def hybrid_info_loss(sheaf: Sheaf) -> float:
    """
    Compute a normalized information‑loss score that blends:

    * The RLCT estimate (capturing asymptotic decay of residuals).
    * The total sheaf Laplacian energy ``tr(L) = ||δ||_F²`` (global inconsistency).

    Both components are first normalised to the interval ``[0, 1]`` using the
    transformation ``x / (x + 1)`` and then multiplied, yielding a score that
    is small when either component indicates low loss and large when both
    indicate high loss.
    """
    rlct = hybrid_rlct_via_sheaf(sheaf)
    laplacian = sheaf.laplacian()
    energy = float(np.trace(laplacian))  # sum of eigenvalues = ||δ||_F²

    norm_rlct = rlct / (rlct + 1.0)
    norm_energy = energy / (energy + 1.0)

    return norm_rlct * norm_energy

# End of module.