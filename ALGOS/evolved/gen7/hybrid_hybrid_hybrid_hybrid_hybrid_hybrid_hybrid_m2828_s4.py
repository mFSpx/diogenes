# DARWIN HAMMER — match 2828, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4.py (gen6)
# born: 2026-05-29T23:46:09Z

"""
Hybrid Clifford‑Sheaf Module
============================

This module fuses the *geometric product* machinery from
``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_geomet_m2707_s0.py`` (Parent A)
with the *cellular sheaf* framework from
``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1887_s4.py`` (Parent B).

Mathematical bridge
-------------------
* In a Clifford algebra Cl(n,0) every basis blade can be represented as a
  binary mask of length ``n``.  Multiplication of blades is a linear map on
  the full multivector space ℝ^{2ⁿ}.
* For a sheaf edge ``(u, v)`` we need two linear restriction maps
  ``src_map`` and ``dst_map``.
* We construct these maps **exactly** as the left‑multiplication matrices
  induced by chosen blades.  Thus the algebraic structure of the Clifford
  product becomes the *restriction* structure of the sheaf, providing a
  mathematically coherent hybrid.

The three public helper functions below illustrate the hybrid operation:
``geometric_product`` (pure Clifford), ``blade_left_mul_matrix`` (matrix
representation of a blade acting on multivectors) and ``apply_hybrid_edge``,
which installs the Clifford‑derived restriction maps on a ``Sheaf`` instance.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Parent A – Clifford algebra helpers
# --------------------------------------------------------------------------- #

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti‑commutativity).  Duplicate indices cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst) - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)  # re‑check previous pair after swap
        elif lst[i] == lst[i + 1]:
            # e_i * e_i = 1 → remove both and keep sign
            del lst[i : i + 2]
            # stay at same index because list shrank
        else:
            i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Multiply two basis blades.

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def geometric_product(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Public wrapper for the Clifford geometric product."""
    return _multiply_blades(blade_a, blade_b)


def blade_to_index(blade: frozenset) -> int:
    """Map a blade to its integer index in the canonical basis of ℝ^{2ⁿ}.

    The index is the binary number whose i‑th bit is 1 iff e_i is present.
    """
    idx = 0
    for i in blade:
        idx |= 1 << i
    return idx


def index_to_blade(idx: int) -> frozenset:
    """Inverse of ``blade_to_index``."""
    blade = set()
    i = 0
    while idx:
        if idx & 1:
            blade.add(i)
        idx >>= 1
        i += 1
    return frozenset(blade)


def blade_left_mul_matrix(blade: frozenset, n: int) -> np.ndarray:
    """Matrix that left‑multiplies any multivector by ``blade`` in Cl(n,0).

    The multivector is represented as a column vector of length 2ⁿ ordered by
    increasing integer index (binary mask).
    """
    dim = 1 << n
    mat = np.zeros((dim, dim), dtype=float)
    for col_idx in range(dim):
        basis = index_to_blade(col_idx)
        prod_blade, sign = geometric_product(blade, basis)
        row_idx = blade_to_index(prod_blade)
        mat[row_idx, col_idx] = sign
    return mat


# --------------------------------------------------------------------------- #
# Parent B – Sheaf implementation (unchanged, only minor type hints added)
# --------------------------------------------------------------------------- #

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    Each node ``n`` carries a vector space ℝ^{dim(n)}.
    Each directed edge ``(u, v)`` carries a pair of linear restriction maps

        src_map : ℝ^{dim(u)} → ℝ^{dim(e)}
        dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}.

    A *section* is an assignment of a vector to every node that is compatible
    with all restriction maps, i.e. for every edge (u, v)

        src_map @ s[u]  ≈  dst_map @ s[v].

    The class provides utilities to set maps, set sections and to project
    a raw query vector onto the nearest compatible section.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    # --------------------------------------------------------------------- #
    # Restriction map handling
    # --------------------------------------------------------------------- #
    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if u not in self.node_dims or v not in self.node_dims:
            raise KeyError(f"Edge {edge} refers to undefined nodes.")
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[edge] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    # --------------------------------------------------------------------- #
    # Section handling
    # --------------------------------------------------------------------- #
    def set_section(self, node: Any, vector: np.ndarray) -> None:
        """Assign a vector to a node, checking dimensionality."""
        if node not in self.node_dims:
            raise KeyError(f"Node {node} undefined.")
        if vector.shape != (self.node_dims[node],):
            raise ValueError(f"Vector shape mismatch for node {node}.")
        self._sections[node] = np.asarray(vector, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        """Retrieve the current section vector for a node."""
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    # --------------------------------------------------------------------- #
    # Compatibility projection (least‑squares)
    # --------------------------------------------------------------------- #
    def project_onto_compatible(self) -> None:
        """
        Perform a single Gauss‑Seidel sweep updating each node's vector to best
        satisfy the edge restrictions in a least‑squares sense.
        """
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            su = self.get_section(u)
            sv = self.get_section(v)

            # Solve src_map @ x ≈ dst_map @ sv for x (the updated su)
            # Using the normal equations (src.T src) x = src.T dst sv
            A = src_map
            B = dst_map @ sv
            ATA = A @ A.T
            if np.linalg.cond(ATA) < 1e12:
                su_new = np.linalg.solve(ATA, A @ B)
                self._sections[u] = su_new
            # Symmetrically update v
            A = dst_map
            B = src_map @ su
            ATA = A @ A.T
            if np.linalg.cond(ATA) < 1e12:
                sv_new = np.linalg.solve(ATA, A @ B)
                self._sections[v] = sv_new

# --------------------------------------------------------------------------- #
# Hybrid utilities
# --------------------------------------------------------------------------- #

def apply_hybrid_edge(
    sheaf: Sheaf,
    edge: Tuple[Any, Any],
    blade_src: frozenset,
    blade_dst: frozenset,
    n: int,
) -> None:
    """
    Install Clifford‑derived restriction maps on *edge*.

    ``blade_src`` acts on the source node, ``blade_dst`` on the destination.
    Both are lifted to linear maps via ``blade_left_mul_matrix``.
    """
    src_dim = sheaf.node_dims[edge[0]]
    dst_dim = sheaf.node_dims[edge[1]]
    if src_dim != (1 << n) or dst_dim != (1 << n):
        raise ValueError("Node dimensions must match 2ⁿ for the chosen Clifford algebra.")

    src_map = blade_left_mul_matrix(blade_src, n)
    dst_map = blade_left_mul_matrix(blade_dst, n)

    # For simplicity we embed the matrices directly (row dimension equals dim)
    sheaf.set_restriction(edge, src_map, dst_map)


def assign_nearest_section(sheaf: Sheaf, query: np.ndarray) -> Any:
    """
    Given a query multivector (as a vector in ℝ^{2ⁿ}), return the node whose
    current section has the largest inner product with the query.
    """
    best_node = None
    best_score = -np.inf
    for node, vec in sheaf._sections.items():
        score = float(np.dot(vec, query))
        if score > best_score:
            best_score = score
            best_node = node
    return best_node


def hybrid_update_rule(sheaf: Sheaf, lr: float = 0.1) -> None:
    """
    Gradient‑like update that nudges each section toward compatibility
    using the Clifford restriction maps.

    For every edge (u, v) we compute the residual
        r = src_map @ s[u] - dst_map @ s[v]
    and move both sections opposite to the gradient of ‖r‖².
    """
    for (u, v), (src_map, dst_map) in sheaf._restrictions.items():
        su = sheaf.get_section(u)
        sv = sheaf.get_section(v)

        residual = src_map @ su - dst_map @ sv

        # Gradient w.r.t su is 2 * src_map.T @ residual
        grad_u = 2.0 * src_map.T @ residual
        # Gradient w.r.t sv is -2 * dst_map.T @ residual
        grad_v = -2.0 * dst_map.T @ residual

        sheaf.set_section(u, su - lr * grad_u)
        sheaf.set_section(v, sv - lr * grad_v)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Parameters
    n = 3                                 # Clifford algebra Cl(3,0)
    dim = 1 << n                          # 2ⁿ = 8
    node_dims = {"A": dim, "B": dim}
    edges = [("A", "B")]

    # Initialise sheaf
    sheaf = Sheaf(node_dims, edges)

    # Random blades for source and destination (choose non‑empty for interest)
    possible_indices = list(range(n))
    blade_src = frozenset(random.sample(possible_indices, k=random.randint(1, n)))
    blade_dst = frozenset(random.sample(possible_indices, k=random.randint(1, n)))

    # Install Clifford‑based restrictions
    apply_hybrid_edge(sheaf, ("A", "B"), blade_src, blade_dst, n)

    # Random sections
    sheaf.set_section("A", np.random.randn(dim))
    sheaf.set_section("B", np.random.randn(dim))

    # Random query vector
    query = np.random.randn(dim)

    # Demonstrate assignment
    nearest = assign_nearest_section(sheaf, query)
    print(f"Nearest node to query: {nearest}")

    # Perform a few hybrid updates
    for _ in range(5):
        hybrid_update_rule(sheaf, lr=0.05)

    # Project onto compatible sections (optional sanity check)
    sheaf.project_onto_compatible()

    # Verify that the restriction residuals are small
    src_map, dst_map = sheaf._restrictions[("A", "B")]
    residual = src_map @ sheaf.get_section("A") - dst_map @ sheaf.get_section("B")
    print(f"Final residual norm (should be small): {np.linalg.norm(residual):.6f}")

    # Show the blades used for the edge
    print(f"Edge ('A','B') uses source blade {sorted(blade_src)} and destination blade {sorted(blade_dst)}.")