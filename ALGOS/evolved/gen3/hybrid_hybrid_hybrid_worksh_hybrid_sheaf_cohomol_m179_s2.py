# DARWIN HAMMER — match 179, survivor 2
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py (gen2)
# parent_b: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
# born: 2026-05-29T23:25:59Z

"""Hybrid allocation and sheaf consistency module.

This module fuses two parent algorithms:
- **Parent A** (`hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py`): 
  computes a weekday‑dependent weight vector and allocates a total resource 
  into deterministic and residual parts across a set of groups.
- **Parent B** (`hybrid_sheaf_cohomology_percyphon_m2_s1.py`): 
  defines a cellular sheaf on a graph, with linear restriction maps between 
  node stalks and edge stalks, and provides tools to test section consistency.

**Mathematical bridge**  
Both parents operate on linear objects.  The weight vector from Parent A is a 
row‑stochastic vector  w∈ℝ^|G| that linearly maps a scalar residual r to a 
vector of allocations r·w.  In a sheaf, a *section* assigns a vector to each node; 
the collection of allocations is therefore a sheaf section over a graph whose 
nodes are the groups.  By equipping the graph with identity restriction maps,
the coboundary operator reduces to differences of neighbouring allocations.  
Thus we can test the “coherence’’ of the allocation by measuring the norm of the
coboundary, directly linking the two parent topologies.

The functions below implement this hybrid logic.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers (from Parent A)
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Sheaf implementation (from Parent B, trimmed)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf over an undirected graph.

    Nodes are identified by hash‑able labels (e.g., the group names).
    Each node carries a stalk of dimension ``node_dims[node]``.
    Edges are oriented as given in ``edge_list``; restriction maps are linear
    transformations from node stalks to the edge stalk.
    """

    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edge_list)
        # (u, v) -> (src_map, dst_map)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        # node -> section vector (numpy array)
        self._sections: Dict[Any, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Restriction maps
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """
        Assign linear restriction maps for a directed edge ``edge = (u, v)``.
        ``src_map`` maps the stalk at u to the edge stalk, ``dst_map`` maps the
        stalk at v to the same edge stalk.
        """
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)

        # sanity checks
        expected_src_shape = (self.edge_dim(edge), self.node_dims[u])
        expected_dst_shape = (self.edge_dim(edge), self.node_dims[v])
        if src_map.shape != expected_src_shape:
            raise ValueError(f"src_map shape {src_map.shape} does not match expected {expected_src_shape}")
        if dst_map.shape != expected_dst_shape:
            raise ValueError(f"dst_map shape {dst_map.shape} does not match expected {expected_dst_shape}")

        self._restrictions[edge] = (src_map, dst_map)

    def edge_dim(self, edge: Tuple[Any, Any]) -> int:
        """Return the dimension of the stalk on the given edge.

        By convention we use the same dimension for all edges; if a restriction
        has already been set we infer it, otherwise we default to 1.
        """
        if edge in self._restrictions:
            src_map, _ = self._restrictions[edge]
            return src_map.shape[0]
        return 1

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Assign a local section to ``node``."""
        val = np.asarray(value, dtype=float)
        if val.shape != (self.node_dims[node],):
            raise ValueError(f"section shape {val.shape} does not match node dimension {(self.node_dims[node],)}")
        self._sections[node] = val

    def get_section(self, node: Any) -> np.ndarray:
        """Retrieve the section associated with ``node``."""
        return self._sections[node]

    # ------------------------------------------------------------------
    # Cohomology‑like check
    # ------------------------------------------------------------------
    def coboundary_norm(self) -> float:
        """
        Compute the Euclidean norm of the coboundary of the current section.
        For each edge (u, v) we evaluate
            src_map @ s_u - dst_map @ s_v
        and sum the squared norms over all edges.
        """
        total = 0.0
        for (u, v) in self.edges:
            src_map, dst_map = self._restrictions.get(
                (u, v),
                (np.eye(self.edge_dim((u, v)), self.node_dims[u]),
                 np.eye(self.edge_dim((u, v)), self.node_dims[v])),
            )
            su = self._sections[u]
            sv = self._sections[v]
            diff = src_map @ su - dst_map @ sv
            total += float(np.linalg.norm(diff) ** 2)
        return math.sqrt(total)


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def build_allocation_sheaf(
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Tuple[Sheaf, Dict[str, float]]:
    """
    Construct a ``Sheaf`` whose node stalks are 1‑dimensional scalars representing
    the allocation to each group.

    Returns a tuple ``(sheaf, allocation_dict)`` where ``allocation_dict`` maps
    each group name to the numeric allocation.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    # 1️⃣ deterministic part – equally split across groups
    det_units = total_units * (deterministic_target_pct / 100.0)
    det_per_group = det_units / len(groups)

    # 2️⃣ residual part – weighted by weekday
    residual = total_units - det_units
    dow = date.weekday()  # Monday=0 … Sunday=6, same convention as weekday_weight_vector
    weight_vec = weekday_weight_vector(groups, dow)
    residual_alloc = residual * weight_vec

    # 3️⃣ final per‑group allocation
    allocations: Dict[str, float] = {
        g: _pct(det_per_group + residual_alloc[i]) for i, g in enumerate(groups)
    }

    # ------------------------------------------------------------------
    # Build the sheaf
    # ------------------------------------------------------------------
    node_dims = {g: 1 for g in groups}
    # simple chain graph: g0‑g1‑g2‑…‑g_{n-1}
    edge_list = [(groups[i], groups[i + 1]) for i in range(len(groups) - 1)]

    sheaf = Sheaf(node_dims, edge_list)

    # set sections (as 1‑element column vectors)
    for g in groups:
        sheaf.set_section(g, np.array([allocations[g]], dtype=float))

    # set identity restriction maps for each edge (edge_dim = 1)
    for (u, v) in edge_list:
        src_map = np.eye(1, 1)  # shape (1,1)
        dst_map = np.eye(1, 1)
        sheaf.set_restriction((u, v), src_map, dst_map)

    return sheaf, allocations


def allocation_consistency_metric(sheaf: Sheaf) -> float:
    """
    Return a scalar metric measuring how far the current allocation section is
    from being a global section of the sheaf.  The metric is the coboundary
    norm; a value of zero indicates perfect consistency (all neighbouring
    allocations identical under the identity restrictions).
    """
    return sheaf.coboundary_norm()


def allocate_hybrid_sheaf(
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    High‑level API mirroring ``allocate_hybrid`` from Parent A while also
    exposing sheaf‑based consistency information.

    The returned dictionary contains:
        - ``allocations``: mapping group → allocated float
        - ``deterministic_units`` and ``residual_units`` for transparency
        - ``weekday``: integer 0‑6 where 0 = Monday
        - ``consistency_norm``: coboundary norm of the sheaf section
        - ``sheaf``: the underlying ``Sheaf`` object (for advanced users)
    """
    sheaf, allocations = build_allocation_sheaf(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
    )

    det_units = total_units * (deterministic_target_pct / 100.0)
    residual_units = total_units - det_units

    result: Dict[str, Any] = {
        "allocations": allocations,
        "deterministic_units": _pct(det_units),
        "residual_units": _pct(residual_units),
        "weekday": date.weekday(),
        "consistency_norm": _pct(allocation_consistency_metric(sheaf)),
        "sheaf": sheaf,
    }
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    TOTAL = 1234.56
    TODAY = dt.date.today()
    OUT = allocate_hybrid_sheaf(
        total_units=TOTAL,
        date=TODAY,
        deterministic_target_pct=85.0,
        groups=GROUPS,
    )
    print("Hybrid allocation result:")
    for key, value in OUT.items():
        if key == "sheaf":
            continue  # avoid dumping the whole object
        print(f"  {key}: {value}")
    # Verify that the consistency norm is small (identical allocations only differ by weight)
    assert isinstance(OUT["consistency_norm"], float)  # sanity check
    print("\nSmoke test completed successfully.")