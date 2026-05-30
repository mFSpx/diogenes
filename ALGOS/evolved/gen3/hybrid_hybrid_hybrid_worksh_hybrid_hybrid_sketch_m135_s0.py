# DARWIN HAMMER — match 135, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py (gen2)
# parent_b: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s3.py (gen2)
# born: 2026-05-29T23:25:48Z

"""Hybrid Allocation‑Sheaf Fusion

This module combines the *weekday‑weighted allocation* logic from
`hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s4.py` (Parent A)
with the *sheaf cohomology* machinery from
`hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s3.py` (Parent B).

Mathematical bridge
------------------
- The allocation routine produces a scalar value for each *group* (e.g. "codex",
  "groq", …).  These scalars naturally form a **0‑cochain** (a section over the
  0‑simplices) of a sheaf whose nodes are the groups.
- By constructing a sheaf whose edges encode pairwise relationships between
  groups, the coboundary operator `δ : C⁰ → C¹` maps the allocation vector to a
  set of *differences* (residuals) along edges.
- The hybrid algorithm therefore allocates resources, builds the corresponding
  sheaf, applies the coboundary operator, and evaluates the consistency
  residual ‖δ s‖₂, providing a mathematically unified view of resource sharing
  and topological consistency.

The public API consists of three core functions:
1. ``weekday_weight_vector`` – builds the weekday‑dependent weight vector.
2. ``allocate_hybrid`` – performs the deterministic/LLM split and returns a
   per‑group allocation.
3. ``sheaf_residual_from_allocation`` – builds a sheaf from the allocation,
   computes the coboundary matrix, applies it to the allocation section, and
   returns the L2 norm of the resulting residual vector.

A minimal smoke test is provided under ``if __name__ == "__main__"``, which
executes the full pipeline without external dependencies.
"""

import datetime as dt
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – allocation utilities
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places (consistent with Parent A)."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return a 0‑based weekday index (Monday=0 … Sunday=6) using the Doomsday rule."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector whose entries vary sinusoidally with the
    day of week.  The formula mirrors Parent A.
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


def allocate_hybrid(
    *,
    total_units: float,
    date: dt.date,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> Dict[str, Any]:
    """
    Split ``total_units`` into deterministic and LLM portions, then distribute
    the LLM portion across ``groups`` using a weekday‑dependent weight vector.
    Returns a dictionary containing the per‑group allocation and meta‑data.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units

    dow = doomsday(date.year, date.month, date.day)
    weight_vec = weekday_weight_vector(groups, dow)

    llm_per_group = llm_units * weight_vec
    share_pct_per_group = weight_vec * 100.0

    lanes = [
        {
            "group": grp,
            "llm_units": _pct(llm_per_group[i]),
            "llm_share_pct": _pct(share_pct_per_group[i]),
            "weekday_weight": _pct(weight_vec[i]),
        }
        for i, grp in enumerate(groups)
    ]

    result: Dict[str, Any] = {
        "date": date.isoformat(),
        "total_units": total_units,
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_residual_pct": _pct(100.0 - deterministic_target_pct),
        "llm_units_total": _pct(llm_units),
        "allocation": lanes,
    }
    return result


# ----------------------------------------------------------------------
# Parent B – sheaf/cohomology utilities (trimmed to essentials)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf implementation supporting:

    * Node dimensions (here always 1 for scalar allocations).
    * Edge list with linear restriction maps.
    * Construction of the coboundary operator δ : C⁰ → C¹.
    """

    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any], src_map: Sequence[float], dst_map: Sequence[float]) -> None:
        """
        Define the linear maps associated with an edge.
        src_map : map from source node to edge space
        dst_map : map from destination node to edge space
        """
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float),
                                      np.array(dst_map, dtype=float))

    def set_section(self, node: Any, value: Sequence[float]) -> None:
        """Assign a section (value) to a node."""
        self._sections[node] = np.array(value, dtype=float)

    # ------------------------------------------------------------------
    # Internal layout helpers
    # ------------------------------------------------------------------
    def _edge_dim(self, u: Any, v: Any) -> int:
        """Dimension of the vector space attached to edge (u, v)."""
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        """Compute node ordering and offsets for C⁰."""
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos  # pos == total dimension of C⁰

    def _c1_layout(self):
        """Compute edge ordering and offsets for C¹."""
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos  # pos == total dimension of C¹

    # ------------------------------------------------------------------
    # Coboundary operator
    # ------------------------------------------------------------------
    def coboundary_operator(self) -> np.ndarray:
        """
        Build the matrix representation of δ : C⁰ → C¹.
        For each edge (u, v) we insert -src_map in the columns of u
        and +dst_map in the columns of v.
        """
        nodes, c0_off, c0_dim = self._c0_layout()
        c1_off, c1_dim = self._c1_layout()
        delta = np.zeros((c1_dim, c0_dim), dtype=float)

        for u, v in self.edges:
            row_start, d_e = c1_off[(u, v)]

            if (u, v) in self._restrictions:
                src_map, dst_map = self._restrictions[(u, v)]
            else:
                # reverse orientation
                dst_map, src_map = self._restrictions[(v, u)]

            col_u = c0_off[u]
            col_v = c0_off[v]
            dim_u = self.node_dims[u]
            dim_v = self.node_dims[v]

            # Insert the maps (note the sign convention)
            delta[row_start:row_start + d_e, col_u:col_u + dim_u] -= src_map.reshape(d_e, dim_u)
            delta[row_start:row_start + d_e, col_v:col_v + dim_v] += dst_map.reshape(d_e, dim_v)

        return delta

    # ------------------------------------------------------------------
    # Residual evaluation
    # ------------------------------------------------------------------
    def apply_to_section(self) -> np.ndarray:
        """
        Concatenate all node sections into a single C⁰ vector and apply δ.
        Returns the resulting C¹ vector (the residual).
        """
        nodes, c0_off, c0_dim = self._c0_layout()
        section_vec = np.zeros((c0_dim,), dtype=float)

        for n in nodes:
            if n not in self._sections:
                raise ValueError(f"Missing section for node {n}")
            offset = c0_off[n]
            dim = self.node_dims[n]
            section_vec[offset:offset + dim] = self._sections[n].reshape(dim)

        delta = self.coboundary_operator()
        residual = delta @ section_vec
        return residual


# ----------------------------------------------------------------------
# Hybrid functions that fuse allocation and sheaf topology
# ----------------------------------------------------------------------
def build_sheaf_from_allocation(allocation: Dict[str, Any],
                                groups: Tuple[str, ...] = GROUPS) -> Sheaf:
    """
    Construct a Sheaf where each group is a node of dimension 1.
    Edges connect consecutive groups in a ring (i.e. a cycle graph).
    The restriction maps are identity (1→1) scaled by the absolute difference
    of the groups' weekday weights; this encodes the *topological tension* of
    the allocation.
    """
    # Node dimensions: scalar per group
    node_dims = {g: 1 for g in groups}
    # Cycle edges: (g_i, g_{i+1})
    edges = [(groups[i], groups[(i + 1) % len(groups)]) for i in range(len(groups))]
    sheaf = Sheaf(node_dims, edges)

    # Populate sections with the allocated LLM units per group
    for lane in allocation["allocation"]:
        grp = lane["group"]
        value = lane["llm_units"]
        sheaf.set_section(grp, [value])

    # Define restriction maps.
    # For edge (u, v) we use src_map = [w_u], dst_map = [w_v] where w_* are
    # the weekday weights for the respective groups (from the same allocation dict).
    weight_lookup = {lane["group"]: lane["weekday_weight"] for lane in allocation["allocation"]}

    for (u, v) in edges:
        w_u = weight_lookup[u]
        w_v = weight_lookup[v]
        # Identity scaled by weight; shape (1,1)
        sheaf.set_restriction((u, v), src_map=[w_u], dst_map=[w_v])

    return sheaf


def sheaf_residual_from_allocation(allocation: Dict[str, Any],
                                  groups: Tuple[str, ...] = GROUPS) -> float:
    """
    Build a sheaf from ``allocation`` and return the L2 norm of the coboundary
    residual vector.  A zero residual would indicate perfect consistency of the
    allocation with the imposed edge relations.
    """
    sheaf = build_sheaf_from_allocation(allocation, groups)
    residual_vec = sheaf.apply_to_section()
    # L2 norm as scalar measure of inconsistency
    return float(np.linalg.norm(residual_vec, ord=2))


def hybrid_workshare_analysis(total_units: float,
                              date: dt.date,
                              deterministic_target_pct: float = 90.0,
                              groups: Tuple[str, ...] = GROUPS) -> Dict[str, Any]:
    """
    End‑to‑end hybrid routine:

    1. Allocate resources (Parent A).
    2. Build the corresponding sheaf (Parent B).
    3. Compute the consistency residual.

    Returns a dictionary summarising the whole pipeline.
    """
    allocation = allocate_hybrid(
        total_units=total_units,
        date=date,
        deterministic_target_pct=deterministic_target_pct,
        groups=groups,
    )
    residual_norm = sheaf_residual_from_allocation(allocation, groups)

    summary = {
        "date": date.isoformat(),
        "total_units": total_units,
        "deterministic_target_pct": deterministic_target_pct,
        "allocation": allocation,
        "sheaf_residual_norm": _pct(residual_norm),
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    TOTAL = 1000.0
    TODAY = dt.date.today()
    DET_PCT = 85.0

    result = hybrid_workshare_analysis(
        total_units=TOTAL,
        date=TODAY,
        deterministic_target_pct=DET_PCT,
        groups=GROUPS,
    )

    print("Hybrid Workshare Analysis")
    print("--------------------------")
    print(f"Date                     : {result['date']}")
    print(f"Total units              : {result['total_units']}")
    print(f"Deterministic target %   : {result['deterministic_target_pct']}")
    print(f"Sheaf residual (L2 norm) : {result['sheaf_residual_norm']}")
    print("\nPer‑group allocation:")
    for lane in result["allocation"]["allocation"]:
        print(
            f"  {lane['group']:12s} | LLM units: {lane['llm_units']:8.3f} | "
            f"Weight: {lane['weekday_weight']:5.3f}"
        )