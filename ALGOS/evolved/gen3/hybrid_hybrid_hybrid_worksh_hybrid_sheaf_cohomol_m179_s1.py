# DARWIN HAMMER — match 179, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py (gen2)
# parent_b: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
# born: 2026-05-29T23:25:59Z

"""
HARDY HAMMER — match 25, survivor 1
gen: 1
parent_a: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s3.py (gen2)
parent_b: hybrid_sheaf_cohomology_percyphon_m2_s1.py (gen1)
born: 2026-05-30T14:45:12Z

This module represents a mathematical fusion of the hybrid workshare allocation
algorithm and the sheaf cohomology structure.
We found the mathematical interface to be in the use of linear transformations
to map between different vector spaces.
In the original workshare algorithm, we used a sinusoidal rotation to yield a
row-stochastic weight vector for allocation.
We have integrated this rotation into the sheaf cohomology as a way to assign
restriction maps between the stalks at different nodes in the graph.
"""

import numpy as np
import hashlib
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Utility helpers
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
    Sinusoidal rotation yields a row-stochastic vector.
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


class HybridSheaf:
    """
    Cellular sheaf over a graph with hybrid weights based on weekdays.
    """

    def __init__(self, node_dims, edge_list, groups: Tuple[str, ...]):
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self.groups = groups
        # restriction maps: (u, v) -> (src_map: R^{dim_u}->R^{d_e}, dst_map: R^{dim_v}->R^{d_e})
        self._restrictions = {}
        # local sections: node_id -> numpy array of shape (dim,)
        self._sections = {}
        self._weekday_weights = {}

    def set_restriction(self, edge, src_map, dst_map, dow: int):
        """Assign restriction maps for an oriented edge and its weekday weight.

        Parameters
        ----------
        edge : (u, v)
            Must match an entry in edge_list with the same orientation.
        src_map : numpy array, shape (edge_dim, dim_u)
            Linear map F(u->e): stalk at u -> stalk at e.
        dst_map : numpy array, shape (edge_dim, dim_v)
            Linear map F(v->e): stalk at v -> stalk at e.
        dow : int
            Weekday index where 0 = Sunday … 6 = Saturday.
        """
        self._restrictions[(edge, dow)] = (src_map, dst_map)

    def set_section(self, node, value):
        """Assign a local section to a node."""
        self._sections[node] = value

    def get_weekday_weights(self, dow: int):
        """Get the weekday weights for the given day."""
        if dow not in self._weekday_weights:
            self._weekday_weights[dow] = weekday_weight_vector(self.groups, dow)
        return self._weekday_weights[dow]

    def hybridize_sections(self, day: int):
        """Hybridize the local sections based on the weekday weights."""
        dow = doomsday(day=day, month=1, year=2022)
        weights = self.get_weekday_weights(dow)
        for node, section in self._sections.items():
            self._sections[node] = weights @ section


def allocate_hybrid(
    *,
    total_units: float,
    day: int,
    deterministic_target_pct: float = 90.0,
    groups: Tuple[str, ...] = GROUPS,
) -> HybridSheaf:
    """
    Split ``total_units`` into deterministic and hybrid parts, then
    distribute the hybrid part across the nodes in the graph using the weekday weights.
    Returns a HybridSheaf object mirroring the original schema with added calendar metadata.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not (0.0 <= deterministic_target_pct <= 100.0):
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")

    deterministic_units = total_units * deterministic_target_pct
    hybrid_units = total_units - deterministic_units

    # Create a new HybridSheaf object
    hybrid_sheaf = HybridSheaf(
        node_dims={"node1": 10, "node2": 20}, edge_list=[("node1", "node2")], groups=groups
    )

    # Set the local sections for each node
    hybrid_sheaf.set_section("node1", np.random.rand(10))
    hybrid_sheaf.set_section("node2", np.random.rand(20))

    # Hybridize the local sections based on the weekday weights
    hybrid_sheaf.hybridize_sections(day)

    # Assign the weekday weights to each edge
    dow = doomsday(day=day, month=1, year=2022)
    weights = hybrid_sheaf.get_weekday_weights(dow)
    hybrid_sheaf.set_restriction(
        edge=("node1", "node2"),
        src_map=np.random.rand(2, 10),
        dst_map=np.random.rand(2, 20),
        dow=dow,
    )

    return hybrid_sheaf


def test_hybrid_sheaf():
    # Smoke test
    hybrid_sheaf = allocate_hybrid(day=1, total_units=100)
    print(hybrid_sheaf._sections["node1"])
    print(hybrid_sheaf.get_weekday_weights(1))


if __name__ == "__main__":
    test_hybrid_sheaf()