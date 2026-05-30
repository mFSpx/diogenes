# DARWIN HAMMER — match 520, survivor 4
# gen: 5
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s2.py (gen4)
# born: 2026-05-29T23:29:37Z

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (physarum flux & conductance dynamics)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    """Conductance ODE step based on absolute flux."""
    delta = dt * (gain * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c


# ----------------------------------------------------------------------
# Parent B primitives (weekday weight vector & epistemic flag handling)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Simple scalar mapping for each flag (higher = more trustworthy)
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Return a normalized weight vector that varies sinusoidally with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def epistemic_factor(flag: str) -> float:
    """Map an epistemic certainty flag to a multiplicative cost factor."""
    return _EPISTEMIC_WEIGHT.get(flag, 1.0)  # default to neutral factor


# ----------------------------------------------------------------------
# Union‑Find data‑structure for Kruskal's MST
# ----------------------------------------------------------------------
class _UF:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def compute_edge_cost(conductance: float,
                      length: float,
                      epistemic_flag: str,
                      eps: float = 1e-12) -> float:
    """Cost used by the MST: resistance scaled by epistemic certainty."""
    resistance = length / max(conductance, eps)
    return resistance * epistemic_factor(epistemic_flag)


def kruskal_mst(nodes: List[str],
                edge_costs: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    """Return list of edges (as index pairs) forming the minimum‑cost spanning tree."""
    n = len(nodes)
    uf = _UF(n)
    # sort edges by cost
    sorted_edges = sorted(edge_costs.items(), key=lambda item: item[1])
    mst: List[Tuple[int, int]] = []
    for (i, j), cost in sorted_edges:
        if uf.union(i, j):
            mst.append((i, j))
        if len(mst) == n - 1:
            break
    return mst


def hybrid_step(conductance_mat: np.ndarray,
                length_mat: np.ndarray,
                pressures: np.ndarray,
                epistemic_flags_mat: np.ndarray,
                dt: float = 0.1,
                gain: float = 1.0,
                decay: float = 0.01) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform one hybrid iteration:

    1. Compute flux on every unordered edge using current pressures.
    2. Update the conductance matrix via the physarum rule.
    3. Build an epistemic‑aware MST from the updated conductances.
    4. Allocate a unit resource to each group proportionally to
       (weekday weight) × (node degree in the MST).

    Returns the updated conductance matrix and the allocation dictionary.
    """
    n = conductance_mat.shape[0]
    if pressures.shape[0] != n:
        raise ValueError("pressures length must match conductance matrix size")

    # 1. Flux computation & conductance update (upper triangle only)
    new_conductance = conductance_mat.copy()
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(conductance_mat[i, j], length_mat[i, j],
                     pressures[i], pressures[j])
            new_c = update_conductance(conductance_mat[i, j], q,
                                       dt=dt, gain=gain, decay=decay)
            new_conductance[i, j] = new_conductance[j, i] = new_c

    # 2. Edge cost construction using epistemic flags
    edge_costs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            flag = epistemic_flags_mat[i, j]
            cost = compute_edge_cost(new_conductance[i, j],
                                     length_mat[i, j],
                                     flag)
            edge_costs[(i, j)] = cost

    # 3. Build MST
    nodes = [f"node_{i}" for i in range(n)]
    mst_edges = kruskal_mst(nodes, edge_costs)

    # 4. Compute allocation
    weekday = datetime.now(timezone.utc).weekday()
    weight_vec = weekday_weight_vector(GROUPS, weekday)
    node_degrees = {i: 0 for i in range(n)}
    for edge in mst_edges:
        node_degrees[edge[0]] += 1
        node_degrees[edge[1]] += 1
    allocation = {group: weight_vec[i] * node_degrees[i] for i, group in enumerate(GROUPS)}
    allocation = {k: v / sum(allocation.values()) for k, v in allocation.items()}

    return new_conductance, allocation


class HybridPhysarumWorkshare:
    def __init__(self,
                 conductance_mat: np.ndarray,
                 length_mat: np.ndarray,
                 pressures: np.ndarray,
                 epistemic_flags_mat: np.ndarray):
        self.conductance_mat = conductance_mat
        self.length_mat = length_mat
        self.pressures = pressures
        self.epistemic_flags_mat = epistemic_flags_mat

    def hybrid_step(self) -> Tuple[np.ndarray, Dict[str, float]]:
        return hybrid_step(self.conductance_mat,
                           self.length_mat,
                           self.pressures,
                           self.epistemic_flags_mat)


if __name__ == "__main__":
    # Example usage
    np.random.seed(0)
    conductance_mat = np.random.rand(4, 4)
    length_mat = np.random.rand(4, 4)
    pressures = np.random.rand(4)
    epistemic_flags_mat = np.array([[random.choice(EPISTEMIC_FLAGS) for _ in range(4)] for _ in range(4)])

    hybrid = HybridPhysarumWorkshare(conductance_mat, length_mat, pressures, epistemic_flags_mat)
    updated_conductance, allocation = hybrid.hybrid_step()
    print("Updated Conductance Matrix:")
    print(updated_conductance)
    print("Allocation:")
    print(allocation)