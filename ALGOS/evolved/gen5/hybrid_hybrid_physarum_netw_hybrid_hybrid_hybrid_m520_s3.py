# DARWIN HAMMER — match 520, survivor 3
# gen: 5
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s2.py (gen4)
# born: 2026-05-29T23:29:37Z

"""Hybrid Physarum‑Bandit‑Workshare Algorithm
================================================

Parents
-------
* **Parent A** – ``physarum_network`` + contextual bandit: defines a flux
  `q = C/ℓ * (p_up‑p_down)` and a conductance update
  `C ← max(0, C + dt·(gain·|q| – decay·C))`.
* **Parent B** – ``hybrid_workshare`` + epistemic‑flag‑weighted minimum‑cost tree:
  provides a weekday‑dependent weight vector for groups and a mapping from
  epistemic certainty flags to scalar modifiers that are applied to edge
  costs in a minimum‑spanning‑tree (MST) construction.

Mathematical Bridge
-------------------
1. Every **action / group** is a node of a fully‑connected graph.
2. Each unordered pair *(i, j)* defines an edge with  
   *conductance* `C_ij` (from Parent A) and *length* `ℓ_ij` (a hyper‑parameter).  
3. The **edge cost** used by the MST of Parent B is  

   
   cost_ij = (ℓ_ij / max(C_ij, eps)) * epistemic_factor(flag_ij)
   

   i.e. the inverse conductance (physarum resistance) scaled by an
   epistemic certainty factor.
4. After the MST is built, the **allocation** of a total resource `U` to the
   groups is driven by the weekday weight vector (Parent B) multiplied by the
   node degree in the MST (a proxy for “network importance” derived from the
   physarum dynamics).

The module implements the fused dynamics, exposing three core functions:
`flux`, `update_conductance`, and `hybrid_step`.  A lightweight
`HybridPhysarumWorkshare` class orchestrates state, and the ``__main__`` block
executes a smoke test."""

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
    # ------------------------------------------------------------------
    # 1. Flux computation & conductance update (upper triangle only)
    # ------------------------------------------------------------------
    new_conductance = conductance_mat.copy()
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(conductance_mat[i, j], length_mat[i, j],
                     pressures[i], pressures[j])
            new_c = update_conductance(conductance_mat[i, j], q,
                                       dt=dt, gain=gain, decay=decay)
            new_conductance[i, j] = new_conductance[j, i] = new_c

    # ------------------------------------------------------------------
    # 2. Edge cost construction using epistemic flags
    # ------------------------------------------------------------------
    edge_costs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            flag = epistemic_flags_mat[i, j]
            cost = compute_edge_cost(new_conductance[i, j],
                                     length_mat[i, j],
                                     flag)
            edge_costs[(i, j)] = cost

    # ------------------------------------------------------------------
    # 3. Minimum‑cost spanning tree (Kruskal)
    # ------------------------------------------------------------------
    nodes = list(range(n))
    mst_edges = kruskal_mst(nodes, edge_costs)

    # node degree in the MST
    degree = np.zeros(n, dtype=int)
    for i, j in mst_edges:
        degree[i] += 1
        degree[j] += 1

    # ------------------------------------------------------------------
    # 4. Allocation using weekday weight vector
    # ------------------------------------------------------------------
    today = datetime.now(timezone.utc)
    dow = today.weekday()  # Monday=0 … Sunday=6
    w_vec = weekday_weight_vector(GROUPS, dow)
    raw_alloc = w_vec * degree
    if raw_alloc.sum() == 0:
        allocation = {g: 0.0 for g in GROUPS}
    else:
        normalized = raw_alloc / raw_alloc.sum()
        allocation = {g: float(v) for g, v in zip(GROUPS, normalized)}

    return new_conductance, allocation


# ----------------------------------------------------------------------
# Light‑weight orchestrator class
# ----------------------------------------------------------------------
@dataclass
class HybridPhysarumWorkshare:
    """Container for the hybrid state."""
    conductance_mat: np.ndarray
    length_mat: np.ndarray
    epistemic_flags_mat: np.ndarray
    pressures: np.ndarray

    @classmethod
    def random_init(cls,
                    n_groups: int = len(GROUPS),
                    seed: int | None = None) -> "HybridPhysarumWorkshare":
        rng = np.random.default_rng(seed)
        # Positive random lengths in (0.1, 1.0)
        length = rng.uniform(0.1, 1.0, size=(n_groups, n_groups))
        length = (length + length.T) / 2.0
        np.fill_diagonal(length, 0.0)

        # Small initial conductances
        conductance = rng.uniform(0.01, 0.1, size=(n_groups, n_groups))
        conductance = (conductance + conductance.T) / 2.0
        np.fill_diagonal(conductance, 0.0)

        # Random epistemic flags per unordered edge
        flags = rng.choice(EPISTEMIC_FLAGS, size=(n_groups, n_groups))
        # Symmetrize (choose upper triangle, mirror)
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                flags[j, i] = flags[i, j]
        np.fill_diagonal(flags, "FACT")  # self‑edges unused

        # Random pressures (e.g. expected rewards)
        pressures = rng.normal(loc=0.0, scale=1.0, size=n_groups)

        return cls(conductance, length, flags, pressures)

    def step(self,
             dt: float = 0.1,
             gain: float = 1.0,
             decay: float = 0.01) -> Dict[str, float]:
        """Execute one hybrid iteration and update internal state."""
        new_c, alloc = hybrid_step(self.conductance_mat,
                                   self.length_mat,
                                   self.pressures,
                                   self.epistemic_flags_mat,
                                   dt=dt,
                                   gain=gain,
                                   decay=decay)
        self.conductance_mat = new_c
        # Simulate pressure drift (e.g., new reward observations)
        self.pressures += np.random.normal(scale=0.05, size=self.pressures.shape)
        return alloc


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a random hybrid system
    system = HybridPhysarumWorkshare.random_init(seed=42)

    print("Initial conductance matrix (excerpt):")
    print(system.conductance_mat[:4, :4])

    # Run a few hybrid iterations
    for step in range(5):
        allocation = system.step()
        print(f"\nStep {step + 1} allocation:")
        for grp, val in allocation.items():
            print(f"  {grp}: {val:.4f}")

    print("\nFinal conductance matrix (excerpt):")
    print(system.conductance_mat[:4, :4])