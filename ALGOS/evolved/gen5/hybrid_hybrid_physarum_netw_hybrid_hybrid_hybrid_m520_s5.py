# DARWIN HAMMER — match 520, survivor 5
# gen: 5
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m97_s2.py (gen4)
# born: 2026-05-29T23:29:37Z

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}
# Large multiplier for edges with near‑zero conductance (prevents overflow)
_INFINITE_COST_MULTIPLIER: float = 1e6


# ----------------------------------------------------------------------
# Parent A primitives (physarum flux & conductance dynamics)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    """Conductance ODE step based on absolute flux."""
    delta = dt * (gain * abs(q) - decay * conductance)
    return max(0.0, conductance + delta)


# ----------------------------------------------------------------------
# Parent B primitives (weekday weight vector & epistemic flag handling)
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Normalized sinusoidal weight vector that varies with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    return (raw / raw.sum()).astype(np.float64)


def epistemic_factor(flag: str) -> float:
    """Map an epistemic certainty flag to a multiplicative cost factor."""
    return _EPISTEMIC_WEIGHT.get(flag, 1.0)  # default to neutral factor


# ----------------------------------------------------------------------
# Union‑Find data‑structure for Kruskal's MST
# ----------------------------------------------------------------------
class _UF:
    __slots__ = ("parent", "rank")

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
# Core hybrid operations – deeper integration
# ----------------------------------------------------------------------
def compute_edge_cost(conductance: float,
                      length: float,
                      epistemic_flag: str,
                      eps: float = 1e-12) -> float:
    """Cost used by the MST: resistance scaled by epistemic certainty."""
    if conductance <= eps:
        # Near‑zero conductance → effectively infinite resistance
        return length * _INFINITE_COST_MULTIPLIER * epistemic_factor(epistemic_flag)
    resistance = length / conductance
    return resistance * epistemic_factor(epistemic_flag)


def kruskal_mst(nodes: List[str],
                edge_costs: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    """Return edges (as index pairs) forming the minimum‑cost spanning tree."""
    n = len(nodes)
    uf = _UF(n)
    sorted_edges = sorted(edge_costs.items(), key=lambda item: item[1])
    mst: List[Tuple[int, int]] = []
    for (i, j), cost in sorted_edges:
        if uf.union(i, j):
            mst.append((i, j))
            if len(mst) == n - 1:
                break
    return mst


def solve_pressures(conductance_mat: np.ndarray,
                    length_mat: np.ndarray,
                    source_idx: int,
                    sink_weights: np.ndarray,
                    eps: float = 1e-12) -> np.ndarray:
    """
    Compute node pressures by solving the linear system L p = b,
    where L is the weighted Laplacian built from conductances and lengths,
    and b encodes a unit source at ``source_idx`` and weighted sinks.
    """
    n = conductance_mat.shape[0]
    # Build weighted adjacency (conductance / length)
    weight = np.where(length_mat > eps,
                      conductance_mat / length_mat,
                      0.0)
    np.fill_diagonal(weight, 0.0)

    # Laplacian
    L = np.diag(weight.sum(axis=1)) - weight

    # Right‑hand side: source injects +1, sinks withdraw according to sink_weights
    b = np.zeros(n, dtype=np.float64)
    b[source_idx] = 1.0
    b += -sink_weights
    # Enforce reference pressure (e.g., p_0 = 0) to make L non‑singular
    ref = 0
    L[ref, :] = 0.0
    L[ref, ref] = 1.0
    b[ref] = 0.0

    pressures = np.linalg.solve(L, b)
    return pressures


def hybrid_step(conductance_mat: np.ndarray,
                length_mat: np.ndarray,
                epistemic_flags_mat: np.ndarray,
                weekday_weights: np.ndarray,
                dt: float = 0.1,
                gain: float = 1.0,
                decay: float = 0.01,
                eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    One iteration of the improved hybrid algorithm.

    1. Derive pressures from the current conductance network using a
       source node (index 0) and sink weights derived from the weekday vector.
    2. Compute fluxes on all edges and update conductances via the physarum rule.
    3. Build an epistemic‑aware MST from the updated conductances.
    4. Allocate resources proportionally to (weekday weight) × (total absolute flux
       incident to a node) – a finer proxy for network importance than degree.

    Returns the updated conductance matrix and a dict mapping group names to
    allocated resource fractions (summing to 1.0).
    """
    n = conductance_mat.shape[0]
    if length_mat.shape != (n, n):
        raise ValueError("length_mat must be square and match conductance_mat size")
    if epistemic_flags_mat.shape != (n, n):
        raise ValueError("epistemic_flags_mat must be square and match conductance_mat size")
    if weekday_weights.shape[0] != n:
        raise ValueError("weekday_weights length must match number of groups")

    # ------------------------------------------------------------------
    # 1. Pressure computation (source = node 0, sinks = weighted by weekday)
    # ------------------------------------------------------------------
    sink_weights = weekday_weights.copy()
    sink_weights[0] = 0.0  # source node does not act as a sink
    pressures = solve_pressures(conductance_mat, length_mat, source_idx=0,
                                sink_weights=sink_weights, eps=eps)

    # ------------------------------------------------------------------
    # 2. Flux computation & conductance update (upper triangle only)
    # ------------------------------------------------------------------
    new_conductance = conductance_mat.copy()
    incident_flux = np.zeros(n, dtype=np.float64)  # for allocation later
    for i in range(n):
        for j in range(i + 1, n):
            q = flux(conductance_mat[i, j], length_mat[i, j],
                     pressures[i], pressures[j], eps=eps)
            new_c = update_conductance(conductance_mat[i, j], q,
                                       dt=dt, gain=gain, decay=decay, eps=eps)
            new_conductance[i, j] = new_conductance[j, i] = new_c
            incident_flux[i] += abs(q)
            incident_flux[j] += abs(q)

    # ------------------------------------------------------------------
    # 3. Edge cost construction using epistemic flags
    # ------------------------------------------------------------------
    edge_costs: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            flag = epistemic_flags_mat[i, j]
            cost = compute_edge_cost(new_conductance[i, j],
                                     length_mat[i, j],
                                     flag,
                                     eps=eps)
            edge_costs[(i, j)] = cost

    mst_edges = kruskal_mst(list(GROUPS), edge_costs)

    # ------------------------------------------------------------------
    # 4. Resource allocation – flux‑based importance × weekday weight
    # ------------------------------------------------------------------
    # Normalise incident flux to avoid division by zero when network is idle
    flux_norm = incident_flux / (incident_flux.sum() + eps)
    raw_allocation = weekday_weights * flux_norm
    allocation = raw_allocation / (raw_allocation.sum() + eps)

    allocation_dict = {group: float(allocation[idx]) for idx, group in enumerate(GROUPS)}
    return new_conductance, allocation_dict


# ----------------------------------------------------------------------
# High‑level orchestrator
# ----------------------------------------------------------------------
@dataclass
class HybridPhysarumWorkshareImproved:
    conductance_mat: np.ndarray
    length_mat: np.ndarray
    epistemic_flags_mat: np.ndarray
    dt: float = 0.1
    gain: float = 1.0
    decay: float = 0.01
    eps: float = 1e-12

    def __post_init__(self) -> None:
        n = len(GROUPS)
        if self.conductance_mat.shape != (n, n):
            raise ValueError("conductance_mat must be square with size len(GROUPS)")
        if self.length_mat.shape != (n, n):
            raise ValueError("length_mat must be square with size len(GROUPS)")
        if self.epistemic_flags_mat.shape != (n, n):
            raise ValueError("epistemic_flags_mat must be square with size len(GROUPS)")

    def step(self, dow: int) -> Dict[str, float]:
        """
        Execute a single hybrid iteration for the given day‑of‑week (0‑6).

        Returns the allocation dictionary for this step.
        """
        weekday_weights = weekday_weight_vector(GROUPS, dow)
        self.conductance_mat, allocation = hybrid_step(
            self.conductance_mat,
            self.length_mat,
            self.epistemic_flags_mat,
            weekday_weights,
            dt=self.dt,
            gain=self.gain,
            decay=self.decay,
            eps=self.eps,
        )
        return allocation

    def state_snapshot(self) -> Dict[str, Any]:
        """Convenient serialisable snapshot of the internal state."""
        return {
            "conductance": self.conductance_mat.tolist(),
            "length": self.length_mat.tolist(),
            "epistemic_flags": self.epistemic_flags_mat.tolist(),
        }


# ----------------------------------------------------------------------
# Simple smoke test (executed when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    n = len(GROUPS)

    # Initialise a modestly connected network
    rng = np.random.default_rng(seed=42)
    init_conductance = rng.uniform(0.1, 1.0, size=(n, n))
    init_conductance = (init_conductance + init_conductance.T) / 2.0
    np.fill_diagonal(init_conductance, 0.0)

    # Random lengths (positive)
    length_mat = rng.uniform(0.5, 2.0, size=(n, n))
    length_mat = (length_mat + length_mat.T) / 2.0
    np.fill_diagonal(length_mat, 0.0)

    # Random epistemic flags
    flag_choices = np.array(EPISTEMIC_FLAGS)
    epistemic_flags_mat = rng.choice(flag_choices, size=(n, n))
    # Ensure symmetry (flags are symmetric for undirected edges)
    for i in range(n):
        for j in range(i + 1, n):
            epistemic_flags_mat[j, i] = epistemic_flags_mat[i, j]
    np.fill_diagonal(epistemic_flags_mat, "FACT")

    hybrid = HybridPhysarumWorkshareImproved(
        conductance_mat=init_conductance,
        length_mat=length_mat,
        epistemic_flags_mat=epistemic_flags_mat,
    )

    today = datetime.now(timezone.utc).weekday()  # 0 = Monday
    allocation = hybrid.step(today)
    print("Resource allocation for day", today, ":", allocation)
    print("Conductance snapshot (truncated):")
    print(np.round(hybrid.conductance_mat, 3))