# DARWIN HAMMER — match 2939, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_serpentina_self_righ_m1651_s0.py (gen5)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s1.py (gen2)
# born: 2026-05-29T23:46:53Z

"""Hybrid Flux-Privacy Model Pool

This module fuses two parent algorithms:

* **Parent A** – Flux‑based conductance network with a sphericity index
  (functions `flux`, `update_conductance`, `sphericity_index`).

* **Parent B** – Model‑pool management with reconstruction risk scoring,
  differential‑privacy aggregation and sparse winner‑take‑all (functions
  `reconstruction_risk_score`, `dp_aggregate`, `winner_take_all`).

**Mathematical bridge**

The bridge is built by letting the *risk‑derived pressure* of a model drive
the flux through a network edge, while the *sphericity* of the model scales
the conductance update.  Concretely:


pressure_i = base_pressure * (1 + risk_i)
flux_e    = g_e / L_e * (pressure_a - pressure_b)
g'_e      = max(0, g_e + Δt·(gain·|flux_e| - decay·g_e)·sphericity_i)


where `risk_i` comes from `reconstruction_risk_score` and
`sphericity_i` from `sphericity_index`.  Differential‑privacy noise can be
added to the flux vector using `dp_aggregate`.  The resulting hybrid
operations are exposed through three public functions.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Ohm‑like flux through an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    """Standard conductance update."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity used as a scaling factor."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


# ----------------------------------------------------------------------
# Core structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str                     # e.g. "T1", "T2", "T3"
    length: float                 # geometric dimensions for sphericity
    width: float
    height: float
    quasi_identifiers: int        # unique QI count for risk scoring
    total_records: int            # total records for risk scoring


class ModelPool:
    """Simple RAM‑bounded pool with risk‑aware eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model; raises if constraints are violated."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting the lowest‑risk models until there is space."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict model with smallest risk score
            lowest = min(self.loaded.values(),
                         key=lambda m: reconstruction_risk_score(m.quasi_identifiers,
                                                                 m.total_records))
            del self.loaded[lowest.name]
        self.load(model)


def reconstruction_risk_score(unique_quasi_identifiers: int,
                             total_records: int) -> float:
    """Risk ∈ [0,1] based on proportion of unique identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """Laplace‑noised sum for differential privacy."""
    total = sum(values)
    noise = np.random.laplace(0.0, sensitivity / epsilon)
    return total + noise


def winner_take_all(scores: List[float]) -> int:
    """Sparse winner‑take‑all: returns index of the maximal score."""
    if not scores:
        raise ValueError("scores list empty")
    return int(np.argmax(scores))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------
@dataclass
class Edge:
    """Network edge linking two models."""
    node_a: str
    node_b: str
    conductance: float
    length: float


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_pressures(pool: ModelPool,
                      base_pressure: float = 1.0) -> Dict[str, float]:
    """
    Map each loaded model to a pressure value.
    Pressure = base_pressure * (1 + risk_score).
    """
    pressures = {}
    for name, model in pool.loaded.items():
        risk = reconstruction_risk_score(model.quasi_identifiers,
                                         model.total_records)
        pressures[name] = base_pressure * (1.0 + risk)
    return pressures


def hybrid_flux_update(pool: ModelPool,
                       edges: List[Edge],
                       dt: float = 1.0,
                       gain: float = 1.0,
                       decay: float = 0.05,
                       base_pressure: float = 1.0) -> List[Edge]:
    """
    Perform one hybrid update step:
      1. Compute pressures from reconstruction risk.
      2. Compute flux on each edge.
      3. Update conductance using sphericity as a multiplicative factor.
    Returns the list of edges with updated conductance.
    """
    pressures = compute_pressures(pool, base_pressure)

    updated_edges = []
    for e in edges:
        if e.node_a not in pressures or e.node_b not in pressures:
            # Edge references an unloaded model; skip update.
            updated_edges.append(e)
            continue

        p_a = pressures[e.node_a]
        p_b = pressures[e.node_b]

        # 1) Flux according to Parent A
        q = flux(e.conductance, e.length, p_a, p_b)

        # 2) Sphericity factor from the *source* node (arbitrary choice)
        src_model = pool.loaded[e.node_a]
        sph = sphericity_index(src_model.length,
                               src_model.width,
                               src_model.height)

        # 3) Conductance update scaled by sphericity (the bridge)
        new_g = update_conductance(e.conductance, q, dt, gain, decay)
        new_g *= sph  # bridge: sphericity modulates learning rate

        updated_edges.append(Edge(e.node_a, e.node_b, new_g, e.length))
    return updated_edges


def dp_noise_fluxes(edges: List[Edge],
                    pool: ModelPool,
                    epsilon: float = 1.0) -> List[Tuple[Edge, float]]:
    """
    Compute fluxes for all edges, then add Laplace noise using DP aggregation.
    Returns a list of (edge, noisy_flux) pairs.
    """
    pressures = compute_pressures(pool)
    raw_fluxes = []
    for e in edges:
        if e.node_a in pressures and e.node_b in pressures:
            q = flux(e.conductance, e.length,
                     pressures[e.node_a], pressures[e.node_b])
        else:
            q = 0.0
        raw_fluxes.append(q)

    # Apply DP noise to each flux individually (noise scale = 1/epsilon)
    noisy_fluxes = [q + np.random.laplace(0.0, 1.0 / epsilon) for q in raw_fluxes]
    return list(zip(edges, noisy_fluxes))


def sparse_wta_on_fluxes(edges: List[Edge],
                         pool: ModelPool) -> Edge:
    """
    Use winner‑take‑all on the absolute flux magnitudes to select the
    most active edge in the current network configuration.
    """
    pressures = compute_pressures(pool)
    fluxes = []
    for e in edges:
        if e.node_a in pressures and e.node_b in pressures:
            q = flux(e.conductance, e.length,
                     pressures[e.node_a], pressures[e.node_b])
        else:
            q = 0.0
        fluxes.append(abs(q))

    winner_idx = winner_take_all(fluxes)
    return edges[winner_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small model pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Create three dummy models with geometric sizes and risk parameters
    m1 = ModelTier(name="A", ram_mb=400, tier="T1",
                  length=2.0, width=1.5, height=1.0,
                  quasi_identifiers=30, total_records=100)
    m2 = ModelTier(name="B", ram_mb=600, tier="T1",
                  length=1.8, width=1.2, height=1.2,
                  quasi_identifiers=50, total_records=120)
    m3 = ModelTier(name="C", ram_mb=800, tier="T2",
                  length=2.5, width=2.0, height=1.5,
                  quasi_identifiers=10, total_records=80)

    # Load models (eviction not needed for this tiny example)
    for m in (m1, m2, m3):
        pool.load(m)

    # Define a simple triangle network
    edges = [
        Edge("A", "B", conductance=1.0, length=1.0),
        Edge("B", "C", conductance=0.8, length=1.2),
        Edge("C", "A", conductance=0.5, length=0.9)
    ]

    # Perform a hybrid flux‑conductance update
    edges = hybrid_flux_update(pool, edges, dt=0.5, gain=1.2, decay=0.03)

    # Add differential‑privacy noise to fluxes
    noisy = dp_noise_fluxes(edges, pool, epsilon=0.8)

    # Select the most active edge via sparse WTA
    winner = sparse_wta_on_fluxes(edges, pool)

    print("Updated edges (conductance):")
    for e in edges:
        print(f"  {e.node_a}<->{e.node_b}: g={e.conductance:.4f}")

    print("\nNoisy fluxes:")
    for e, q in noisy:
        print(f"  {e.node_a}<->{e.node_b}: flux≈{q:.4f}")

    print(f"\nWinner‑take‑all selected edge: {winner.node_a}<->{winner.node_b}")