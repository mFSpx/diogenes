# DARWIN HAMMER — match 3442, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_fracti_hybrid_hybrid_shanno_m1432_s0.py (gen6)
# born: 2026-05-29T23:50:06Z

"""Hybrid Physarum–Hypervector Network Fusion

Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s0.py
- hybrid_hybrid_hybrid_fracti_hybrid_hybrid_shanno_m1432_s0.py

Bridge:
The physarum model provides scalar fluxes on graph edges derived from
conductances, edge lengths and node pressures.  Hyperdimensional computing
offers a binding operator (circular convolution) that can merge two
hypervectors into a single representation.  The fusion maps each edge
flux to a *scaled* binding of the hypervectors attached to its incident
nodes.  Conductance updates (physarum ODE) modulate the magnitude of the
edge‑flux hypervectors, while the bound hypervectors are accumulated on
the nodes, producing a joint dynamical system that simultaneously
optimises network flow and evolves a distributed hyperdimensional state."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Physarum core (Parent A)
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
# Hyperdimensional core (Parent B)
# ----------------------------------------------------------------------
def random_hv(d: int = 10000, kind: str = "complex", seed=None):
    """Generate a random hypervector of dimension d."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"kind must be 'complex', 'bipolar', or 'real'; got {kind!r}")


def bind(X, Y):
    """Bind two hypervectors via circular convolution (frequency‑domain multiplication)."""
    X = np.asarray(X)
    Y = np.asarray(Y)
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z, Y):
    """Invert binding: recover X from Z = X (*) Y."""
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------
Edge = Tuple[int, int, float]  # (node_i, node_j, length)


def generate_node_hypervectors(num_nodes: int, dim: int = 2048, seed: int = 0) -> List[np.ndarray]:
    """Create a distinct hypervector for each node."""
    rng = np.random.default_rng(seed)
    return [random_hv(dim, kind="complex", seed=int(rng.integers(0, 2**31 - 1))) for _ in range(num_nodes)]


def compute_pressures(num_nodes: int,
                      sources: Sequence[int],
                      sinks: Sequence[int],
                      base_pressure: float = 1.0,
                      rng_seed: int = None) -> np.ndarray:
    """
    Very simple pressure model:
    - source nodes receive +base_pressure
    - sink nodes receive -base_pressure
    - all other nodes receive a random value in [-base_pressure/2, base_pressure/2]
    """
    rng = np.random.default_rng(rng_seed)
    pressures = rng.uniform(-base_pressure / 2, base_pressure / 2, size=num_nodes)
    pressures[list(sources)] = base_pressure
    pressures[list(sinks)] = -base_pressure
    return pressures


def compute_edge_fluxes(conductance_mat: np.ndarray,
                        edges: List[Edge],
                        pressures: np.ndarray) -> Dict[Edge, float]:
    """Return a mapping edge → flux using the physarum flux equation."""
    fluxes: Dict[Edge, float] = {}
    for i, j, length in edges:
        c = conductance_mat[i, j]
        p_i = pressures[i]
        p_j = pressures[j]
        fluxes[(i, j, length)] = flux(c, length, p_i, p_j)
    return fluxes


def bind_edge_hypervectors(edge: Edge,
                           q: float,
                           node_hvs: List[np.ndarray]) -> np.ndarray:
    """
    Produce a hypervector that encodes the edge state:
    - Bind the hypervectors of the two incident nodes.
    - Scale the resulting vector by the absolute flux magnitude (|q|).
    """
    i, j, _ = edge
    bound = bind(node_hvs[i], node_hvs[j])
    # Scaling preserves direction (complex phase) while modulating amplitude.
    return bound * abs(q)


def hybrid_iteration(conductance_mat: np.ndarray,
                     edges: List[Edge],
                     node_hvs: List[np.ndarray],
                     sources: Sequence[int],
                     sinks: Sequence[int],
                     dt: float = 0.1,
                     gain: float = 1.0,
                     decay: float = 0.01,
                     rng_seed: int = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    One hybrid update step:
    1. Compute node pressures.
    2. Compute edge fluxes (physarum).
    3. Update conductances using the physarum ODE.
    4. For each edge, bind the incident node hypervectors and scale by flux.
    5. Accumulate the resulting edge hypervectors onto the incident nodes
       (simple addition) – this propagates the flow information into the
       hyperdimensional representation.
    6. Normalise node hypervectors to unit norm to avoid numerical blow‑up.
    Returns the updated conductance matrix and node hypervectors.
    """
    num_nodes = conductance_mat.shape[0]

    # 1. Pressures
    pressures = compute_pressures(num_nodes, sources, sinks, rng_seed=rng_seed)

    # 2. Edge fluxes
    edge_fluxes = compute_edge_fluxes(conductance_mat, edges, pressures)

    # 3. Conductance update
    new_conductance = conductance_mat.copy()
    for (i, j, _), q in edge_fluxes.items():
        c_old = conductance_mat[i, j]
        c_new = update_conductance(c_old, q, dt=dt, gain=gain, decay=decay)
        new_conductance[i, j] = new_conductance[j, i] = c_new

    # 4‑5. Hypervector propagation
    node_updates = [np.zeros_like(node_hvs[0]) for _ in range(num_nodes)]
    for edge, q in edge_fluxes.items():
        hv_edge = bind_edge_hypervectors(edge, q, node_hvs)
        i, j, _ = edge
        node_updates[i] += hv_edge
        node_updates[j] += hv_edge

    # 6. Apply updates and normalise
    new_node_hvs = []
    for hv, upd in zip(node_hvs, node_updates):
        hv_new = hv + upd
        norm = np.linalg.norm(hv_new)
        if norm > 0:
            hv_new = hv_new / norm
        new_node_hvs.append(hv_new)

    return new_conductance, new_node_hvs


# ----------------------------------------------------------------------
# Simple demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Graph: triangle (3 nodes, 3 edges)
    num_nodes = 3
    edges: List[Edge] = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 0, 1.0),
    ]

    # Initialise conductance matrix (symmetric, zero diagonal)
    conductance = np.zeros((num_nodes, num_nodes))
    for i, j, _ in edges:
        conductance[i, j] = conductance[j, i] = random.uniform(0.1, 1.0)

    # Node hypervectors
    node_hvs = generate_node_hypervectors(num_nodes, dim=1024, seed=42)

    # Define sources and sinks
    sources = [0]
    sinks = [2]

    # Run a few hybrid iterations
    for step in range(5):
        conductance, node_hvs = hybrid_iteration(
            conductance,
            edges,
            node_hvs,
            sources,
            sinks,
            dt=0.05,
            gain=1.2,
            decay=0.02,
            rng_seed=step,
        )
        # Simple diagnostics
        total_conductance = np.sum(conductance) / 2  # each edge counted twice
        print(f"Step {step+1}: total conductance = {total_conductance:.4f}")

    print("Hybrid iteration completed without errors.")