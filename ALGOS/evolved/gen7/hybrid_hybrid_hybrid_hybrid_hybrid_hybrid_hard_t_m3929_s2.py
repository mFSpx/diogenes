# DARWIN HAMMER — match 3929, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m2320_s0.py (gen6)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s2.py (gen3)
# born: 2026-05-29T23:52:42Z

"""Hybrid Algorithm Fusion of Physarum‑Network & Krampus Curvature with Hard‑Truth Stylometry

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_physarum_netw_m2320_s0.py (Physarum flux & Ollivier‑Ricci curvature)
- hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s2.py (Stylometry bilinear compatibility & brain‑map scaling)

Mathematical Bridge
-------------------
1. **Curvature (c)** is obtained from the Krampus feature set via the Ollivier‑Ricci
   average (Parent A).
2. **Compatibility scalar (s)** is the bilinear form `vᵀ P m` from the stylometry
   pipeline (Parent B).
3. **Flux (φ)** is the total absolute flux through a Physarum‑style conductance
   network (`φ = Σ |flux_i|`).
4. The unified scaling factor that drives the brain‑map is  

   `factor = s · c · φ · r`

   where **r** is an external reliability scalar (Parent B).  
   The brain‑map is then `brainmap = factor · I₂`.

The code below implements this bridge, provides three public functions that
demonstrate the hybrid operation, and ends with a smoke‑test that runs
without external data."""

import math
import random
import sys
import pathlib
import numpy as np
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Parent A – feature extraction, curvature, and Physarum flux utilities
# ----------------------------------------------------------------------
def extract_full_features() -> Dict[str, float]:
    """Return a deterministic feature dictionary used for curvature."""
    return {
        "visceral_ratio": 0.5,
        "tech_ratio": 0.3,
        "legal_osint_ratio": 0.2,
        "ledger_density": 0.1,
        "recursion_score": 0.4,
        "directive_ratio": 0.6,
        "target_density": 0.7,
        "forensic_shield_ratio": 0.8,
        "poetic_entropy": 0.9,
        "dissociative_index": 0.1,
        "wrath_velocity": 0.2,
        "bureaucratic_weaponization_index": 0.3,
        "resource_exhaustion_metric": 0.4,
    }

def krampus_ollivier_ricci_curvature(features: Dict[str, float]) -> float:
    """Simple Ollivier‑Ricci curvature: average of three core ratios."""
    viscera = features["visceral_ratio"]
    tech = features["tech_ratio"]
    legal_osint = features["legal_osint_ratio"]
    return (viscera + tech + legal_osint) / 3.0

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Conductance adaptation rule used in the Physarum network."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

# ----------------------------------------------------------------------
# Parent B – stylometry compatibility and brain‑map scaling
# ----------------------------------------------------------------------
def compute_compatibility(v: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """
    Bilinear compatibility: s = vᵀ P m.
    v is (n,), m is (2,), P is (n,2).
    """
    if v.ndim != 1 or m.ndim != 1 or P.ndim != 2:
        raise ValueError('Invalid dimensions for v, m or P')
    return float(v @ (P @ m))

def brainmap_matrix(factor: float) -> np.ndarray:
    """2‑D identity scaled by the supplied factor."""
    return factor * np.eye(2)

# ----------------------------------------------------------------------
# Hybrid Core – integration of both topologies
# ----------------------------------------------------------------------
def total_network_flux(conductances: np.ndarray,
                       edge_lengths: np.ndarray,
                       pressures: np.ndarray) -> float:
    """
    Compute the sum of absolute fluxes over all edges.
    - conductances: (E,)
    - edge_lengths: (E,)
    - pressures: (N,) where each edge connects node i to node i+1 (simple chain)
    """
    if not (conductances.shape == edge_lengths.shape):
        raise ValueError('conductances and edge_lengths must share shape')
    E = conductances.shape[0]
    if pressures.shape[0] < E + 1:
        raise ValueError('pressures must have at least E+1 entries for a chain topology')
    fluxes = np.empty(E, dtype=float)
    for i in range(E):
        fluxes[i] = flux(conductances[i], edge_lengths[i], pressures[i], pressures[i + 1])
    return float(np.sum(np.abs(fluxes)))

def hybrid_factor(v: np.ndarray,
                  m: np.ndarray,
                  P: np.ndarray,
                  conductances: np.ndarray,
                  edge_lengths: np.ndarray,
                  pressures: np.ndarray,
                  reliability: float) -> float:
    """
    Compute the unified scaling factor:
        factor = s · c · φ · r
    where
        s – stylometry compatibility,
        c – curvature from Krampus features,
        φ – total absolute flux,
        r – external reliability scalar.
    """
    # 1️⃣ Curvature from Parent A
    features = extract_full_features()
    c = krampus_ollivier_ricci_curvature(features)

    # 2️⃣ Compatibility scalar from Parent B
    s = compute_compatibility(v, m, P)

    # 3️⃣ Flux from the Physarum network (Parent A)
    phi = total_network_flux(conductances, edge_lengths, pressures)

    # 4️⃣ Combine with reliability
    factor = s * c * phi * reliability
    return factor

def hybrid_brainmap(v: np.ndarray,
                    m: np.ndarray,
                    P: np.ndarray,
                    conductances: np.ndarray,
                    edge_lengths: np.ndarray,
                    pressures: np.ndarray,
                    reliability: float) -> np.ndarray:
    """
    Produce the 2×2 brain‑map matrix using the hybrid factor.
    """
    factor = hybrid_factor(v, m, P, conductances, edge_lengths, pressures, reliability)
    return brainmap_matrix(factor)

def adapt_network(conductances: np.ndarray,
                  edge_lengths: np.ndarray,
                  pressures: np.ndarray,
                  dt: float = 1.0,
                  gain: float = 1.0,
                  decay: float = 0.05) -> np.ndarray:
    """
    Perform one adaptation step on the entire Physarum network using the
    flux‑based update rule from Parent A.
    Returns the updated conductance array.
    """
    E = conductances.shape[0]
    if not (edge_lengths.shape == (E,) and pressures.shape[0] >= E + 1):
        raise ValueError('Mismatched network dimensions')
    new_conductances = np.empty_like(conductances)
    for i in range(E):
        q = flux(conductances[i], edge_lengths[i], pressures[i], pressures[i + 1])
        new_conductances[i] = update_conductance(conductances[i], q, dt, gain, decay)
    return new_conductances

# ----------------------------------------------------------------------
# Example / Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Stylometry side (Parent B)
    n_features = 10
    np.random.seed(0)
    v = np.random.rand(n_features)                 # synthetic text feature vector
    m = np.array([0.7, 0.3])                        # model‑resource vector
    P = np.random.rand(n_features, 2)              # random projection matrix

    # 2️⃣ Physarum network side (Parent A)
    E = 5                                          # number of edges in a simple chain
    conductances = np.full(E, 1.0)                 # start with unit conductance
    edge_lengths = np.linspace(1.0, 2.0, E)        # varying lengths
    pressures = np.linspace(1.0, 0.0, E + 1)       # linear pressure drop

    # 3️⃣ Reliability scalar (Parent B)
    reliability = 0.85

    # Compute hybrid brain‑map
    brain = hybrid_brainmap(v, m, P, conductances, edge_lengths, pressures, reliability)
    print("Hybrid brain‑map matrix:\n", brain)

    # Perform a network adaptation step and recompute
    conductances = adapt_network(conductances, edge_lengths, pressures)
    brain_after = hybrid_brainmap(v, m, P, conductances, edge_lengths, pressures, reliability)
    print("\nBrain‑map after one conductance adaptation step:\n", brain_after)

    # Simple sanity check: factor should be non‑negative
    factor = hybrid_factor(v, m, P, conductances, edge_lengths, pressures, reliability)
    assert factor >= 0.0, "Hybrid factor must be non‑negative"
    print("\nHybrid scaling factor:", factor)