# DARWIN HAMMER — match 3443, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1599_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py (gen4)
# born: 2026-05-29T23:50:17Z

"""Hybrid Physarum‑Sheaf Curvature Algorithm
Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1599_s3.py
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py

Mathematical Bridge
Both parent algorithms manipulate a graph structure.  
*Physarum‑Infotaxis* evolves edge conductances `c_ij` via a flux
`Φ_ij = c_ij / ℓ_ij (p_i‑p_j)` and updates `c_ij` with a term proportional
to the absolute flux, weighted by an **information density** `ρ` derived from
feature vectors.  
*Sheaf‑Cohomology* attaches to each edge a linear restriction map
`R_ij : ℝ^{d_i} → ℝ^{d_j}` and to each node a section `s_i`. Consistency of a
section across an edge is measured by `‖R_ij s_i – s_j‖`. Edge weights can be
modulated by a **semantic similarity** scalar `σ_ij`.

The hybrid unifies these by:
1. Computing a scalar information density `ρ` from the Physarum feature vector
   (e.g. normalized L2‑norm).
2. Scaling each restriction map `R_ij` by `ρ·σ_ij`, thus letting the
   information‑richness of the environment influence sheaf consistency.
3. Updating conductances with the usual Physarum rule but adding a curvature‑
   derived bias `κ_ij` obtained from a (mock) Ollivier‑Ricci feature vector.
   The bias is injected as `c_ij ← c_ij + η·ρ·κ_ij·|Φ_ij|`.

The three core functions below implement:
- `flux` and `update_conductance` (Physarum core).
- `apply_sheaf_restriction` (Sheaf core with the information‑semantic bridge).
- `hybrid_step` (one iteration that couples both dynamics).

All operations are pure NumPy; no external scientific libraries are required.
"""

import numpy as np
import random
import math
import sys
import pathlib
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Physarum core utilities
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Ohmic flux on an edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05,
                       curvature_bias: float = 0.0, rho: float = 1.0) -> float:
    """
    Physarum conductance update with optional curvature bias.
    curvature_bias is added linearly after the usual gain·|q| term.
    """
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    inc = dt * (gain * abs(q) + curvature_bias * rho - decay * conductance)
    return max(0.0, conductance + inc)


# ----------------------------------------------------------------------
# Sheaf core utilities
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf on a directed graph.
    - node_dims: dict {node: dimension}
    - edges: list of (u, v, length) tuples
    - restrictions[(u,v)] = (src_map, dst_map) where maps are NumPy arrays
    - sections[node] = vector in ℝ^{dim(node)}
    """
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)                     # (u, v, length)
        self._restrictions = {}                      # (u,v) → (src_map, dst_map)
        self.sections = {n: np.zeros(d) for n, d in self.node_dims.items()}

    def set_restriction(self, edge, src_map, dst_map):
        """Store linear maps for edge (u,v)."""
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u] or dst_map.shape[1] != self.node_dims[v]:
            raise ValueError('Restriction map dimensions do not match node dimensions')
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value):
        """Assign a section (vector) to a node."""
        val = np.array(value, dtype=float)
        if val.shape[0] != self.node_dims[node]:
            raise ValueError('Section dimension mismatch')
        self.sections[node] = val

    def edge_consistency(self, u, v):
        """‖R_uv s_u – s_v‖₂, where R_uv = dst_map @ src_map⁻¹ (simplified)."""
        if (u, v) not in self._restrictions:
            raise KeyError(f'No restriction defined for edge {(u, v)}')
        src_map, dst_map = self._restrictions[(u, v)]
        # Simplified restriction: apply src_map then dst_map
        transformed = dst_map @ (src_map @ self.sections[u])
        diff = transformed - self.sections[v]
        return np.linalg.norm(diff)


# ----------------------------------------------------------------------
# Feature extraction (mocked, deterministic for reproducibility)
# ----------------------------------------------------------------------
def extract_features(seed: int = None) -> dict:
    """Return a deterministic pseudo‑random feature vector."""
    rng = random.Random(seed)
    return {
        "info_density": rng.random(),
        "curvature_x": rng.random(),
        "curvature_y": rng.random(),
        "curvature_z": rng.random(),
    }


def information_density(features: dict) -> float:
    """Scalar information density ρ ∈ (0,1] derived from feature vector."""
    vals = np.array(list(features.values()))
    norm = np.linalg.norm(vals)
    return min(1.0, norm / (np.sqrt(len(vals))))


def curvature_vector(features: dict) -> np.ndarray:
    """Return a 3‑D curvature bias vector κ."""
    return np.array([features.get("curvature_x", 0.0),
                     features.get("curvature_y", 0.0),
                     features.get("curvature_z", 0.0)])


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def apply_sheaf_restriction(sheaf: Sheaf, rho: float, semantic_sim: dict):
    """
    Scale each restriction map by the product ρ·σ_uv where σ_uv is the
    semantic similarity for edge (u,v). This implements the bridge from
    information density to sheaf consistency.
    """
    for (u, v) in sheaf._restrictions:
        sigma = semantic_sim.get((u, v), 1.0)  # default similarity = 1
        scale = rho * sigma
        src_map, dst_map = sheaf._restrictions[(u, v)]
        sheaf._restrictions[(u, v)] = (src_map * scale, dst_map * scale)


def hybrid_step(sheaf: Sheaf,
                pressures: dict,
                edge_conductances: dict,
                edge_lengths: dict,
                semantic_sim: dict,
                dt: float = 1.0,
                gain: float = 1.0,
                decay: float = 0.05,
                curvature_gain: float = 0.1,
                feature_seed: int = None):
    """
    Perform one iteration of the coupled Physarum‑Sheaf system.
    1. Extract features → ρ and κ.
    2. Scale sheaf restrictions (information‑semantic bridge).
    3. Compute fluxes and update conductances with curvature bias.
    4. Update node sections by a simple gradient step toward consistency.
    """
    # ----- 1. Feature‑derived scalars -----
    feats = extract_features(feature_seed)
    rho = information_density(feats)                     # information density
    kappa = curvature_vector(feats)                      # curvature bias vector

    # ----- 2. Sheaf restriction scaling -----
    apply_sheaf_restriction(sheaf, rho, semantic_sim)

    # ----- 3. Physarum flux & conductance update -----
    new_conductances = {}
    for (u, v, length) in sheaf.edges:
        c = edge_conductances.get((u, v), 1.0)
        p_u = pressures.get(u, 0.0)
        p_v = pressures.get(v, 0.0)

        q = flux(c, length, p_u, p_v)
        # Use a scalar projection of κ onto the edge direction (mocked by dot with unit vector)
        direction = np.array([1.0, 0.0, 0.0])  # placeholder direction
        curvature_bias = curvature_gain * float(np.dot(kappa, direction))

        c_new = update_conductance(c, q, dt=dt, gain=gain,
                                   decay=decay,
                                   curvature_bias=curvature_bias,
                                   rho=rho)
        new_conductances[(u, v)] = c_new

    # ----- 4. Section relaxation toward sheaf consistency -----
    for (u, v, _) in sheaf.edges:
        # Compute inconsistency
        inc = sheaf.edge_consistency(u, v)
        # Simple gradient descent on the destination node section
        if inc > 0:
            grad = (sheaf.sections[v] - sheaf.sections[u]) / inc
            sheaf.sections[v] = sheaf.sections[v] - 0.1 * rho * grad  # step size scaled by ρ

    return new_conductances, rho, kappa


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph with 3 nodes
    node_dims = {"A": 2, "B": 2, "C": 2}
    edges = [("A", "B", 1.0), ("B", "C", 1.5), ("C", "A", 2.0)]

    # Initialise sheaf
    sheaf = Sheaf(node_dims, edges)
    # Random linear restrictions (2×2 matrices)
    for (u, v, _) in edges:
        src = np.eye(2) + 0.1 * np.random.randn(2, 2)
        dst = np.eye(2) + 0.1 * np.random.randn(2, 2)
        sheaf.set_restriction((u, v), src, dst)

    # Random sections
    for n in node_dims:
        sheaf.set_section(n, np.random.randn(node_dims[n]))

    # Pressures at nodes (Physarum)
    pressures = {"A": 1.0, "B": 0.0, "C": -0.5}

    # Edge conductances and lengths
    edge_conductances = { (u, v): 1.0 for (u, v, _) in edges }
    edge_lengths = { (u, v): l for (u, v, l) in edges }

    # Semantic similarity matrix (mocked)
    semantic_sim = {("A", "B"): 0.9, ("B", "C"): 0.7, ("C", "A"): 0.5}

    # Run a few hybrid steps
    for step in range(5):
        edge_conductances, rho, kappa = hybrid_step(
            sheaf,
            pressures,
            edge_conductances,
            edge_lengths,
            semantic_sim,
            dt=0.5,
            gain=1.2,
            decay=0.03,
            curvature_gain=0.2,
            feature_seed=step  # deterministic per step
        )
        print(f"Step {step+1}: rho={rho:.3f}, κ={kappa}, conductances={list(edge_conductances.values())}")

    print("Final node sections:")
    for n, sec in sheaf.sections.items():
        print(f"  {n}: {sec}")