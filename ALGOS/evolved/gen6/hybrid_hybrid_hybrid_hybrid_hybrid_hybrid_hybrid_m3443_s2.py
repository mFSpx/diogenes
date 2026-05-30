# DARWIN HAMMER — match 3443, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1599_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m536_s2.py (gen4)
# born: 2026-05-29T23:50:17Z

import numpy as np
import random
from typing import Dict, Tuple, List


# ----------------------------------------------------------------------
# Feature extraction (deterministic)
# ----------------------------------------------------------------------
def _rng(seed: int = None) -> random.Random:
    return random.Random(seed)


def extract_features(seed: int = None) -> dict:
    """Deterministic pseudo‑random feature vector."""
    rng = _rng(seed)
    return {
        "info_density": rng.random(),
        "curvature_x": rng.random(),
        "curvature_y": rng.random(),
        "curvature_z": rng.random(),
    }


def information_density(features: dict) -> float:
    """Scalar information density ρ ∈ (0,1] derived **only** from the info_density entry."""
    val = float(features.get("info_density", 0.0))
    return max(1e-6, min(1.0, val))


def curvature_vector(features: dict) -> np.ndarray:
    """3‑D curvature bias vector κ."""
    return np.array([
        features.get("curvature_x", 0.0),
        features.get("curvature_y", 0.0),
        features.get("curvature_z", 0.0),
    ], dtype=float)


# ----------------------------------------------------------------------
# Physarum core utilities
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Ohmic flux on an edge."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float,
                       gain: float, decay: float,
                       curvature_bias: float, rho: float) -> float:
    """Physarum conductance update with curvature bias."""
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    inc = dt * (gain * abs(q) + curvature_bias * rho - decay * conductance)
    return max(0.0, conductance + inc)


# ----------------------------------------------------------------------
# Sheaf core utilities
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf on a directed graph with explicit per‑edge scaling.
    - node_dims: {node: dimension}
    - edges: list of (u, v, length) tuples
    - node_positions: {node: np.ndarray} for geometric direction.
    - restrictions[(u,v)] = (src_map, dst_map) where maps are NumPy arrays.
    - scalings[(u,v)] = scalar factor applied to the restriction operator.
    - sections[node] = vector in ℝ^{dim(node)}.
    """

    def __init__(self,
                 node_dims: Dict,
                 edges: List[Tuple],
                 node_positions: Dict):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)                     # (u, v, length)
        self.node_positions = {
            n: np.asarray(p, dtype=float) for n, p in node_positions.items()
        }
        self._restrictions: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self.scalings: Dict[Tuple, float] = {}
        self.sections: Dict = {n: np.zeros(d) for n, d in self.node_dims.items()}

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple, src_map, dst_map):
        """Store linear maps for edge (u, v)."""
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension mismatch")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension mismatch")
        self._restrictions[(u, v)] = (src_map, dst_map)
        self.scalings[(u, v)] = 1.0

    def set_section(self, node, value):
        """Assign a section (vector) to a node."""
        val = np.asarray(value, dtype=float)
        if val.shape[0] != self.node_dims[node]:
            raise ValueError("section dimension mismatch")
        self.sections[node] = val

    def edge_direction(self, u, v) -> np.ndarray:
        """Unit vector from u to v using stored positions."""
        pu = self.node_positions[u]
        pv = self.node_positions[v]
        vec = pv - pu
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.zeros_like(vec)
        return vec / norm

    def edge_consistency(self, u, v) -> float:
        """‖scale·R_uv s_u – s_v‖₂ where scale = self.scalings[(u,v)]."""
        if (u, v) not in self._restrictions:
            raise KeyError(f"No restriction defined for edge {(u, v)}")
        src_map, dst_map = self._restrictions[(u, v)]
        scale = self.scalings.get((u, v), 1.0)
        transformed = scale * (dst_map @ (src_map @ self.sections[u]))
        diff = transformed - self.sections[v]
        return np.linalg.norm(diff)


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def apply_sheaf_scaling(sheaf: Sheaf,
                        rho: float,
                        semantic_sim: Dict[Tuple, float]) -> None:
    """
    Update per‑edge scaling factors by ρ·σ_uv.
    Existing scalings are multiplied, preserving the original maps.
    """
    for (u, v) in sheaf._restrictions:
        sigma = semantic_sim.get((u, v), 1.0)
        sheaf.scalings[(u, v)] = rho * sigma


def hybrid_step(sheaf: Sheaf,
                pressures: Dict,
                edge_conductances: Dict[Tuple, float],
                semantic_sim: Dict[Tuple, float],
                dt: float = 1.0,
                gain: float = 1.0,
                decay: float = 0.05,
                curvature_gain: float = 0.1,
                feature_seed: int = None) -> Tuple[Dict[Tuple, float], Dict]:
    """
    Perform one iteration of the coupled Physarum‑Sheaf system.

    Returns
    -------
    new_conductances : dict mapping edge → updated conductance
    new_sections     : dict mapping node → updated section vector
    """
    # --------------------------------------------------------------
    # 1. Feature‑derived scalars
    # --------------------------------------------------------------
    feats = extract_features(feature_seed)
    rho = information_density(feats)                 # information density
    kappa = curvature_vector(feats)                  # curvature bias vector

    # --------------------------------------------------------------
    # 2. Sheaf scaling (information‑semantic bridge)
    # --------------------------------------------------------------
    apply_sheaf_scaling(sheaf, rho, semantic_sim)

    # --------------------------------------------------------------
    # 3. Physarum flux & conductance update
    # --------------------------------------------------------------
    new_conductances = {}
    for (u, v, length) in sheaf.edges:
        c = edge_conductances.get((u, v), 1.0)
        p_u = pressures.get(u, 0.0)
        p_v = pressures.get(v, 0.0)

        q = flux(c, length, p_u, p_v)

        # curvature bias projected onto geometric edge direction
        direction = sheaf.edge_direction(u, v)          # unit vector
        curvature_proj = float(np.dot(kappa, direction))
        curvature_bias = curvature_gain * curvature_proj

        new_c = update_conductance(c, q, dt, gain, decay,
                                   curvature_bias, rho)
        new_conductances[(u, v)] = new_c

    # --------------------------------------------------------------
    # 4. Section update – gradient descent on total inconsistency
    # --------------------------------------------------------------
    # Compute per‑node gradient as sum of incoming/outgoing consistency errors
    grad = {n: np.zeros(sheaf.node_dims[n]) for n in sheaf.node_dims}
    lr = 0.1 * rho  # learning rate modulated by information density

    for (u, v, _) in sheaf.edges:
        # forward consistency error
        err_fwd = (sheaf.scalings[(u, v)] *
                   (sheaf._restrictions[(u, v)][1] @
                    (sheaf._restrictions[(u, v)][0] @ sheaf.sections[u])) -
                   sheaf.sections[v])
        grad[u] += sheaf._restrictions[(u, v)][0].T @ \
                  (sheaf._restrictions[(u, v)][1].T @ err_fwd)
        grad[v] -= err_fwd

        # backward consistency (if reverse edge exists)
        if (v, u) in sheaf._restrictions:
            err_rev = (sheaf.scalings[(v, u)] *
                       (sheaf._restrictions[(v, u)][1] @
                        (sheaf._restrictions[(v, u)][0] @ sheaf.sections[v])) -
                       sheaf.sections[u])
            grad[v] += sheaf._restrictions[(v, u)][0].T @ \
                      (sheaf._restrictions[(v, u)][1].T @ err_rev)
            grad[u] -= err_rev

    new_sections = {}
    for n, sec in sheaf.sections.items():
        new_sec = sec - lr * grad[n]
        new_sections[n] = new_sec
        sheaf.sections[n] = new_sec  # update in‑place for next iteration

    return new_conductances, new_sections


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    node_dims = {"A": 2, "B": 2, "C": 2}
    edges = [("A", "B", 1.0), ("B", "C", 1.5), ("C", "A", 2.0)]
    positions = {"A": (0, 0, 0), "B": (1, 0, 0), "C": (0, 1, 0)}

    sheaf = Sheaf(node_dims, edges, positions)

    # Random linear maps (deterministic for demo)
    rng = np.random.default_rng(42)
    for (u, v, _) in edges:
        src = rng.normal(size=(2, node_dims[u]))
        dst = rng.normal(size=(2, node_dims[v]))
        sheaf.set_restriction((u, v), src, dst)

    # Initialise sections
    for n in node_dims:
        sheaf.set_section(n, rng.normal(size=node_dims[n]))

    pressures = {"A": 1.0, "B": 0.0, "C": -1.0}
    conductances = {("A", "B"): 1.0, ("B", "C"): 1.0, ("C", "A"): 1.0}
    semantic_sim = {("A", "B"): 0.9, ("B", "C"): 0.7, ("C", "A"): 0.5}

    new_c, new_s = hybrid_step(sheaf,
                               pressures,
                               conductances,
                               semantic_sim,
                               dt=0.5,
                               gain=1.2,
                               decay=0.03,
                               curvature_gain=0.15,
                               feature_seed=7)

    print("Updated conductances:", new_c)
    print("Updated sections:", new_s)