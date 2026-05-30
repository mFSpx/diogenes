# DARWIN HAMMER — match 4340, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m1964_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_fracti_m2589_s0.py (gen6)
# born: 2026-05-29T23:55:09Z

"""Hybrid Fusion Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m1964_s0.py (State‑Space / Epistemic framework)
- hybrid_hybrid_hybrid_krampu_hybrid_hybrid_fracti_m2589_s0.py (Graph curvature, sheaf cohomology, fractional binding, Gini weighting)

Mathematical Bridge:
The state‑transition matrix **A** of the State‑Space Model (Parent A) is interpreted as the weighted adjacency matrix of a graph.
Edges inherit the epistemic‑certainty weights derived from the flag set.  
This graph is equipped with a **Sheaf** (Parent B) whose node dimensions are the state dimensions and whose restriction maps are built from the entries of **A**.
We then
1. Apply a fractional power **α** to **A** (fractional binding from HDC).
2. Compute an Ollivier‑Ricci curvature approximation on the sheaf‑augmented graph.
3. Use the curvature values together with external reward estimates in a weighted Gini coefficient.

The resulting scalar blends dynamical, topological and statistical information into a single hybrid score.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & State‑Space utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
EP_FLAG_WEIGHTS = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.1,
    "SURE_MAYBE": 0.3,
}

@dataclass(frozen=True)
class Morphology:
    """Physical parameters of an engine endpoint."""
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """(length·width·height)^(1/3) / length."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    """(length+width)/(2·height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """A toy righting‑time index based on morphology."""
    inertia = m.mass * (m.length ** 2 + m.width ** 2) / 12.0
    return b * inertia / (k * neck_lever)

def epistemic_weight(flag: str) -> float:
    """Map an epistemic flag to a numeric confidence weight."""
    return EP_FLAG_WEIGHTS.get(flag.upper(), 0.0)

def build_state_matrix(m: Morphology, flag: str) -> np.ndarray:
    """
    Construct a 3×3 state‑transition matrix A from morphology and epistemic confidence.
    The diagonal encodes self‑dynamics (scaled by sphericity), the off‑diagonals encode
    cross‑couplings weighted by flatness and the epistemic flag.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    w = epistemic_weight(flag)

    A = np.array([
        [0.9 * sph,          w * flat * 0.1,  w * flat * 0.05],
        [w * flat * 0.1,    0.85 * sph,      w * flat * 0.07],
        [w * flat * 0.05,   w * flat * 0.07, 0.8 * sph]
    ], dtype=float)

    # Ensure stochastic stability (spectral radius < 1) by scaling if needed
    eigvals = np.linalg.eigvals(A)
    rho = max(abs(eigvals))
    if rho >= 1.0:
        A *= 0.9 / rho
    return A

# ----------------------------------------------------------------------
# Parent B – Sheaf, curvature & Gini utilities
# ----------------------------------------------------------------------
class Sheaf:
    """Simple sheaf over a graph defined by adjacency matrix."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edge_list)              # list of (u, v)
        self._restrictions: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[int, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[int, int], src_map: np.ndarray, dst_map: np.ndarray):
        u, v = edge
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: int, value: np.ndarray):
        self._sections[node] = np.asarray(value, dtype=float)

    def restriction(self, edge: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        return self._restrictions[edge]

def adjacency_to_sheaf(A: np.ndarray) -> Sheaf:
    """
    Convert a square adjacency matrix A into a Sheaf.
    Each node gets a dimension equal to the number of rows of A.
    For each non‑zero entry A[u, v] we create an edge (u, v) with a restriction map
    that scales the source vector by the edge weight.
    """
    n = A.shape[0]
    node_dims = {i: n for i in range(n)}
    edges = [(i, j) for i in range(n) for j in range(n) if abs(A[i, j]) > 1e-12]

    sheaf = Sheaf(node_dims, edges)

    for (i, j) in edges:
        w = A[i, j]
        # Restriction from i to j: multiply by w, identity in opposite direction
        src_map = np.eye(n) * w
        dst_map = np.eye(n)        # trivial restriction back
        sheaf.set_restriction((i, j), src_map, dst_map)

    # Initialise a random section at each node (could be state vector)
    for i in range(n):
        sheaf.set_section(i, np.random.randn(n))
    return sheaf

def ollivier_ricci_curvature(sheaf: Sheaf) -> Dict[Tuple[int, int], float]:
    """
    Very coarse Ollivier‑Ricci curvature approximation:
    κ(u,v) = 1 - (||μ_u - μ_v||_1) / d(u,v)
    where μ_u is the normalized restriction map from u to its neighbours,
    and d(u,v) is the edge weight (taken from the restriction src_map diagonal).
    """
    curvature = {}
    for (u, v) in sheaf.edges:
        src_map, _ = sheaf.restriction((u, v))
        # Edge weight as average absolute diagonal entry
        d_uv = np.mean(np.abs(np.diag(src_map))) + 1e-12
        # Probability measures μ_u, μ_v as normalized row sums of restriction maps
        mu_u = np.abs(src_map).sum(axis=1)
        mu_u = mu_u / (mu_u.sum() + 1e-12)
        # For reverse direction we use the restriction of (v,u) if it exists,
        # otherwise fallback to identity (no transport)
        if (v, u) in sheaf._restrictions:
            rev_map, _ = sheaf.restriction((v, u))
            mu_v = np.abs(rev_map).sum(axis=1)
        else:
            mu_v = np.ones_like(mu_u)
        mu_v = mu_v / (mu_v.sum() + 1e-12)

        distance = np.linalg.norm(mu_u - mu_v, 1)
        curvature[(u, v)] = 1.0 - distance / d_uv
    return curvature

def weighted_gini(values: List[float], weights: List[float]) -> float:
    """
    Compute the Gini coefficient with sample weights.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")
    # Sort by value
    sorted_idx = np.argsort(values)
    sorted_vals = np.array(values)[sorted_idx]
    sorted_wts = np.array(weights)[sorted_idx]

    cumw = np.cumsum(sorted_wts)
    cumwv = np.cumsum(sorted_vals * sorted_wts)
    total_w = cumw[-1]
    total_wv = cumwv[-1]

    if total_w == 0 or total_wv == 0:
        return 0.0

    # Gini formula for weighted data
    B = np.sum(cumwv * sorted_wts) / (total_w * total_wv)
    return 1.0 - 2.0 * B

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def fractional_power_binding(A: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute A^α via eigen‑decomposition.
    For non‑symmetric A we use the real Schur decomposition approximation
    (here we fall back to eig for simplicity, assuming A is diagonalizable).
    """
    eigvals, eigvecs = np.linalg.eig(A)
    D_alpha = np.diag(np.real(eigvals) ** alpha)
    A_alpha = eigvecs @ D_alpha @ np.linalg.inv(eigvecs)
    # Remove possible complex residuals due to numerical error
    return np.real_if_close(A_alpha, tol=1e-8)

def hybrid_score(morph: Morphology, flag: str, rewards: List[float], alpha: float = 0.5) -> float:
    """
    End‑to‑end hybrid computation:
    1. Build state matrix A.
    2. Apply fractional power α.
    3. Build sheaf from A^α.
    4. Compute curvature on the sheaf.
    5. Combine curvature (averaged) with rewards via weighted Gini.
    Returns a scalar score.
    """
    # 1‑2. State matrix and fractional binding
    A = build_state_matrix(morph, flag)
    A_alpha = fractional_power_binding(A, alpha)

    # 3. Sheaf construction
    sheaf = adjacency_to_sheaf(A_alpha)

    # 4. Curvature
    curv = ollivier_ricci_curvature(sheaf)
    avg_curv = np.mean(list(curv.values())) if curv else 0.0

    # 5. Weighted Gini: curvature acts as weight for each reward
    # Align lengths: replicate avg_curv for each reward
    weights = [1.0 + avg_curv for _ in rewards]  # baseline 1 plus curvature influence
    gini = weighted_gini(rewards, weights)

    # Final blended score (higher curvature & lower inequality => higher score)
    score = avg_curv * (1.0 - gini) + np.mean(rewards) * 0.1
    return float(score)

def simulate_hybrid_step(num_steps: int = 5) -> List[float]:
    """
    Run a short simulation producing a list of hybrid scores.
    Each step randomly perturbs morphology and epistemic flag.
    """
    scores = []
    for _ in range(num_steps):
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.2, 1.0),
            mass=random.uniform(10.0, 100.0)
        )
        flag = random.choice(EPISTEMIC_FLAGS)
        # Simulated external rewards (e.g., RL returns)
        rewards = [random.expovariate(1.0) for _ in range(4)]
        score = hybrid_score(morph, flag, rewards, alpha=0.6)
        scores.append(score)
    return scores

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Fixed seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example morphology and flag
    example_morph = Morphology(length=1.2, width=0.8, height=0.5, mass=45.0)
    example_flag = "PROBABLE"
    example_rewards = [0.9, 1.2, 0.4, 1.0]

    print("Hybrid score (single run):", hybrid_score(example_morph, example_flag, example_rewards))

    print("Simulation of 5 steps:", simulate_hybrid_step(5))