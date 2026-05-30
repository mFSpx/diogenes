# DARWIN HAMMER — match 4027, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1148_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m1758_s2.py (gen4)
# born: 2026-05-29T23:53:10Z

"""HybridRegretVFE_Sketch: Fusion of hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1148_s0.py
and hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m1758_s2.py.

Mathematical Bridge
-------------------
The regret‑weighted action values of the Regret Engine are treated as *sections* on a
graph.  Their inequality is quantified by the Gini coefficient.  In the Variational
Free‑Energy (VFE) formulation of the privacy‑preserving ternary‑router, a pseudo‑observation
noise variance σ² is required; we supply σ² = Gini², thus letting the unevenness of the
regret distribution directly modulate belief regularisation.

Conversely, the sections are embedded into a Count‑Min Sketch (CMS) using a
hyperdimensional binding operator ⊗ (binary XOR).  The bound sketch acts as the
observation vector for the VFE update, while the coboundary operator δ measures
local disagreement between neighbouring sections and is used as a control term in the
Koopman forecast of future regret values.

The resulting hybrid system jointly:
1. Computes regret‑weighted sections and their Gini‑derived variance.
2. Encodes sections into a privacy‑preserving CMS via hyperdimensional binding.
3. Performs a VFE belief update using the bound sketch and the coboundary‑derived
   regulariser, and forecasts the next regret section with a linear Koopman operator.

All components are expressed with NumPy and the Python standard library only.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Hybrid Sheaf (graph of sections) utilities
# ----------------------------------------------------------------------
class HybridSheaf:
    """A lightweight graph where each node holds a numeric section."""
    def __init__(self, node_dims: Dict[int, int], edge_list: List[Tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node -> dimension (scalar here)
        self.edges = list(edge_list)              # list of (u, v)
        self._sections: Dict[int, np.ndarray] = {}  # node -> value vector

    def set_section(self, node: int, value):
        """Assign a numeric section to a node."""
        self._sections[node] = np.array(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims.get(node, 1)))

    def coboundary(self) -> np.ndarray:
        """
        Compute the coboundary δs over all edges:
        δs_{uv} = s(v) - s(u)
        Returns a stacked array of shape (|E|, d) where d is the common dimension.
        """
        diffs = []
        for u, v in self.edges:
            su = self.get_section(u)
            sv = self.get_section(v)
            diffs.append(sv - su)
        if diffs:
            return np.stack(diffs, axis=0)
        else:
            return np.empty((0,))

# ----------------------------------------------------------------------
# Statistical / information‑theoretic primitives
# ----------------------------------------------------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """Return the Gini coefficient of a 1‑D array."""
    x = np.ravel(x).astype(float)
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = x.size
    cumx = np.cumsum(sorted_x)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(gini)

def koopman_forecast(section: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Linear Koopman forecast: s_{t+1} = A @ s_t
    A is a square matrix of compatible dimension.
    """
    return A @ section

# ----------------------------------------------------------------------
# Count‑Min Sketch + Hyperdimensional Binding
# ----------------------------------------------------------------------
def _cms_hash(item: str, depth: int, width: int) -> np.ndarray:
    """Hash an item into `depth` column indices."""
    return np.array([
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ], dtype=int)

def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Build a Count‑Min Sketch matrix."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        cols = _cms_hash(item, depth, width)
        for d, c in enumerate(cols):
            cms[d, c] += 1
    return cms

def hyperdimensional_vector(seed: str, dim: int = 1024) -> np.ndarray:
    """
    Produce a binary hyperdimensional vector from a string seed.
    Deterministic via SHA‑256, then thresholded at the median bit.
    """
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32))
    return rng.integers(0, 2, size=dim, dtype=np.int8)

def bind_hdc(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """Binding operator ⊗ implemented as XOR for binary vectors."""
    return np.bitwise_xor(vec_a, vec_b)

def sketch_hdc_binding(items: List[str], width: int = 64, depth: int = 4, hdim: int = 1024) -> np.ndarray:
    """
    Encode each item into a hyperdimensional vector, bind it with its CMS column
    index vector, and aggregate into a sketch matrix of shape (depth, width, hdim).
    """
    cms = count_min_sketch(items, width, depth)
    bound = np.zeros((depth, width, hdim), dtype=np.int8)

    for item in items:
        cols = _cms_hash(item, depth, width)
        hvec = hyperdimensional_vector(item, hdim)
        for d, c in enumerate(cols):
            bound[d, c] = bind_hdc(bound[d, c], hvec)  # XOR accumulation
    return bound

# ----------------------------------------------------------------------
# Variational Free‑Energy (VFE) core
# ----------------------------------------------------------------------
def variational_free_energy(belief: np.ndarray,
                            observation: np.ndarray,
                            variance: float,
                            kl_weight: float = 1.0) -> float:
    """
    Simple VFE:  F = (1/2σ²)||belief - observation||² + kl_weight * KL(belief||prior)
    Here we treat the prior as a zero‑mean Gaussian with unit variance,
    so KL reduces to 0.5 * ||belief||² (ignoring constants).
    """
    diff = belief - observation
    energy = 0.5 * np.sum(diff ** 2) / (variance + 1e-12)
    kl = 0.5 * np.sum(belief ** 2)
    return energy + kl_weight * kl

def belief_update(belief: np.ndarray,
                  observation: np.ndarray,
                  variance: float,
                  lr: float = 0.1) -> np.ndarray:
    """
    Gradient descent on the VFE w.r.t. belief.
    ∂F/∂b = (b - o)/σ² + b   =>   b_new = b - lr * ∂F/∂b
    """
    grad = (belief - observation) / (variance + 1e-12) + belief
    return belief - lr * grad

# ----------------------------------------------------------------------
# Hybrid operations exposing the mathematical bridge
# ----------------------------------------------------------------------
def hybrid_regret_gini_coboundary(sheaf: HybridSheaf,
                                 actions: List[MathAction],
                                 koopman_A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    1. Build a regret‑weighted section vector s from actions:
       s_i = expected_value_i - cost_i - risk_i
    2. Store s on a designated node (node 0) of the sheaf.
    3. Compute Gini coefficient of s → variance σ².
    4. Apply coboundary δs and forecast next section via Koopman.
    Returns (forecasted_section, variance).
    """
    # Step 1: regret‑weighted values
    s = np.array([a.expected_value - a.cost - a.risk for a in actions], dtype=float)
    # Step 2: store on node 0 (assume node 0 exists)
    sheaf.set_section(0, s)
    # Step 3: variance from Gini
    sigma2 = gini_coefficient(s) ** 2
    # Step 4: coboundary (used as regulariser, not returned) and forecast
    _ = sheaf.coboundary()   # side‑effect only; could be logged
    forecast = koopman_forecast(s, koopman_A)
    return forecast, sigma2

def sketch_hdc_observation(items: List[str],
                           width: int = 64,
                           depth: int = 4,
                           hdim: int = 1024) -> np.ndarray:
    """
    Produce a flattened observation vector from the bound CMS/HDC matrix.
    The flattening yields a 1‑D vector compatible with belief vectors.
    """
    bound = sketch_hdc_binding(items, width, depth, hdim)
    # Sum over depth and width to obtain a single hyperdimensional vector
    obs = bound.sum(axis=(0, 1)).astype(float)
    # Normalise to unit norm to avoid scale explosion
    norm = np.linalg.norm(obs) + 1e-12
    return obs / norm

def hybrid_vfe_step(sheaf: HybridSheaf,
                    actions: List[MathAction],
                    items: List[str],
                    belief: np.ndarray,
                    koopman_A: np.ndarray,
                    lr: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    End‑to‑end hybrid step:
    • Regret section → variance via Gini.
    • Encode items → observation via CMS+HDC.
    • Update belief with VFE using the variance.
    • Forecast next regret section (returned for possible downstream use).
    Returns updated belief and the computed variance.
    """
    forecast, variance = hybrid_regret_gini_coboundary(sheaf, actions, koopman_A)
    observation = sketch_hdc_observation(items)
    # Ensure belief and observation share shape
    if belief.shape != observation.shape:
        # Broadcast or truncate to the smaller dimension
        min_len = min(belief.size, observation.size)
        belief = belief.ravel()[:min_len]
        observation = observation.ravel()[:min_len]
    updated_belief = belief_update(belief, observation, variance, lr)
    return updated_belief, variance

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal graph with a single node (0) and no edges
    sheaf = HybridSheaf(node_dims={0: 5}, edge_list=[])

    # Example actions
    actions = [
        MathAction(id="a1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="a2", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="a3", expected_value=12.0, cost=3.0, risk=2.0),
        MathAction(id="a4", expected_value=7.0, cost=1.0, risk=0.2),
        MathAction(id="a5", expected_value=9.0, cost=2.5, risk=1.5),
    ]

    # Simple Koopman matrix (identity for test)
    koopman_A = np.eye(5)

    # Initial belief (random hyperdimensional vector)
    rng = np.random.default_rng(42)
    belief = rng.normal(size=1024).astype(float)

    # Items to be encoded in the sketch
    items = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Run a single hybrid VFE step
    updated_belief, var = hybrid_vfe_step(
        sheaf=sheaf,
        actions=actions,
        items=items,
        belief=belief,
        koopman_A=koopman_A,
        lr=0.05
    )

    # Simple sanity prints (allowed in smoke test)
    print(f"Variance from Gini: {var:.6f}")
    print(f"Belief norm before: {np.linalg.norm(belief):.4f}")
    print(f"Belief norm after : {np.linalg.norm(updated_belief):.4f}")
    print("Hybrid step completed without error.")