# DARWIN HAMMER — match 5146, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m42_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1601_s0.py (gen6)
# born: 2026-05-30T00:00:07Z

"""
Hybrid Fusion Module: SheafVRAM + Morphology‑Decision Engine

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m42_s0.py (SheafVRAM with energy‑based
  restrictions)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1601_s0.py (Morphology‑based recovery
  priority, regret‑weighted sigmoid, MinHash Jaccard and SSIM similarity)

Mathematical Bridge:
Both parents compute a scalar “quality” measure that guides updates:
* SheafVRAM evaluates an **energy**  E = Σ‖R_uv·s_u – L_uv·s_v‖² over all directed edges.
* The decision engine evaluates a **score** S = p·g(R)·(1+J)·(1+SSIM)·dance,
  where p is a morphology‑derived priority and g is a sigmoid of the regret term R.

The hybrid algorithm fuses them by treating the energy as a *regularizer* for the
decision score, yielding a combined objective

    H = E · S

The gradient of H with respect to node sections drives a unified update rule that
simultaneously respects sheaf consistency and morphology‑driven priorities.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple, Any

# ----------------------------------------------------------------------
# Core data structures from Parent A
# ----------------------------------------------------------------------
class SheafVRAM:
    """
    Cellular sheaf on a directed graph with VRAM scheduling information.
    Nodes hold vector sections; edges carry paired restriction matrices.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]], vram_slots: List[int]):
        self.node_dims = node_dims                     # dimension of each node's vector space
        self.edges = edges                             # list of directed edges (u, v)
        self.vram_slots = vram_slots
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self._vram_plan: Dict[int, Any] = {}

    def set_restriction(self, edge: Tuple[Any, Any], src_map: np.ndarray, dst_map: np.ndarray) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Set the section (vector) at the given node."""
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("section dimension mismatch for node")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections[node]

    def set_vram_plan(self, slot: int, plan: Any) -> None:
        """Associate a VRAM allocation plan with a slot."""
        self._vram_plan[slot] = plan

# ----------------------------------------------------------------------
# Core data structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """A simplified right‑ing time proxy used as part of the priority."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * (k * fi + neck_lever)

# ----------------------------------------------------------------------
# Helper similarity functions (placeholders for MinHash & SSIM)
# ----------------------------------------------------------------------
def minhash_jaccard(sig_a: Iterable[int], sig_b: Iterable[int]) -> float:
    """Very rough Jaccard similarity using set intersection over union."""
    set_a, set_b = set(sig_a), set(sig_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Simplified Structural Similarity:
    1 - normalized L2 distance, clipped to [0,1].
    """
    if img_a.shape != img_b.shape:
        raise ValueError("SSIM inputs must share shape")
    diff = np.linalg.norm(img_a - img_b)
    norm = np.linalg.norm(img_a) + np.linalg.norm(img_b) + 1e-12
    return max(0.0, 1.0 - diff / norm)

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_energy(sheaf: SheafVRAM) -> float:
    """
    Compute the total sheaf energy:
        E = Σ_{(u,v)} ‖R_uv·s_u – L_uv·s_v‖²
    where (R_uv, L_uv) are the restriction matrices for edge u→v.
    """
    total = 0.0
    for (u, v) in sheaf.edges:
        src_map, dst_map = sheaf._restrictions[(u, v)]
        s_u = sheaf.get_section(u)
        s_v = sheaf.get_section(v)
        diff = src_map @ s_u - dst_map @ s_v
        total += float(np.linalg.norm(diff) ** 2)
    return total

def morphology_priority(m: Morphology) -> float:
    """
    Derive a recovery priority p from morphology.
    Combines sphericity (higher → more streamlined) and right‑ing time index.
    Normalized to (0,1) via a logistic map.
    """
    sph = sphericity_index(m.length, m.width, m.height)
    rt = righting_time_index(m)
    raw = 0.6 * sph + 0.4 * (1.0 / (1.0 + rt))  # smaller rt → higher priority
    return 1.0 / (1.0 + math.exp(-10 * (raw - 0.5)))  # steep logistic

def decision_score(
    morph: Morphology,
    expected: float,
    cost: float,
    risk: float,
    counterfactual: float,
    sig: Iterable[int],
    sig_ref: Iterable[int],
    payload: np.ndarray,
    prototype: np.ndarray,
    dance: float,
) -> float:
    """
    Compute the decision engine score S:
        S = p * g(R) * (1+J) * (1+SSIM) * dance
    """
    # Regret term and sigmoid weighting
    R = expected - cost - risk + counterfactual
    g = 1.0 / (1.0 + math.exp(-R))  # sigmoid

    # Similarities
    J = minhash_jaccard(sig, sig_ref)
    SS = ssim(payload, prototype)

    p = morphology_priority(morph)

    return p * g * (1.0 + J) * (1.0 + SS) * dance

def hybrid_objective(sheaf: SheafVRAM,
                     morph: Morphology,
                     expected: float,
                     cost: float,
                     risk: float,
                     counterfactual: float,
                     sig: Iterable[int],
                     sig_ref: Iterable[int],
                     payload: np.ndarray,
                     prototype: np.ndarray,
                     dance: float) -> float:
    """
    Combined objective H = E * S.
    """
    E = hybrid_energy(sheaf)
    S = decision_score(morph, expected, cost, risk, counterfactual,
                       sig, sig_ref, payload, prototype, dance)
    return E * S

def hybrid_update_rule(sheaf: SheafVRAM,
                       morph: Morphology,
                       expected: float,
                       cost: float,
                       risk: float,
                       counterfactual: float,
                       sig: Iterable[int],
                       sig_ref: Iterable[int],
                       payload: np.ndarray,
                       prototype: np.ndarray,
                       dance: float,
                       lr: float = 0.01) -> None:
    """
    Perform a single gradient‑descent step on node sections to minimise H.
    For each node u we approximate ∂H/∂s_u ≈ 2 * Σ_{v} (R_uv·s_u – L_uv·s_v)·R_uvᵀ
    multiplied by the scalar factor S (since H = E·S and S is independent of sections).
    Sections are updated in‑place.
    """
    S = decision_score(morph, expected, cost, risk, counterfactual,
                       sig, sig_ref, payload, prototype, dance)

    # Compute gradients per node
    grads: Dict[Any, np.ndarray] = {node: np.zeros(dim) for node, dim in sheaf.node_dims.items()}

    for (u, v) in sheaf.edges:
        R_uv, L_uv = sheaf._restrictions[(u, v)]
        s_u = sheaf.get_section(u)
        s_v = sheaf.get_section(v)
        diff = R_uv @ s_u - L_uv @ s_v          # shape (k,)
        # Gradient w.r.t s_u
        grads[u] += 2.0 * (R_uv.T @ diff)
        # Gradient w.r.t s_v (negative because diff depends on s_v via -L_uv·s_v)
        grads[v] -= 2.0 * (L_uv.T @ diff)

    # Apply updates scaled by S and learning rate
    for node, grad in grads.items():
        new_section = sheaf.get_section(node) - lr * S * grad
        sheaf.set_section(node, new_section)

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph: two nodes (0,1) with 3‑dimensional sections
    node_dims = {0: 3, 1: 3}
    edges = [(0, 1)]
    vram_slots = [0]

    sheaf = SheafVRAM(node_dims, edges, vram_slots)

    # Random restriction matrices (k=2 rows)
    R = np.array([[1, 0, 0],
                  [0, 1, 0]], dtype=float)
    L = np.array([[0, 1, 0],
                  [0, 0, 1]], dtype=float)
    sheaf.set_restriction((0, 1), R, L)

    # Initialise sections
    sheaf.set_section(0, np.array([0.5, -0.2, 0.1]))
    sheaf.set_section(1, np.array([0.0, 0.3, -0.1]))

    # Morphology instance
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=1.5)

    # Dummy signatures for MinHash similarity
    sig = [1, 3, 5, 7, 9]
    sig_ref = [2, 3, 5, 8, 9]

    # Payload & prototype as small "images"
    payload = np.array([[0.2, 0.3], [0.4, 0.5]])
    prototype = np.array([[0.1, 0.35], [0.45, 0.55]])

    # Control signal
    dance = random.uniform(0.8, 1.2)

    # Compute initial objective
    H0 = hybrid_objective(sheaf, morph,
                          expected=1.0, cost=0.3, risk=0.2, counterfactual=0.1,
                          sig=sig, sig_ref=sig_ref,
                          payload=payload, prototype=prototype,
                          dance=dance)
    print(f"Initial hybrid objective H = {H0:.6f}")

    # Perform a few update steps
    for step in range(5):
        hybrid_update_rule(sheaf, morph,
                           expected=1.0, cost=0.3, risk=0.2, counterfactual=0.1,
                           sig=sig, sig_ref=sig_ref,
                           payload=payload, prototype=prototype,
                           dance=dance,
                           lr=0.05)
        H = hybrid_objective(sheaf, morph,
                             expected=1.0, cost=0.3, risk=0.2, counterfactual=0.1,
                             sig=sig, sig_ref=sig_ref,
                             payload=payload, prototype=prototype,
                             dance=dance)
        print(f"Step {step+1}: H = {H:.6f}")

    # Verify that sections remain the correct shape
    for node in node_dims:
        sec = sheaf.get_section(node)
        assert sec.shape == (node_dims[node],), "Section shape corrupted"
    print("Smoke test completed successfully.")