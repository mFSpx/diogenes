# DARWIN HAMMER — match 500, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s0.py (gen2)
# born: 2026-05-29T23:29:33Z

"""
Hybrid Algorithm: Sheaf‑Associative Memory with Signal‑Honesty Regularization

Parents:
- hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (Sheaf + Dense Associative Memory)
- hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s0.py (Signal/Noise scoring & honesty metric)

Mathematical Bridge:
The bridge is the scalar *signal_score* (and derived *honesty*) from the
conduit‑cockpit side.  We treat this scalar as a weighting factor that modulates
the energy landscape of the dense associative memory.  Concretely, for a
sheaf‑section vector **x** (concatenation of node vectors) we define

    E_mem(x) = -½ xᵀ W x + bᵀ x                     (associative memory energy)
    E_sheaf(x) = Σ_{(u,v)∈E} ‖R_uv_src x_u – R_uv_dst x_v‖²   (sheaf consistency)

The total hybrid energy is

    E_hybrid(x) = (1 – σ)·E_mem(x) + σ·E_sheaf(x),

where σ = signal_score ∈ [0,1] (higher signal → stronger sheaf regularisation,
lower signal → memory‑driven dynamics).  The honesty metric is identical to
signal_score, so it directly scales the update magnitudes.

The three core functions below implement:
1. signal/noise extraction (parent B),
2. hybrid energy evaluation (fusion of both parents),
3. gradient‑based update of the memory matrix using the signal‑derived scale.

A simple gradient descent retrieval routine demonstrates the unified system.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "signal_scores",
    "cockpit_honesty",
    "hybrid_energy",
    "hybrid_update_rule",
    "hybrid_retrieve",
]

# ----------------------------------------------------------------------
# Parent B – signal / noise utilities
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Estimate Shannon entropy of the first *sample* bytes of *data*."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0  # bits → bytes

def shannon_entropy(chunk):
    """Classic Shannon entropy (base‑2) for a list of byte values."""
    entropy = 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / len(chunk)
        entropy += -p_x * math.log(p_x, 2)
    return entropy

def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> tuple[float, float]:
    """Return (signal, noise) in [0,1] derived from content characteristics."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)
    signal = max(0.0, min(1.0,
        0.20 + status_bonus + mime_bonus + size_bonus + keyword_bonus + structure_bonus + 0.12 * entropy))
    noise = max(0.0, min(1.0,
        0.58 - 0.22 * entropy - keyword_bonus - structure_bonus + (0.12 if size < 64 else 0.0)))
    return signal, noise

def cockpit_honesty(signal_score: float) -> float:
    """In the original code honesty = signal_score; kept for semantic clarity."""
    return signal_score

# ----------------------------------------------------------------------
# Parent A – Sheaf structure
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * node_dims: dict mapping node identifier → dimension of its vector space.
    * edges: list of (u, v) tuples (directed).
    * Restrictions are linear maps stored per edge:
          src_map : ℝ^{dim(u)} → ℝ^{dim(e)}
          dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}
    * Sections assign a vector to each node.
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = {k: int(v) for k, v in node_dims.items()}
        self.edges = list(edges)
        self._restrictions = {}          # (u,v) → (src_map, dst_map)
        self._sections = {}              # node → np.ndarray

    # ------------------------------------------------------------------
    # Restriction handling
    # ------------------------------------------------------------------
    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)

        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")

        self._restrictions[(u, v)] = (src_map, dst_map)

    # ------------------------------------------------------------------
    # Section handling
    # ------------------------------------------------------------------
    def set_section(self, node, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.shape[0] != self.node_dims[node]:
            raise ValueError(f"Section vector size for node {node} must be {self.node_dims[node]}")
        self._sections[node] = value

    def get_section(self, node):
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def all_section_vectors(self) -> list[np.ndarray]:
        """Return a list of vectors ordered by node key insertion order."""
        return [self.get_section(node) for node in self.node_dims]

    def concatenated_section(self) -> np.ndarray:
        """Flattened concatenation of all node vectors."""
        return np.concatenate(self.all_section_vectors())

    # ------------------------------------------------------------------
    # Sheaf consistency penalty
    # ------------------------------------------------------------------
    def consistency_penalty(self) -> float:
        """Σ_{edge} || src_map·x_u – dst_map·x_v ||²."""
        total = 0.0
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            xu = self.get_section(u)
            xv = self.get_section(v)
            diff = src_map @ xu - dst_map @ xv
            total += np.linalg.norm(diff) ** 2
        return total

# ----------------------------------------------------------------------
# Parent A – Dense Associative Memory (simplified)
# ----------------------------------------------------------------------
class DenseAssociativeMemory:
    """
    Fully‑connected associative memory with a symmetric weight matrix W
    and bias vector b.  Energy for a state vector x ∈ ℝⁿ:

        E_mem(x) = -½ xᵀ W x + bᵀ x
    """

    def __init__(self, dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # Ensure symmetry
        A = rng.normal(scale=0.1, size=(dim, dim))
        self.W = (A + A.T) / 2.0
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return -0.5 * x @ self.W @ x + self.b @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x E_mem = -W x + b."""
        x = np.asarray(x, dtype=float)
        return -self.W @ x + self.b

    def update(self, grad: np.ndarray, lr: float) -> None:
        """Simple gradient ascent on the energy (i.e., descent on -E)."""
        self.W += lr * np.outer(grad, grad)  # crude Hebbian‑style update
        self.b += lr * grad

# ----------------------------------------------------------------------
# Hybrid core functions (fusion of both parents)
# ----------------------------------------------------------------------
def hybrid_energy(sheaf: Sheaf, memory: DenseAssociativeMemory, data: bytes,
                  status_code: int | None = None, mime: str = "",
                  keyword_hits: int = 0, structural_links: int = 0) -> float:
    """
    Compute the blended energy:

        E_hybrid = (1 - σ) * E_mem(x) + σ * λ * E_sheaf(x)

    where σ = signal_score (scaled by honesty) and λ is a fixed regularisation
    coefficient (set to 1.0 for simplicity).  The section vector x is taken from
    the sheaf's current sections.
    """
    signal, _ = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    honesty = cockpit_honesty(signal)          # same numeric value
    σ = honesty                                 # weight in [0,1]

    x = sheaf.concatenated_section()
    if x.size != memory.W.shape[0]:
        raise ValueError("Dimension mismatch between sheaf concatenated vector and memory matrix.")

    E_mem = memory.energy(x)
    E_sheaf = sheaf.consistency_penalty()
    λ = 1.0
    total = (1 - σ) * E_mem + σ * λ * E_sheaf
    return total

def hybrid_update_rule(sheaf: Sheaf, memory: DenseAssociativeMemory, data: bytes,
                       lr: float = 1e-3,
                       status_code: int | None = None, mime: str = "",
                       keyword_hits: int = 0, structural_links: int = 0) -> None:
    """
    Perform a single update step on the memory parameters using the gradient of
    the hybrid energy with respect to the concatenated section vector x.
    The step size is modulated by the signal_score (higher signal → smaller
    memory‑driven changes, stronger sheaf regularisation).
    """
    signal, _ = signal_scores(data, status_code, mime, keyword_hits, structural_links)
    σ = cockpit_honesty(signal)

    x = sheaf.concatenated_section()
    if x.size != memory.W.shape[0]:
        raise ValueError("Dimension mismatch between sheaf vector and memory matrix.")

    # Gradient of E_mem w.r.t x is -W x + b (already provided)
    grad_mem = memory.gradient(x)

    # Gradient of sheaf penalty w.r.t each node vector.
    # For simplicity we compute a numerical approximation using finite differences.
    eps = 1e-6
    grad_sheaf = np.zeros_like(x)
    for i in range(x.size):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[i] += eps
        x_neg[i] -= eps
        # temporarily set sections to perturbed vectors
        _apply_concatenated_to_sheaf(sheaf, x_pos)
        pen_pos = sheaf.consistency_penalty()
        _apply_concatenated_to_sheaf(sheaf, x_neg)
        pen_neg = sheaf.consistency_penalty()
        grad_sheaf[i] = (pen_pos - pen_neg) / (2 * eps)
    # Restore original sections
    _apply_concatenated_to_sheaf(sheaf, x)

    # Total gradient of hybrid energy w.r.t x
    total_grad = (1 - σ) * grad_mem + σ * grad_sheaf

    # Update memory parameters using the total gradient (Hebbian style)
    memory.update(total_grad, lr * (1 - σ))  # scale lr by (1-σ) to keep balance

def _apply_concatenated_to_sheaf(sheaf: Sheaf, vec: np.ndarray) -> None:
    """Utility: split *vec* according to node dimensions and store back into sheaf."""
    idx = 0
    for node, dim in sheaf.node_dims.items():
        segment = vec[idx: idx + dim]
        sheaf.set_section(node, segment)
        idx += dim

def hybrid_retrieve(sheaf: Sheaf, memory: DenseAssociativeMemory, data: bytes,
                    steps: int = 200, lr: float = 1e-2,
                    status_code: int | None = None, mime: str = "",
                    keyword_hits: int = 0, structural_links: int = 0) -> np.ndarray:
    """
    Gradient‑descent retrieval: start from a random sheaf section, then iteratively
    minimise the hybrid energy.  Returns the final concatenated section vector.
    """
    # Initialise random sections
    rng = np.random.default_rng()
    for node, dim in sheaf.node_dims.items():
        sheaf.set_section(node, rng.normal(scale=0.1, size=dim))

    for _ in range(steps):
        # Compute current energy and its gradient w.r.t x via finite differences
        signal, _ = signal_scores(data, status_code, mime, keyword_hits, structural_links)
        σ = cockpit_honesty(signal)

        x = sheaf.concatenated_section()
        # Memory gradient
        grad_mem = memory.gradient(x)
        # Sheaf penalty gradient (finite diff as before)
        eps = 1e-6
        grad_sheaf = np.zeros_like(x)
        for i in range(x.size):
            x_pos = x.copy()
            x_neg = x.copy()
            x_pos[i] += eps
            x_neg[i] -= eps
            _apply_concatenated_to_sheaf(sheaf, x_pos)
            pen_pos = sheaf.consistency_penalty()
            _apply_concatenated_to_sheaf(sheaf, x_neg)
            pen_neg = sheaf.consistency_penalty()
            grad_sheaf[i] = (pen_pos - pen_neg) / (2 * eps)
        _apply_concatenated_to_sheaf(sheaf, x)  # restore

        total_grad = (1 - σ) * grad_mem + σ * grad_sheaf
        # Gradient descent step on x (note: we directly modify sections)
        x_new = x - lr * total_grad
        _apply_concatenated_to_sheaf(sheaf, x_new)

    return sheaf.concatenated_section()

# ----------------------------------------------------------------------
# Smoke test