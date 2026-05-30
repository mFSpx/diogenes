# DARWIN HAMMER — match 500, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# parent_b: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s0.py (gen2)
# born: 2026-05-29T23:29:33Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Utility: signal / noise extraction (Parent B)
# ----------------------------------------------------------------------
def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Estimate Shannon entropy (in bytes) of the first *sample* bytes."""
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0  # bits → bytes


def shannon_entropy(chunk: Iterable[int]) -> float:
    """Classic Shannon entropy (base‑2) for a list of byte values."""
    entropy = 0.0
    n = len(chunk)
    if n == 0:
        return 0.0
    for x in set(chunk):
        p_x = chunk.count(x) / n
        entropy += -p_x * math.log(p_x, 2)
    return entropy


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """Return (signal, noise) ∈ [0,1] derived from content characteristics."""
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml")) else 0.02
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)

    signal = max(
        0.0,
        min(
            1.0,
            0.20
            + status_bonus
            + mime_bonus
            + size_bonus
            + keyword_bonus
            + structure_bonus
            + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58
            - 0.22 * entropy
            - keyword_bonus
            - structure_bonus
            + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


def cockpit_honesty(signal_score: float) -> float:
    """Identity mapping kept for semantic clarity; can be extended later."""
    return signal_score


# ----------------------------------------------------------------------
# Parent A – Sheaf structure
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    *node_dims*: mapping node → dimension of its local vector space.
    *edges*: list of (u, v) directed edges.
    *restrictions*: per edge linear maps (src_map, dst_map) with a common
    codomain dimension.
    *sections*: assignment of a vector to each node.
    """

    def __init__(self, node_dims: Dict, edges: List[Tuple]):
        self.node_dims: Dict = {k: int(v) for k, v in node_dims.items()}
        self.edges: List[Tuple] = list(edges)
        self._restrictions: Dict[Tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict = {}

    # ------------------------------------------------------------------
    # Restriction handling
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
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

    def get_section(self, node) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def all_section_vectors(self) -> List[np.ndarray]:
        """Return vectors ordered by the insertion order of node_dims keys."""
        return [self.get_section(node) for node in self.node_dims]

    def concatenated_section(self) -> np.ndarray:
        """Flattened concatenation of all node vectors."""
        return np.concatenate(self.all_section_vectors())

    # ------------------------------------------------------------------
    # Consistency penalty & its gradient
    # ------------------------------------------------------------------
    def consistency_penalty(self) -> float:
        """Σ_{edge} ‖src·x_u – dst·x_v‖²."""
        total = 0.0
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            diff = src_map @ self.get_section(u) - dst_map @ self.get_section(v)
            total += np.linalg.norm(diff) ** 2
        return total

    def consistency_gradient(self) -> np.ndarray:
        """
        Gradient of the consistency penalty w.r.t. the concatenated section vector.
        Returns a vector of length Σ dim(node).
        """
        # Initialise gradient blocks per node
        grad_blocks: Dict = {node: np.zeros(self.node_dims[node]) for node in self.node_dims}

        for (u, v), (src_map, dst_map) in self._restrictions.items():
            xu = self.get_section(u)
            xv = self.get_section(v)
            diff = src_map @ xu - dst_map @ xv  # shape (r,)

            # ∂/∂x_u  ‖diff‖² = 2·srcᵀ·diff
            grad_blocks[u] += 2.0 * src_map.T @ diff
            # ∂/∂x_v  ‖diff‖² = -2·dstᵀ·diff
            grad_blocks[v] -= 2.0 * dst_map.T @ diff

        # Concatenate in the same order as concatenated_section()
        return np.concatenate([grad_blocks[node] for node in self.node_dims])


# ----------------------------------------------------------------------
# Parent A – Dense Associative Memory (simplified)
# ----------------------------------------------------------------------
class DenseAssociativeMemory:
    """
    Fully‑connected associative memory with a symmetric weight matrix *W*
    and bias vector *b*.  Energy for a state vector x ∈ ℝⁿ:

        E_mem(x) = -½ xᵀ W x + bᵀ x
    """

    def __init__(self, dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        A = rng.normal(scale=0.1, size=(dim, dim))
        self.W = (A + A.T) / 2.0  # enforce symmetry
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return -0.5 * x @ self.W @ x + self.b @ x

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """∇_x E_mem = -W x + b."""
        x = np.asarray(x, dtype=float)
        return -self.W @ x + self.b

    def hebbian_update(self, grad: np.ndarray, lr: float) -> None:
        """
        Crude Hebbian‑style update that respects symmetry.
        Uses outer product of the gradient (rank‑1) scaled by *lr*.
        """
        self.W += lr * (np.outer(grad, grad) + np.outer(grad, grad).T) / 2.0
        self.b += lr * grad


# ----------------------------------------------------------------------
# Hybrid core: deeper mathematical integration
# ----------------------------------------------------------------------
def _normalize_energy(value: float, scale: float) -> float:
    """Scale an energy term to a comparable magnitude."""
    # Prevent division by zero; add a tiny epsilon.
    return value / (abs(scale) + 1e-12)


def hybrid_energy(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    sigma: float,
) -> float:
    """
    Compute a *scale‑balanced* hybrid energy.

    The raw energies live on different scales (quadratic form vs. sum of squares).
    We normalise each term by a characteristic magnitude:
        *mem_scale* = Frobenius norm of W + ||b||₂
        *sheaf_scale* = √(number_of_edges)  (rough proxy for typical penalty size)

    The final energy is a convex combination controlled by σ ∈ [0,1].
    """
    x = sheaf.concatenated_section()
    mem_raw = dam.energy(x)
    sheaf_raw = sheaf.consistency_penalty()

    mem_scale = np.linalg.norm(dam.W, "fro") + np.linalg.norm(dam.b)
    sheaf_scale = math.sqrt(len(sheaf._restrictions) + 1e-12)

    mem_norm = _normalize_energy(mem_raw, mem_scale)
    sheaf_norm = _normalize_energy(sheaf_raw, sheaf_scale)

    return (1.0 - sigma) * mem_norm + sigma * sheaf_norm


def hybrid_gradient(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the pair (grad_mem, grad_sheaf) already weighted by σ.
    The caller can combine them as needed.
    """
    x = sheaf.concatenated_section()
    grad_mem = dam.gradient(x)          # shape (dim,)
    grad_sheaf = sheaf.consistency_gradient()  # shape (dim,)

    # Weight each component according to σ, mirroring the energy weighting.
    grad_mem_weighted = (1.0 - sigma) * grad_mem
    grad_sheaf_weighted = sigma * grad_sheaf
    return grad_mem_weighted, grad_sheaf_weighted


def hybrid_update_rule(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    signal: float,
    base_lr: float = 1e-3,
    honesty_lambda: float = 0.05,
) -> None:
    """
    Perform a single hybrid update step.

    * signal* is the raw signal_score ∈ [0,1].
    * honesty* is derived via ``cockpit_honesty`` and used as a regulariser.
    * base_lr* is the nominal learning rate; it is modulated by signal
      (higher signal → larger step on the sheaf side) and by an honesty‑penalty
      that discourages the model from drifting too far from the measured signal.
    """
    sigma = signal  # direct mapping; can be replaced by a smoother function if desired
    honesty = cockpit_honesty(signal)

    # Compute weighted gradients
    grad_mem, grad_sheaf = hybrid_gradient(sheaf, dam, sigma)

    # Combine into a single update direction for the concatenated state.
    combined_grad = grad_mem + grad_sheaf

    # Adaptive learning rate: base_lr scaled by (signal + ε) and softened by honesty.
    lr = base_lr * (0.5 + 0.5 * sigma)  # keep lr in a reasonable range
    lr += honesty_lambda * (honesty - sigma)  # honesty regularisation term (usually zero)

    # Update the associative memory (Hebbian style) using the *combined* gradient.
    dam.hebbian_update(combined_grad, lr)

    # Update the sheaf sections via a simple gradient descent step.
    # We distribute the combined gradient back to node blocks.
    offset = 0
    for node in sheaf.node_dims:
        dim = sheaf.node_dims[node]
        node_grad = combined_grad[offset : offset + dim]
        current = sheaf.get_section(node)
        sheaf.set_section(node, current - lr * node_grad)  # descent on penalty
        offset += dim


def hybrid_retrieve(
    sheaf: Sheaf,
    dam: DenseAssociativeMemory,
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    steps: int = 50,
    base_lr: float = 1e-3,
) -> np.ndarray:
    """
    End‑to‑end retrieval routine.

    1. Compute signal/noise from the raw *data*.
    2. Initialise the sheaf sections (if not already) with small random values.
    3. Run ``steps`` gradient‑descent iterations on the hybrid energy.
    4. Return the final concatenated state vector.
    """
    signal, _ = signal_scores(
        data,
        status_code=status_code,
        mime=mime,
        keyword_hits=keyword_hits,
        structural_links=structural_links,
    )

    # Ensure the sheaf has a section for every node; initialise lazily.
    rng = np.random.default_rng()
    for node, dim in sheaf.node_dims.items():
        if node not in sheaf._sections:
            sheaf.set_section(node, rng.normal(scale=0.01, size=dim))

    # Verify dimensional compatibility.
    total_dim = sum(sheaf.node_dims.values())
    if total_dim != dam.W.shape[0]:
        raise ValueError(
            f"Dimension mismatch: sheaf concatenated size {total_dim} ≠ DAM dimension {dam.W.shape[0]}"
        )

    for _ in range(steps):
        hybrid_update_rule(sheaf, dam, signal, base_lr=base_lr)

    return sheaf.concatenated_section()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "signal_scores",
    "cockpit_honesty",
    "hybrid_energy",
    "hybrid_gradient",
    "hybrid_update_rule",
    "hybrid_retrieve",
]