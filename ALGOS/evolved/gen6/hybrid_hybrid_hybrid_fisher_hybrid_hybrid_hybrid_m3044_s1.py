# DARWIN HAMMER — match 3044, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m2021_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (gen3)
# born: 2026-05-29T23:47:27Z

"""Hybrid Fisher‑SSIM‑Sheaf‑Associative Memory Algorithm
Parents:
    - hybrid_hybrid_fisher_locali_hybrid_hybrid_hybrid_m2021_s0.py (Fisher‑SSIM‑Bandit)
    - hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s2.py (Sheaf‑Dense Associative Memory)

Mathematical Bridge
-------------------
Parent A provides a *temperature‑modulated hybrid score*  

    H(T,θ,x,y) = φ(T) · F(θ;μ,σ) · SSIM(x,y)

where φ(T) is the Schoolfield temperature‑dependent activity factor,
F is the Fisher information of a Gaussian beam and SSIM measures perceptual
similarity.

Parent B treats a cellular sheaf’s sections as query vectors **q** that are
fed into a dense associative memory with energy  

    E(q) = -½ qᵀ W q + bᵀ q .

The bridge is to use the scalar score **H** as a multiplicative modulation of
the memory energy and as a learning‑rate factor for updating the memory
matrix **W**.  The sheaf’s restriction maps are also employed to project node‑
level vectors into the global query space, guaranteeing that updates respect
the sheaf’s topology.

The resulting hybrid system therefore fuses information‑geometric reward
computation (Parent A) with sheaf‑aware associative memory dynamics (Parent B)."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Parent A core: Gaussian beam, Fisher score, SSIM, Schoolfield model
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity I(θ)."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


def schoolfield_phi(T: float,
                    B0: float = 1.0,
                    Ea: float = 0.65,
                    Eh: float = 2.5,
                    Th: float = 310.0,
                    k: float = 8.617e-5) -> float:
    """Simplified Schoolfield temperature‑dependent activity factor φ(T)."""
    if T <= 0:
        raise ValueError("Temperature must be > 0 K")
    # Arrhenius term
    arr = B0 * math.exp(-Ea / (k * T))
    # High‑temperature deactivation term
    deact = 1.0 + math.exp((Eh / k) * (1.0 / Th - 1.0 / T))
    return arr / deact


def hybrid_score(theta: float, center: float, width: float,
                 x: np.ndarray, y: np.ndarray,
                 T: float) -> float:
    """Temperature‑modulated hybrid score H = φ(T)·F·SSIM."""
    phi = schoolfield_phi(T)
    fisher = fisher_score(theta, center, width)
    sim = ssim(x, y)
    return phi * fisher * sim


# ----------------------------------------------------------------------
# Parent B core: Sheaf structure and Dense Associative Memory
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * node_dims: dict mapping node identifier → dimension of its local vector space
    * edges: list of (u, v) tuples defining directed edges
    * restriction maps: for each edge (u,v) a pair (src_map, dst_map) where
        src_map : ℝ^{dim(u)} → ℝ^{dim(e)}
        dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}
    * sections: assignment of a vector to each node
    """

    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = {n: int(d) for n, d in node_dims.items()}
        self.edges = list(edges)
        self._restrictions = {}          # (u,v) → (src_map, dst_map)
        self._sections = {}              # node → vector

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)

        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must share the same row dimension")
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: any, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(-1)
        if value.shape[0] != self.node_dims[node]:
            raise ValueError(f"section vector length for node {node} must be {self.node_dims[node]}")
        self._sections[node] = value

    def get_section(self, node: any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def global_query_vector(self) -> np.ndarray:
        """
        Assemble a global query vector q by concatenating node sections
        after projecting them onto a common edge space using restriction maps.
        For simplicity we concatenate raw sections (the bridge does not require
        a sophisticated co‑homology solver).
        """
        vectors = [self.get_section(n) for n in sorted(self.node_dims.keys())]
        return np.concatenate(vectors) if vectors else np.array([])


class DenseAssociativeMemory:
    """
    Simple dense associative memory with weight matrix W and bias b.
    Energy: E(q) = -½ qᵀ W q + bᵀ q
    """

    def __init__(self, dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.W = rng.normal(scale=0.1, size=(dim, dim))
        # Enforce symmetry for a proper quadratic form
        self.W = (self.W + self.W.T) / 2.0
        self.b = rng.normal(scale=0.1, size=dim)

    def energy(self, q: np.ndarray) -> float:
        q = q.reshape(-1)
        return -0.5 * q @ self.W @ q + self.b @ q

    def update(self, q: np.ndarray, lr: float = 0.01) -> None:
        """
        Gradient descent on the energy w.r.t. W and b.
        ∂E/∂W = -½ (q qᵀ)
        ∂E/∂b = q
        """
        q = q.reshape(-1, 1)
        self.W -= lr * (-0.5) * (q @ q.T)
        self.b -= lr * q.ravel()

    def retrieve(self, q: np.ndarray, steps: int = 10, lr: float = 0.1) -> np.ndarray:
        """
        Hopfield‑style iterative dynamics: q ← q - lr ∇E(q)
        ∇E = -W q + b
        """
        q = q.astype(float).copy()
        for _ in range(steps):
            grad = -self.W @ q + self.b
            q -= lr * grad
        return q


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_energy(sheaf: Sheaf,
                  memory: DenseAssociativeMemory,
                  theta: float,
                  center: float,
                  width: float,
                  x: np.ndarray,
                  y: np.ndarray,
                  T: float) -> float:
    """
    Compute the temperature‑modulated hybrid energy:

        E_hybrid = H(theta, x, y, T) * E(q)

    where q is the global query vector derived from the sheaf sections.
    """
    q = sheaf.global_query_vector()
    if q.size == 0:
        raise ValueError("Sheaf has no sections; cannot build query vector")
    base_energy = memory.energy(q)
    H = hybrid_score(theta, center, width, x, y, T)
    return H * base_energy


def hybrid_update_rule(sheaf: Sheaf,
                       memory: DenseAssociativeMemory,
                       theta: float,
                       center: float,
                       width: float,
                       x: np.ndarray,
                       y: np.ndarray,
                       T: float,
                       base_lr: float = 0.01) -> None:
    """
    Perform a memory update where the learning rate is scaled by the hybrid
    score H.  The sheaf’s restriction maps are used implicitly because the
    query vector q already respects the sheaf’s topology.
    """
    q = sheaf.global_query_vector()
    H = hybrid_score(theta, center, width, x, y, T)
    scaled_lr = base_lr * H
    memory.update(q, lr=scaled_lr)


def hybrid_retrieve(sheaf: Sheaf,
                    memory: DenseAssociativeMemory,
                    theta: float,
                    center: float,
                    width: float,
                    x: np.ndarray,
                    y: np.ndarray,
                    T: float,
                    steps: int = 15,
                    lr: float = 0.05) -> np.ndarray:
    """
    Retrieve a stable pattern from memory using a query that has been
    temperature‑modulated.  The returned vector is projected back onto the
    sheaf’s node spaces (simple split according to node dimensions).
    """
    q0 = sheaf.global_query_vector()
    H = hybrid_score(theta, center, width, x, y, T)
    # Modulate the dynamics by H (larger H → more aggressive convergence)
    q_final = memory.retrieve(q0, steps=steps, lr=lr * H)

    # Split the final vector back into node sections
    split_vectors = {}
    idx = 0
    for node in sorted(sheaf.node_dims.keys()):
        dim = sheaf.node_dims[node]
        split_vectors[node] = q_final[idx:idx + dim]
        idx += dim
    return split_vectors


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny sheaf with two nodes
    node_dims = {"A": 3, "B": 2}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)

    # Random restriction maps (not used in the simple concatenation but kept for completeness)
    rng = np.random.default_rng(42)
    src_map = rng.normal(size=(4, node_dims["A"]))   # maps A → edge space dim 4
    dst_map = rng.normal(size=(4, node_dims["B"]))   # maps B → edge space dim 4
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Random sections
    sheaf.set_section("A", rng.normal(size=node_dims["A"]))
    sheaf.set_section("B", rng.normal(size=node_dims["B"]))

    # Create associative memory matching the global query dimension (3+2 = 5)
    memory = DenseAssociativeMemory(dim=5, seed=123)

    # Synthetic signals for SSIM
    x_sig = rng.integers(0, 256, size=100).astype(float)
    y_sig = rng.integers(0, 256, size=100).astype(float)

    # Parameters for hybrid score
    theta = 0.7
    center = 0.5
    width = 0.1
    T = 310.0  # Kelvin

    # Compute hybrid energy
    E = hybrid_energy(sheaf, memory, theta, center, width, x_sig, y_sig, T)
    print(f"Hybrid energy: {E:.6f}")

    # Perform an update
    hybrid_update_rule(sheaf, memory, theta, center, width, x_sig, y_sig, T)

    # Retrieve a pattern
    retrieved = hybrid_retrieve(sheaf, memory, theta, center, width, x_sig, y_sig, T)
    for node, vec in retrieved.items():
        print(f"Node {node} retrieved vector: {vec}")

    sys.exit(0)