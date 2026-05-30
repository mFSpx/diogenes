# DARWIN HAMMER — match 2744, survivor 7
# gen: 5
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s2.py (gen4)
# born: 2026-05-29T23:45:38Z

"""Hybrid Algorithm: RBF‑Fisher‑SSIM Fusion

Parents:
- hybrid_perceptual_nlms_rbf_surrogate (Algorithm A)
- hybrid_fisher_ssim_sheaf_associative (Algorithm B)

Mathematical Bridge
-------------------
Algorithm A provides a radial‑basis surrogate whose prediction is a weighted sum of
Gaussian kernels centred on perceptual hashes.  Algorithm B supplies a *Fisher*
information weight for each centre (derived from a Gaussian‑beam model) and a
*Structural Similarity Index* (SSIM) that measures agreement between vectors.

The hybrid system therefore:

1. Computes a Fisher score `F_i(θ)` for every RBF centre `c_i`.  This score
   multiplies the corresponding kernel contribution, turning the plain RBF
   expansion into a Fisher‑weighted surrogate:
   `ŷ(x) = Σ_i w_i·F_i·φ(‖x‑c_i‖)`.

2. Uses SSIM between the current input `x` and a reference packet `p` to
   modulate the Normalised Least‑Mean‑Squares (NLMS) learning rate.  The effective
   step size becomes `μ·SSIM(x,p)`, allowing the update to adapt faster when the
   input resembles the reference and slower otherwise.

3. Embeds the surrogate inside a sheaf‑like associative memory: each node of the
   sheaf stores a vector, the hybrid energy of the sheaf is the sum of Fisher‑
   weighted RBF prediction errors, further scaled by SSIM between node vectors
   and the packet vector.

The following module implements these ideas, exposing three core hybrid
operations:
`hybrid_predict`, `hybrid_nlms_update`, and `sheaf_hybrid_energy`.  All
dependencies are limited to the Python standard library and NumPy."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Vector = np.ndarray


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same shape")
    return float(np.linalg.norm(a - b))


def solve_linear(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a·x = b by Gaussian elimination (no external libs)."""
    n = len(b)
    m = np.hstack((a.astype(float), b[:, None].astype(float)))
    for col in range(n):
        pivot = np.argmax(np.abs(m[col:, col])) + col
        if abs(m[pivot, col]) < 1e-12:
            raise ValueError("singular surrogate system")
        if pivot != col:
            m[[col, pivot]] = m[[pivot, col]]
        m[col] = m[col] / m[col, col]
        for row in range(n):
            if row == col:
                continue
            factor = m[row, col]
            m[row] -= factor * m[col]
    return m[:, -1]


# ----------------------------------------------------------------------
# Algorithm A – RBF surrogate with NLMS
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    """Plain RBF surrogate (no Fisher weighting)."""
    centers: np.ndarray          # shape (n_centers, dim)
    weights: np.ndarray          # shape (n_centers,)
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Standard RBF prediction Σ w_i φ(||x‑c_i||)."""
        kernels = np.array([gaussian(euclidean(x, c), self.epsilon) for c in self.centers])
        return float(np.dot(self.weights, kernels))


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares weight update.
    Returns a new weight vector.
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in the interval (0, 2)")
    y = float(np.dot(weights, x))
    error = target - y
    power = float(np.dot(x, x) + eps)
    delta = (mu / power) * error * x
    return weights + delta


# ----------------------------------------------------------------------
# Algorithm B – Fisher, SSIM and Sheaf structures
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray,
         y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """1‑D Structural Similarity Index (SSIM)."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))
    sigma_x = float(np.var(x, ddof=0))
    sigma_y = float(np.var(y, ddof=0))
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


@dataclass
class Sheaf:
    """Very lightweight sheaf: nodes store vectors, edges encode adjacency."""
    nodes: Dict[int, np.ndarray] = field(default_factory=dict)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def adjacency_matrix(self) -> np.ndarray:
        """Return adjacency matrix A where A[i, j] = 1 if (i, j) ∈ edges."""
        n = len(self.nodes)
        A = np.zeros((n, n), dtype=float)
        id_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(self.nodes))}
        for u, v in self.edges:
            i, j = id_to_idx[u], id_to_idx[v]
            A[i, j] = A[j, i] = 1.0
        return A

    def node_vectors(self) -> np.ndarray:
        """Stack node vectors into a matrix V (n_nodes × dim)."""
        ordered_ids = sorted(self.nodes)
        return np.vstack([self.nodes[i] for i in ordered_ids])


# ----------------------------------------------------------------------
# Hybrid Core – mathematical fusion of A and B
# ----------------------------------------------------------------------
def fisher_weights(centers: np.ndarray,
                   theta: float,
                   width: float) -> np.ndarray:
    """
    Compute Fisher information for each centre.
    Treat the centre's first component as the “θ” argument of the beam.
    """
    thetas = centers[:, 0]  # simple projection onto first dimension
    return np.array([fisher_score(t, theta, width) for t in thetas])


def hybrid_predict(surrogate: RBFSurrogate,
                   x: Vector,
                   theta: float,
                   width: float) -> float:
    """
    Fisher‑weighted RBF prediction.
    ŷ = Σ_i w_i · F_i(θ) · φ(||x‑c_i||)
    """
    fisher = fisher_weights(surrogate.centers, theta, width)
    kernels = np.array([gaussian(euclidean(x, c), surrogate.epsilon) for c in surrogate.centers])
    return float(np.dot(surrogate.weights * fisher, kernels))


def hybrid_nlms_update(surrogate: RBFSurrogate,
                       x: Vector,
                       target: float,
                       packet: Vector,
                       theta: float,
                       width: float,
                       mu: float = 0.5,
                       eps: float = 1e-9) -> RBFSurrogate:
    """
    NLMS weight update where the learning rate is modulated by SSIM(x, packet)
    and the gradient is additionally scaled by Fisher scores.
    Returns a new RBFSurrogate instance (immutable dataclass).
    """
    # 1. Compute base prediction and error
    pred = hybrid_predict(surrogate, x, theta, width)
    error = target - pred

    # 2. SSIM‑modulated step size
    similarity = ssim(x, packet)
    mu_eff = mu * similarity  # 0 ≤ similarity ≤ 1

    # 3. Kernel vector φ(x) (size = n_centers)
    phi = np.array([gaussian(euclidean(x, c), surrogate.epsilon) for c in surrogate.centers])

    # 4. Fisher weighting for the gradient
    fisher = fisher_weights(surrogate.centers, theta, width)

    # NLMS formula with extra Fisher factor:
    # Δw = (μ_eff / (||φ||² + eps)) * error * (F ⊙ φ)
    power = float(np.dot(phi, phi) + eps)
    delta = (mu_eff / power) * error * (fisher * phi)

    new_weights = surrogate.weights + delta
    return RBFSurrogate(centers=surrogate.centers, weights=new_weights, epsilon=surrogate.epsilon)


def sheaf_hybrid_energy(sheaf: Sheaf,
                        packet: Vector,
                        theta: float,
                        width: float,
                        epsilon: float = 1.0) -> float:
    """
    Energy of a sheaf under the hybrid model.
    For each node i we compute a Fisher‑weighted RBF error with respect to the
    packet vector and sum them, optionally scaling by SSIM between node and packet.
    """
    if not sheaf.nodes:
        return 0.0

    # Build a temporary surrogate whose centres are the node vectors.
    centers = np.vstack([v for v in sheaf.nodes.values()])
    # Initialise equal weights (1.0) – they will be scaled by Fisher inside the loop.
    dummy_weights = np.ones(centers.shape[0])

    surrogate = RBFSurrogate(centers=centers, weights=dummy_weights, epsilon=epsilon)

    total_energy = 0.0
    for node_id, vec in sheaf.nodes.items():
        # Prediction of the node vector using the surrogate built from all nodes
        pred = hybrid_predict(surrogate, vec, theta, width)

        # Error w.r.t. packet (simple squared error)
        err = np.linalg.norm(vec - packet) ** 2

        # SSIM modulation (higher similarity reduces contribution)
        sim = ssim(vec, packet)

        # Combine: Fisher‑weighted error * (1 - SSIM)
        total_energy += err * (1.0 - sim) * pred  # pred already contains Fisher weighting
    return float(total_energy)


# ----------------------------------------------------------------------
# Demonstration functions (three distinct hybrid operations)
# ----------------------------------------------------------------------
def demo_hybrid_prediction():
    """Showcase of Fisher‑weighted RBF prediction."""
    dim = 5
    n_cent = 8
    centers = np.random.rand(n_cent, dim)
    weights = np.random.randn(n_cent)
    surrogate = RBFSurrogate(centers, weights, epsilon=0.8)

    x = np.random.rand(dim)
    theta, width = 0.3, 0.1
    y = hybrid_predict(surrogate, x, theta, width)
    print(f"Hybrid prediction: {y:.6f}")


def demo_hybrid_learning():
    """One NLMS step with SSIM‑modulated learning rate."""
    dim = 4
    n_cent = 6
    centers = np.random.rand(n_cent, dim)
    weights = np.zeros(n_cent)
    surrogate = RBFSurrogate(centers, weights, epsilon=1.2)

    x = np.random.rand(dim)
    packet = np.random.rand(dim)            # reference for SSIM
    target = 0.7
    theta, width = 0.5, 0.2

    updated = hybrid_nlms_update(surrogate, x, target, packet, theta, width, mu=0.6)
    print("Weight change norm:", np.linalg.norm(updated.weights - surrogate.weights))


def demo_sheaf_energy():
    """Compute hybrid energy on a tiny sheaf."""
    # Create three nodes with 3‑D vectors
    nodes = {
        0: np.array([0.2, 0.5, 0.1]),
        1: np.array([0.8, 0.3, 0.4]),
        2: np.array([0.5, 0.7, 0.9]),
    }
    edges = [(0, 1), (1, 2)]
    sheaf = Sheaf(nodes=nodes, edges=edges)

    packet = np.array([0.4, 0.6, 0.3])
    theta, width = 0.4, 0.15
    energy = sheaf_hybrid_energy(sheaf, packet, theta, width, epsilon=1.0)
    print(f"Sheaf hybrid energy: {energy:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=== Hybrid RBF‑Fisher‑SSIM Demo ===")
    demo_hybrid_prediction()
    demo_hybrid_learning()
    demo_sheaf_energy()
    print("All demo functions executed successfully.")