# DARWIN HAMMER — match 3096, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_omni_c_hybrid_ternary_lens__m1536_s0.py (gen6)
# born: 2026-05-29T23:47:56Z

"""Hybrid Algorithm: Fisher‑Caputo‑Chaotic‑JEPA Fusion

Parents
-------
- **Parent A** – `hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s0.py`  
  Provides Gaussian beam, Fisher‑score computation and a Caputo fractional
  derivative implementation for modelling algebraic decay of edge weights.

- **Parent B** – `hybrid_hybrid_hybrid_omni_c_hybrid_ternary_lens__m1536_s0.py`  
  Supplies chaotic graph generation, latent vectors, pheromone probabilities,
  entropy‑based reconstruction risk, a Fisher information matrix and a JEPA‑style
  energy functional.

Mathematical Bridge
-------------------
The bridge is the **Fisher information**:
* From Parent A the scalar Fisher score `fisher_score(θ)` is used to build a
  diagonal Fisher information matrix `I(p)` for the pheromone probability
  vector `p` (Parent B).
* The Caputo fractional derivative models the *temporal decay* of the chaotic
  adjacency matrix edge‑weights, turning the binary adjacency into a
  continuously evolving weighted matrix `W(t)`.
* The weighted Fisher matrix `I_w = risk_score·I(p)` replaces the constant
  precision matrix in the JEPA energy `E(z) = ½ zᵀ I_w z – Σ log p_i`.

The resulting hybrid operates in four stages:
1. Generate a chaotic graph (`A`) and a latent vector (`z`).
2. Compute pheromone probabilities `p`, entropy, and the diagonal Fisher matrix
   `I(p)` using the Gaussian‑beam Fisher score.
3. Obtain a reconstruction‑risk scalar via SSIM between the current weighted
   adjacency and a reference pattern; use it to weight `I(p)`.
4. Compute the weighted JEPA energy and perform a single gradient descent step
   on `z`. Edge‑weights evolve by a Caputo fractional derivative of an
   exponential decay kernel.

The module below implements this fusion with three public functions that
demonstrate the hybrid workflow and a smoke‑test in `__main__`.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian envelope used as a weighting kernel."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a Gaussian‑beam parameter."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index (simplified, 1‑D version)."""
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('samples must not be empty')
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function (real argument)."""
    # Coefficients from Numerical Recipes
    g = 7
    p = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_derivative(f, alpha: float, t: float, dt: float = 0.01) -> float:
    """
    Caputo fractional derivative of order ``alpha`` of scalar function ``f`` at time ``t``.
    Uses a simple trapezoidal rule on a uniform grid.
    """
    if not (0 < alpha < 1):
        raise ValueError('alpha must be in (0,1)')
    tau = np.arange(0, t, dt)
    if tau.size == 0:
        return 0.0
    f_tau = f(tau)
    kernel = (t - tau) ** (-alpha)
    integral = np.trapz(f_tau * kernel, tau)
    return integral / gamma_lanczos(1 - alpha)

# ----------------------------------------------------------------------
# Parent B utilities (chaotic graph + JEPA core)
# ----------------------------------------------------------------------
def chaotic_graph(num_nodes: int, chaos_factor: float = 3.9) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a directed chaotic adjacency matrix and a latent vector.
    Logistic map iterates provide binary connections.
    """
    def logistic_map(x: float) -> float:
        return chaos_factor * x * (1 - x)

    # Initialise a random seed vector in (0,1)
    seed = np.random.rand(num_nodes)
    A = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        x = seed[i]
        for j in range(num_nodes):
            x = logistic_map(x)
            A[i, j] = 1 if x > 0.5 else 0

    # Latent vector sampled from standard normal
    z = np.random.randn(num_nodes)
    return A, z

def pheromone_probabilities(num_nodes: int) -> np.ndarray:
    """Mock pheromone probabilities that sum to 1."""
    raw = np.random.rand(num_nodes)
    return raw / raw.sum()

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector."""
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def fisher_information_matrix(p: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Build a diagonal Fisher information matrix using the scalar Fisher score
    from Parent A evaluated on each probability entry.
    """
    scores = np.vectorize(lambda pi: fisher_score(pi, center, width))(p)
    return np.diag(scores)

def reconstruction_risk_score(weighted_adj: np.ndarray,
                              reference: np.ndarray) -> float:
    """
    Compute a scalar risk score based on SSIM between the current weighted
    adjacency and a reference pattern (both flattened).
    """
    return 1.0 - ssim(weighted_adj.ravel(), reference.ravel())

def weighted_fisher_matrix(I: np.ndarray, risk_score: float) -> np.ndarray:
    """Scale the Fisher matrix by the reconstruction risk."""
    return I * risk_score

def weighted_jepa_energy(z: np.ndarray, Iw: np.ndarray, p: np.ndarray,
                        eps: float = 1e-12) -> float:
    """
    JEPA‑style energy where the precision matrix is the weighted Fisher matrix.
    Includes the log‑probability term.
    """
    quad = 0.5 * z @ (Iw @ z)
    log_term = -np.sum(np.log(np.clip(p, eps, 1.0)))
    return quad + log_term

# ----------------------------------------------------------------------
# Hybrid core functions (demonstrate integration)
# ----------------------------------------------------------------------
def generate_hybrid_graph(num_nodes: int,
                          chaos_factor: float,
                          beam_center: float,
                          beam_width: float,
                          time: float,
                          alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    1. Create a chaotic binary adjacency ``A`` and latent vector ``z``.
    2. Convert ``A`` to a weighted matrix ``W(t)`` whose non‑zero entries follow
       a Gaussian‑beam profile (θ is the edge index) and decay according to a
       Caputo fractional derivative of an exponential kernel.
    Returns ``W`` and ``z``.
    """
    A, z = chaotic_graph(num_nodes, chaos_factor)

    # Edge‑index based angle for the Gaussian beam (simple linear mapping)
    edge_indices = np.arange(num_nodes * num_nodes).reshape(num_nodes, num_nodes)
    theta = edge_indices.astype(float)

    # Base beam weighting
    beam_weights = np.vectorize(lambda th: gaussian_beam(th, beam_center, beam_width))(theta)

    # Fractional decay factor (same for all edges, derived from Caputo derivative)
    decay_factor = caputo_derivative(lambda tau: np.exp(-tau), alpha, time)

    W = A.astype(float) * beam_weights * decay_factor
    return W, z

def hybrid_energy_step(num_nodes: int,
                       chaos_factor: float,
                       beam_center: float,
                       beam_width: float,
                       time: float,
                       alpha: float,
                       reference_adj: np.ndarray,
                       learning_rate: float = 0.01) -> tuple[float, np.ndarray]:
    """
    Executes one hybrid iteration:
    * builds weighted adjacency ``W``,
    * draws pheromone probabilities,
    * constructs weighted Fisher matrix,
    * evaluates JEPA energy,
    * takes a gradient step on the latent vector ``z``.
    Returns the energy value and the updated latent vector.
    """
    W, z = generate_hybrid_graph(num_nodes, chaos_factor,
                                 beam_center, beam_width,
                                 time, alpha)

    # 1. Pheromone probabilities and entropy (entropy not used further but
    #    demonstrates the audit side of Parent B)
    p = pheromone_probabilities(num_nodes)
    _ = entropy(p)  # placeholder for audit side effects

    # 2. Fisher information matrix from Parent A's score
    I = fisher_information_matrix(p, center=beam_center, width=beam_width)

    # 3. Reconstruction risk based on current weighted adjacency
    risk = reconstruction_risk_score(W, reference_adj)

    # 4. Weighted Fisher matrix and JEPA energy
    Iw = weighted_fisher_matrix(I, risk)
    energy = weighted_jepa_energy(z, Iw, p)

    # 5. Simple gradient descent on z (∇E = Iw·z)
    grad = Iw @ z
    z_new = z - learning_rate * grad

    return energy, z_new

def run_hybrid_simulation(steps: int = 5,
                          num_nodes: int = 8,
                          chaos_factor: float = 3.9,
                          beam_center: float = 15.0,
                          beam_width: float = 10.0,
                          alpha: float = 0.7,
                          learning_rate: float = 0.05) -> None:
    """
    Runs a short simulation, printing energy at each step.
    A random reference adjacency matrix is generated once.
    """
    # Fixed reference pattern for risk evaluation
    ref_A, _ = chaotic_graph(num_nodes, chaos_factor)
    ref_W, _ = generate_hybrid_graph(num_nodes, chaos_factor,
                                     beam_center, beam_width,
                                     time=1.0, alpha=alpha)

    for step in range(steps):
        t = step + 1.0  # simple monotonic time
        energy, _ = hybrid_energy_step(num_nodes,
                                       chaos_factor,
                                       beam_center,
                                       beam_width,
                                       time=t,
                                       alpha=alpha,
                                       reference_adj=ref_W,
                                       learning_rate=learning_rate)
        print(f"Step {step+1:02d} | Time {t:.2f} | Energy {energy:.6f}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    run_hybrid_simulation()