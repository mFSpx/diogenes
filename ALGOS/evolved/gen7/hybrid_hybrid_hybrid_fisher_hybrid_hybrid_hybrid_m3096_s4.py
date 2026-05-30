# DARWIN HAMMER — match 3096, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_omni_c_hybrid_ternary_lens__m1536_s0.py (gen6)
# born: 2026-05-29T23:47:56Z

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

    seed = np.random.rand(num_nodes)
    A = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        x = seed[i]
        for j in range(num_nodes):
            x = logistic_map(x)
            A[i, j] = 1 if x > 0.5 else 0

    z = np.random.randn(num_nodes)
    return A, z

def pheromone_probabilities(num_nodes: int) -> np.ndarray:
    raw = np.random.rand(num_nodes)
    return raw / raw.sum()

def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def fisher_information_matrix(p: np.ndarray, center: float, width: float) -> np.ndarray:
    scores = np.vectorize(lambda pi: fisher_score(pi, center, width))(p)
    return np.diag(scores)

def reconstruction_risk_score(weighted_adj: np.ndarray,
                              reference: np.ndarray) -> float:
    return 1.0 - ssim(weighted_adj.ravel(), reference.ravel())

def weighted_fisher_matrix(I: np.ndarray, risk_score: float) -> np.ndarray:
    return I * risk_score

def weighted_jepa_energy(z: np.ndarray, Iw: np.ndarray, p: np.ndarray,
                        eps: float = 1e-12) -> float:
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
    A, z = chaotic_graph(num_nodes, chaos_factor)

    edge_theta = np.arange(np.size(A))
    A_gaussian = A * gaussian_beam(edge_theta, beam_center, beam_width)

    def exp_kernel(t: float) -> float:
        return np.exp(-t)

    W = A_gaussian * (1 - caputo_derivative(exp_kernel, alpha, time))

    return W, z

def compute_hybrid_energy(num_nodes: int,
                          chaos_factor: float,
                          beam_center: float,
                          beam_width: float,
                          time: float,
                          alpha: float,
                          reference: np.ndarray) -> float:
    W, z = generate_hybrid_graph(num_nodes, chaos_factor, beam_center, beam_width, time, alpha)

    p = pheromone_probabilities(num_nodes)
    I = fisher_information_matrix(p, beam_center, beam_width)

    risk_score = reconstruction_risk_score(W, reference)
    Iw = weighted_fisher_matrix(I, risk_score)

    return weighted_jepa_energy(z, Iw, p)

def hybrid_workflow(num_nodes: int,
                     chaos_factor: float,
                     beam_center: float,
                     beam_width: float,
                     time: float,
                     alpha: float,
                     reference: np.ndarray) -> None:
    energy = compute_hybrid_energy(num_nodes, chaos_factor, beam_center, beam_width, time, alpha, reference)
    print(f'Hybrid energy: {energy}')

if __name__ == '__main__':
    num_nodes = 10
    chaos_factor = 3.9
    beam_center = 0.5
    beam_width = 0.1
    time = 1.0
    alpha = 0.5
    reference = np.random.rand(num_nodes, num_nodes)

    hybrid_workflow(num_nodes, chaos_factor, beam_center, beam_width, time, alpha, reference)