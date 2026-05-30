# DARWIN HAMMER — match 4725, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s6.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s2.py (gen5)
# born: 2026-05-29T23:57:43Z

"""
Hybrid Algorithm combining fractional Caputo kernel weighting with epistemic‑certainty‑aware
minimum‑cost tree construction, decreasing‑rate pruning and Gaussian beam & Fisher information.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s6.py (Caputo kernel + Voronoi geometry)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1673_s2.py (Hybrid Cockpit-Metrics & Liquid-Time-Constant Diffusion)

Mathematical bridge:
The Caputo kernel provides a time‑dependent scaling factor κ(t;α)=t^{α‑1}/Γ(α) that can be
applied to any scalar cost. The Gaussian beam & Fisher information provide a weighting factor
for the edge cost. By multiplying the original cost with κ(t;α) and the weighting factor, we
obtain a unified, time‑varying edge weight

    w_{ij}(t) = d(p_i,p_j) · ϕ(flag_{ij}) · κ(t;α) · I(θ)

which simultaneously embeds fractional‑calculus dynamics, epistemic‑certainty weighting and
information weighting. The resulting weighted graph is fed to a classic minimum‑spanning‑tree
(Kruskal) algorithm, after which a decreasing‑rate pruning schedule removes edges with probability
p(t)=λ·e^{‑α·t}.
"""

import math
import random
import sys
import pathlib
from typing import List, Tuple, Dict, Hashable
import numpy as np

# ----------------------------------------------------------------------
# Fractional calculus (Parent A)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857
])

def _gamma(z: float) -> float:
    """Lanczos approximation of the Gamma function."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Unnormalised Caputo kernel κ(t;α)=t^{α‑1}/Γ(α) for a vector of time indices."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return t ** (alpha - 1) / _gamma(alpha)

# ----------------------------------------------------------------------
# Gaussian beam & Fisher information (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single-parameter Gaussian model.
    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_edge_weight(alpha: float, t: float, d: float, flag: float, theta: float, center: float, width: float) -> float:
    """Hybrid edge weight combining Caputo kernel, epistemic certainty and Fisher information."""
    return d * flag * caputo_kernel(alpha, np.array([t]))[0] * fisher_score(theta, center, width)

def hybrid_mst(graph: Dict[int, List[Tuple[int, float, float, float, float, float, float]]], alpha: float, t: float) -> List[Tuple[int, int, float]]:
    """Hybrid minimum spanning tree algorithm combining Caputo kernel, epistemic certainty and Fisher information."""
    mst = []
    visited = set()
    edges = []
    for node, neighbors in graph.items():
        for neighbor, weight, flag, theta, center, width, _ in neighbors:
            weight = hybrid_edge_weight(alpha, t, weight, flag, theta, center, width)
            edges.append((node, neighbor, weight))
    edges.sort(key=lambda x: x[2])
    for node, neighbor, weight in edges:
        if node not in visited or neighbor not in visited:
            mst.append((node, neighbor, weight))
            visited.add(node)
            visited.add(neighbor)
    return mst

def hybrid_pruning(mst: List[Tuple[int, int, float]], alpha: float, t: float, lambda_: float) -> List[Tuple[int, int, float]]:
    """Hybrid pruning algorithm combining Caputo kernel and decreasing-rate pruning."""
    pruned_mst = []
    for node, neighbor, weight in mst:
        if random.random() > lambda_ * math.exp(-alpha * t):
            pruned_mst.append((node, neighbor, weight))
    return pruned_mst

if __name__ == "__main__":
    graph = {
        0: [(1, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0), (2, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0)],
        1: [(0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0), (2, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0)],
        2: [(0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0), (1, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0)]
    }
    alpha = 0.5
    t = 1.0
    lambda_ = 0.1
    mst = hybrid_mst(graph, alpha, t)
    pruned_mst = hybrid_pruning(mst, alpha, t, lambda_)
    print(pruned_mst)