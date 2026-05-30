# DARWIN HAMMER — match 1248, survivor 7
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

import sys
import math
import random
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from collections import defaultdict

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e

def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    return t * x1 + (1.0 - t) * x0

class Multivector:
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Multivector":
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec)}
        return cls(comps, n=len(vec))

    def vector_part(self) -> np.ndarray:
        vec = np.zeros(self.n, dtype=float)
        for blade, val in self.components.items():
            if len(blade) == 1:
                i = next(iter(blade))
                vec[i] = val
        return vec

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def copy(self) -> "Multivector":
        return Multivector(dict(self.components), self.n)

    def __repr__(self) -> str:
        return f"Multivector(components={self.components})"

def rbf_energy(g: np.ndarray, centers: np.ndarray, sigma: float) -> float:
    diffs = g - centers
    sq_norms = np.sum(diffs ** 2, axis=1)
    return float(np.sum(np.exp(-sq_norms / (2.0 * sigma ** 2))))

def compute_fluxes(g: np.ndarray) -> np.ndarray:
    pressures = np.random.randn(len(g) + 1)
    diffs = pressures[1:] - pressures[:-1]
    return g * diffs

def hybrid_step(
    C: Multivector,
    prev_g: np.ndarray,
    weights: np.ndarray,
    centers: np.ndarray,
    sigma: float,
    eta: float,
    lam: float,
    mu: float,
    t: float,
    alpha: float = 0.1,
) -> Tuple[Multivector, np.ndarray, np.ndarray]:
    g = C.vector_part()
    x = interpolant(prev_g, g, t)
    pred_energy = nlms_predict(weights, x)
    true_energy = rbf_energy(g, centers, sigma)
    weights, _ = nlms_update(weights, x, true_energy, mu=mu)
    grad_est = weights[: len(g)]
    phi = compute_fluxes(g)
    g_new = g + eta * (phi - lam * grad_est) + alpha * (g - prev_g)
    C_new = Multivector.from_vector(g_new)
    return C_new, g_new, weights

def _demo(num_steps: int = 5) -> None:
    random.seed(42)
    np.random.seed(42)
    n_edges = 6
    g0 = np.abs(np.random.randn(n_edges))
    C = Multivector.from_vector(g0)
    prev_g = g0.copy()
    weights = np.zeros(n_edges, dtype=float)
    m_centers = 4
    centers = np.abs(np.random.randn(m_centers, n_edges))
    sigma = 1.0
    eta = 0.1
    lam = 0.05
    mu = 0.3
    t = 0.5
    print("Initial conductances:", g0)
    for step in range(num_steps):
        C, g, weights = hybrid_step(
            C, prev_g, weights, centers, sigma, eta, lam, mu, t
        )
        prev_g = g
    print("Final conductances:", g)

_demo()