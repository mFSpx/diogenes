# DARWIN HAMMER — match 4396, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_caputo_fracti_m258_s2.py (gen3)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s2.py (gen3)
# born: 2026-05-29T23:55:28Z

import numpy as np
from dataclasses import dataclass
from math import exp, sqrt, gamma

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
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(coeffs.shape[0], dtype=float).reshape((-1,) + (1,) * x.ndim)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents * x
    return np.max(terms, axis=0)

def caputo_fractional_integral(values: np.ndarray, alpha: float, dt: float = 1.0) -> np.ndarray:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    values = np.asarray(values, dtype=float)
    n = values.size
    coeff = (dt ** alpha) / gamma(alpha + 1.0)
    j = np.arange(n, dtype=float)
    w = (j + 1) ** alpha - j ** alpha
    integral = np.empty_like(values)
    for idx in range(n):
        integral[idx] = coeff * np.dot(w[:idx + 1][::-1], values[:idx + 1])
    return integral

def compute_priority_sequence(tree_edges, morphologies):
    edges = sorted(tree_edges, key=lambda e: e[2])
    priorities = []
    for _, child, _ in edges:
        if child not in morphologies:
            raise KeyError(f"Missing morphology for node {child}")
        priorities.append(recovery_priority(morphologies[child]))
    return np.array(priorities, dtype=float)

def tropical_recovery_polynomial(priorities):
    coeffs = np.full(len(priorities), -np.inf, dtype=float)
    current_max = -np.inf
    for i, p in enumerate(priorities):
        current_max = max(current_max, p)
        coeffs[i] = current_max
    return coeffs

def hybrid_fractional_tropical_recovery(tree_edges, morphologies, alpha=0.5, dt=1.0):
    priorities = compute_priority_sequence(tree_edges, morphologies)
    maxima = []
    for depth in range(1, len(priorities) + 1):
        coeffs = tropical_recovery_polynomial(priorities[:depth])
        val = t_polyval(coeffs, depth)
        maxima.append(val)
    maxima = np.array(maxima, dtype=float)
    frac_integral = caputo_fractional_integral(maxima, alpha, dt)
    return frac_integral

def decision_hygiene_score(poly_coeffs, x_grid):
    vals = t_polyval(poly_coeffs, x_grid)
    shift = -np.min(vals)
    probs = (vals + shift) + 1e-12
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def improved_hybrid_fractional_tropical_recovery(tree_edges, morphologies, alpha=0.5, dt=1.0, gamma=0.1):
    priorities = compute_priority_sequence(tree_edges, morphologies)
    maxima = []
    for depth in range(1, len(priorities) + 1):
        coeffs = tropical_recovery_polynomial(priorities[:depth])
        val = t_polyval(coeffs, depth)
        maxima.append(val)
    maxima = np.array(maxima, dtype=float)
    frac_integral = caputo_fractional_integral(maxima, alpha, dt)
    weighted_frac_integral = (1 - gamma) * frac_integral + gamma * maxima
    return weighted_frac_integral

def improved_decision_hygiene_score(poly_coeffs, x_grid, beta=0.1):
    vals = t_polyval(poly_coeffs, x_grid)
    shift = -np.min(vals)
    probs = (vals + shift) + 1e-12
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    return (1 - beta) * entropy + beta * np.mean(probs)

# Example usage
tree_edges = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
morphologies = {0: Morphology(1, 1, 1, 1), 1: Morphology(2, 2, 2, 2), 2: Morphology(3, 3, 3, 3), 3: Morphology(4, 4, 4, 4)}
x_grid = np.array([1, 2, 3])

result = improved_hybrid_fractional_tropical_recovery(tree_edges, morphologies)
print(result)

score = improved_decision_hygiene_score(tropical_recovery_polynomial(compute_priority_sequence(tree_edges, morphologies)), x_grid)
print(score)