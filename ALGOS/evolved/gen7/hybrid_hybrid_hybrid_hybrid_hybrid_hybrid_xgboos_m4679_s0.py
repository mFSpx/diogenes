# DARWIN HAMMER — match 4679, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s3.py (gen6)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_hybrid_m1262_s2.py (gen6)
# born: 2026-05-29T23:57:24Z

import math
import numpy as np
import random
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return sphericity * math.exp(-0.5 * z * z)

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Optimal leaf weight for XGBoost."""
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def tropical_max_plus_algebra(C: np.ndarray) -> float:
    # Define tropical max-plus operations
    def t_add(a: float, b: float) -> float:
        return max(a, b)

    def t_mul(a: float, b: float) -> float:
        return a + b

    # Compute tropical max-plus product
    product = C

    def compute_product() -> float:
        result = 0
        for i in range(len(C)):
            result = t_add(result, t_mul(C[i], product))
        return result

    return compute_product()

def compute_illumination(theta: float, center: float, width: float, sphericity: float) -> float:
    return gaussian_beam(theta, center, width, sphericity)

def compute_similarity(y_true: np.ndarray, margin: np.ndarray) -> float:
    g, h = binary_logistic_grad_hess(y_true, margin)
    # Here we integrate the tropical max-plus algebra with the SSIM metric
    return sigmoid(margin) + tropical_max_plus_algebra(g)

def hybrid_operation(margin: np.ndarray, theta: float, center: float, width: float, sphericity: float) -> float:
    # Here we integrate the XGBoost loss function with the gaussian beam and tropical max-plus algebra
    illumination = compute_illumination(theta, center, width, sphericity)
    similarity = compute_similarity(np.zeros_like(margin), margin)
    return illumination + similarity

if __name__ == "__main__":
    # Smoke test
    margin = np.array([1.0, 2.0, 3.0])
    theta = 1.0
    center = 2.0
    width = 3.0
    sphericity = 0.5

    try:
        result = hybrid_operation(margin, theta, center, width, sphericity)
        print(result)
    except Exception as e:
        print(f"Error: {e}")