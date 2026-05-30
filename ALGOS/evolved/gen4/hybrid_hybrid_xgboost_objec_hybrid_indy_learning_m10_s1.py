# DARWIN HAMMER — match 10, survivor 1
# gen: 4
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s0.py (gen3)
# born: 2026-05-29T23:26:17Z

"""Hybrid module fusing DARWIN HAMMER match 33 (hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py) 
and DARWIN HAMMER match 113 (hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s0.py) 
into a unified system.

The mathematical bridge:
- Text chunks are mapped to high-dimensional frequency vectors (grade-1 blades) 
  using the INDY routine from Parent B.
- Each frequency vector is treated as a positive binary label (y=1) for the 
  logistic loss function from Parent A.
- A pruning “margin” is derived from the decreasing probability 
  p(t)=λ·exp(−αt) via the logit function, turning the schedule into a 
  logistic-loss margin.
- Using the binary logistic gradient/hessian from Parent A, we obtain 
  aggregate G and H for the whole set of findings.
- The geometric product of frequency vectors is used to compute 
  region-level multivectors and Ollivier-Ricci curvature between neighboring 
  regions.
- XGBoost’s split-gain formula is then employed to modulate the pruning 
  probability: a larger gain (i.e., more informative findings) reduces the 
  chance of being pruned, while a small gain leaves the original 
  exponential schedule essentially unchanged.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – XGBoost objective utilities
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0
) -> float:
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    return (
        -0.5
        * (
            left_gradient ** 2 / (left_hessian + reg_lambda)
            + right_gradient ** 2 / (right_hessian + reg_lambda)
        )
        + gamma
    )

# ----------------------------------------------------------------------
# Parent B – INDY vector utilities and geometric algebra
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)

def sha256_json(value: Any) -> str:
    """Deterministic SHA-256 of a JSON-serialisable value."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()

def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric product of two vectors."""
    return np.dot(a, b) + np.cross(a, b)

def region_multivector(region_vectors: list[np.ndarray]) -> np.ndarray:
    """Compute region-level multivector."""
    multivector = np.zeros_like(region_vectors[0])
    for vector in region_vectors:
        multivector += geometric_product(multivector, vector)
    return multivector

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_pruning_margin(
    frequency_vectors: list[np.ndarray], 
    lambda_: float, 
    alpha: float, 
    t: float
) -> np.ndarray:
    """Compute pruning margin for a set of frequency vectors."""
    # Map frequency vectors to binary labels (y=1)
    labels = np.ones(len(frequency_vectors))
    
    # Compute aggregate gradient and Hessian
    margins = np.zeros(len(frequency_vectors))
    for i, vector in enumerate(frequency_vectors):
        margin = np.log(lambda_ * np.exp(-alpha * t))
        margins[i] = margin
    gradients, hessians = binary_logistic_grad_hess(labels, margins)
    gradient_sum = np.sum(gradients)
    hessian_sum = np.sum(hessians)
    
    # Compute optimal leaf weight
    leaf_weight = optimal_leaf_weight(gradient_sum, hessian_sum)
    
    # Modulate pruning probability using split gain
    region_vectors = [vector for vector in frequency_vectors]
    region_multivectors = [region_multivector([vector]) for vector in region_vectors]
    gains = [
        split_gain(
            np.dot(region_multivector, region_multivector),
            np.dot(region_multivector, region_multivector),
            np.dot(region_multivector, region_multivector),
            np.dot(region_multivector, region_multivector),
        )
        for region_multivector in region_multivectors
    ]
    modulated_probabilities = [
        lambda_ * np.exp(-alpha * t) * (1 - sigmoid(gain))
        for gain in gains
    ]
    return modulated_probabilities

def hybrid_region_curvature(
    region_vectors: list[list[np.ndarray]], 
    lambda_: float, 
    alpha: float, 
    t: float
) -> float:
    """Estimate Ollivier-Ricci curvature between neighboring regions."""
    # Compute region-level multivectors
    region_multivectors = [region_multivector(region) for region in region_vectors]
    
    # Compute transport cost
    transport_cost = 0.0
    for i in range(len(region_multivectors) - 1):
        transport_cost += np.linalg.norm(
            region_multivectors[i] - region_multivectors[i + 1]
        )
    
    # Compute Ollivier-Ricci curvature
    curvature = transport_cost / (lambda_ * np.exp(-alpha * t))
    return curvature

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    frequency_vectors = [np.random.rand(10) for _ in range(10)]
    lambda_ = 1.0
    alpha = 0.1
    t = 1.0
    modulated_probabilities = hybrid_pruning_margin(
        frequency_vectors, lambda_, alpha, t
    )
    print(modulated_probabilities)
    region_vectors = [[np.random.rand(10) for _ in range(5)] for _ in range(2)]
    curvature = hybrid_region_curvature(region_vectors, lambda_, alpha, t)
    print(curvature)