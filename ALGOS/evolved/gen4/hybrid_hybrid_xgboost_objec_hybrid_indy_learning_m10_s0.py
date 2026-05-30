# DARWIN HAMMER — match 10, survivor 0
# gen: 4
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s0.py (gen3)
# born: 2026-05-29T23:26:17Z

# hybrid_xgboost_objective_hybrid_ternary_lens_audit_decreasing_pruning_geometric_geomet_m113_m33_s3.py
# DARWIN HAMMER — match 113, 33, survivor 3
# gen: 4
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s2.py (gen2)
# parent_b: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s0.py (gen3)
# born: 2026-05-29T23:30:15Z

"""
Hybrid module combining XGBoost objective mathematics with ternary lens audit pruning
and INDY vector chunking with geometric algebra Voronoi partitioning.

Mathematical bridge:
- Each text chunk is mapped to a high-dimensional frequency vector whose
  axes are the ontology terms loaded by the INDY routine.
- A set of seed vectors defines a Voronoi diagram in the same space; each
  chunk-vector is assigned to the nearest seed region.
- For every region we aggregate the multivectors of its members (pairwise
  geometric product) to obtain a region-level multivector.
- A set of audit findings is treated as a positive binary label (y=1).
- A pruning “margin’’ is derived from the decreasing probability p(t)=λ·exp(−αt)
  via the logit function, turning the schedule into a logistic-loss margin.
- Using the binary logistic gradient/hessian from the XGBoost objective we obtain aggregate
  G and H for the whole set of findings.
- The geometric product of the region multivectors is used to estimate the Ollivier-Ricci curvature
  between neighboring regions, which is then modulated by the pruning margin.
- A larger gain (i.e. more informative findings) reduces the chance of being pruned,
  while a small gain leaves the original exponential schedule essentially unchanged.

The resulting functions expose a hybrid pruning operation that is fully
differentiable in the XGBoost sense yet respects the original decreasing-rate
policy and geometric connectivity analysis.
"""

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# INDY vector utilities (parent A)
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


# ----------------------------------------------------------------------
# XGBoost objective utilities (parent B)
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
    gamma: float = 0.0
) -> float:
    return 0.5 * np.log(
        (left_hessian + reg_lambda) / (right_hessian + reg_lambda)
    ) + gamma


# ----------------------------------------------------------------------
# Geometric algebra utilities
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geometric product of two multivectors."""
    return np.tensordot(a, b, axes=([0], [0])) + np.tensordot(
        a, b, axes=([0], [1])
    )


def voronoi_regions(chunk_vectors: np.ndarray, seed_vectors: np.ndarray) -> np.ndarray:
    """Assign each chunk-vector to the nearest seed region."""
    distances = np.linalg.norm(chunk_vectors[:, None] - seed_vectors[None, :], axis=2)
    return np.argmin(distances, axis=1)


def region_multivectors(chunk_vectors: np.ndarray, seed_vectors: np.ndarray) -> np.ndarray:
    """Aggregate the multivectors of each region."""
    regions = voronoi_regions(chunk_vectors, seed_vectors)
    region_multivectors = []
    for i in np.unique(regions):
        region_vectors = chunk_vectors[regions == i]
        multivector = np.zeros(region_vectors.shape[0])
        for j in range(region_vectors.shape[0]):
            multivector = geometric_product(region_vectors[j], multivector)
        region_multivectors.append(multivector)
    return np.array(region_multivectors)


def ollivier_ricci_curvature(
    region_multivectors: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    """Estimate the Ollivier-Ricci curvature between neighboring regions."""
    transport_costs = np.sum(region_multivectors * region_multivectors, axis=1)
    return 1 - np.exp(-distances) / transport_costs


# ----------------------------------------------------------------------
# Hybrid pruning operation
# ----------------------------------------------------------------------
def hybrid_pruning(
    chunk_vectors: np.ndarray,
    seed_vectors: np.ndarray,
    audit_findings: np.ndarray,
    lambda_: float,
    alpha: float,
    gamma: float = 0.0,
    reg_lambda: float = 1.0,
) -> np.ndarray:
    """Modulate the pruning schedule by the Ollivier-Ricci curvature."""
    region_multivectors = region_multivectors(chunk_vectors, seed_vectors)
    distances = np.linalg.norm(seed_vectors[:, None] - seed_vectors[None, :], axis=2)
    curvature = ollivier_ricci_curvature(region_multivectors, distances)
    margins = -alpha * np.log(lambda_ * np.exp(-alpha * distances))
    binary_logistic_grad, binary_logistic_hess = binary_logistic_grad_hess(
        audit_findings, margins
    )
    split_gains = split_gain(
        binary_logistic_grad,
        binary_logistic_hess,
        np.zeros_like(binary_logistic_grad),
        np.zeros_like(binary_logistic_hess),
        reg_lambda=reg_lambda,
        gamma=gamma,
    )
    return curvature * np.exp(-split_gains)


if __name__ == "__main__":
    import numpy as np

    # Test the hybrid pruning operation
    chunk_vectors = np.random.rand(10, 5)
    seed_vectors = np.random.rand(3, 5)
    audit_findings = np.random.randint(2, size=10)
    lambda_ = 0.5
    alpha = 0.1
    gamma = 0.0
    reg_lambda = 1.0

    hybrid_pruning_result = hybrid_pruning(
        chunk_vectors, seed_vectors, audit_findings, lambda_, alpha, gamma, reg_lambda
    )
    print(hybrid_pruning_result)