# DARWIN HAMMER — match 2812, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m702_s0.py (gen5)
# parent_b: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s1.py (gen2)
# born: 2026-05-29T23:46:00Z

"""Hybrid Endpoint-RBF-XGBoost Module
Combines:
- Parent A: geometric endpoint description + RBF surrogate (Gaussian RBF, linear solve).
- Parent B: XGBoost binary logistic gradient/hessian, optimal leaf weight, split gain.

Mathematical bridge:
The surrogate learns a similarity s ∈ [0,1] from geometric features.
That similarity is used as the pseudo‑label y_true in the binary logistic loss of XGBoost.
Thus the RBF linear system (A·w = y) feeds directly into the gradient/hessian
computation (g = p‑y, h = p·(1‑p)), and the resulting gradients determine optimal leaf
weights. The final hybrid score is obtained by weighting the surrogate similarity
with the XGBoost leaf weight and passing it through a sigmoid for a calibrated
output.
"""

import sys
import math
import random
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ---------- Parent A structures ----------
@dataclass
class Endpoint:
    length: float
    width: float
    height: float
    mass: float

class Morphology:
    """Geometric description of an endpoint."""
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint

    def get_geometric_properties(self) -> Vector:
        return (self.endpoint.length,
                self.endpoint.width,
                self.endpoint.height,
                self.endpoint.mass)

class RBF_Surrogate:
    """Gaussian RBF surrogate that solves A·w = y for training data."""
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon
        self.X_train: List[Vector] = []
        self.weights: List[float] = []

    # ---- Linear solver (Gauss‑Jordan with partial pivoting) ----
    def _solve_linear(self, a: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        # Augmented matrix
        m = [row[:] + [rhs] for row, rhs in zip(a, b)]

        for col in range(n):
            # Partial pivot
            pivot_row = max(range(col, n), key=lambda r: abs(m[r][col]))
            if abs(m[pivot_row][col]) < 1e-12:
                raise ValueError("Singular matrix")
            m[col], m[pivot_row] = m[pivot_row], m[col]

            # Normalize pivot row
            pivot_val = m[col][col]
            m[col] = [v / pivot_val for v in m[col]]

            # Eliminate other rows
            for r in range(n):
                if r == col:
                    continue
                factor = m[r][col]
                m[r] = [rv - factor * cv for rv, cv in zip(m[r], m[col])]

        return [row[-1] for row in m]

    def fit(self, X: List[Vector], y: List[float]) -> None:
        """Fit surrogate: solve (K)·w = y where K_ij = gaussian(||x_i‑x_j||)."""
        self.X_train = X
        n = len(X)
        K = [[gaussian(euclidean(X[i], X[j]), self.epsilon) for j in range(n)] for i in range(n)]
        self.weights = self._solve_linear(K, y)

    def predict(self, X_new: List[Vector]) -> List[float]:
        """Predict similarity for new feature vectors."""
        if not self.X_train or not self.weights:
            raise RuntimeError("Surrogate not fitted")
        preds = []
        for x in X_new:
            val = sum(w * gaussian(euclidean(x, xi), self.epsilon)
                      for w, xi in zip(self.weights, self.X_train))
            preds.append(val)
        return preds

# ---------- Parent B structures ----------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Optimal leaf weight for XGBoost."""
    return -gradient_sum / (hessian_sum + reg_lambda)

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """XGBoost split gain."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent_gain = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    left_gain = gl ** 2 / (hl + reg_lambda)
    right_gain = gr ** 2 / (hr + reg_lambda)
    return 0.5 * (left_gain + right_gain - parent_gain) - gamma

# ---------- Hybrid Functions ----------
def compute_surrogate_similarity(endpoints: List[Endpoint],
                                 surrogate: RBF_Surrogate) -> List[float]:
    """Map endpoints → geometric vectors → surrogate similarity."""
    feats = [Morphology(ep).get_geometric_properties() for ep in endpoints]
    return surrogate.predict(feats)

def hybrid_grad_hess_from_similarity(similarity: List[float],
                                     margins: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treat surrogate similarity as pseudo‑labels y_true and compute
    binary logistic gradients/hessians w.r.t. given margins.
    """
    y = np.clip(np.array(similarity), 0.0, 1.0)  # ensure valid probability range
    m = np.array(margins)
    return binary_logistic_grad_hess(y, m)

def hybrid_leaf_weights_from_grad_hess(grad: np.ndarray,
                                      hess: np.ndarray,
                                      reg_lambda: float = 1.0) -> np.ndarray:
    """Compute optimal leaf weight for each instance."""
    return -grad / (hess + reg_lambda)

def hybrid_final_score(endpoints: List[Endpoint],
                       surrogate: RBF_Surrogate,
                       margins: List[float],
                       reg_lambda: float = 1.0) -> List[float]:
    """
    Full hybrid pipeline:
    1. Surrogate similarity s_i.
    2. Gradient/Hessian from logistic loss using s_i as label.
    3. Optimal leaf weight w_i = -g_i/(h_i+λ).
    4. Final calibrated score = sigmoid(s_i * w_i).
    """
    s = np.array(compute_surrogate_similarity(endpoints, surrogate))
    g, h = hybrid_grad_hess_from_similarity(s.tolist(), margins)
    w = hybrid_leaf_weights_from_grad_hess(g, h, reg_lambda)
    raw = s * w
    return sigmoid(raw).tolist()

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create synthetic endpoints
    random.seed(42)
    eps = [Endpoint(length=random.uniform(1, 10),
                    width=random.uniform(1, 5),
                    height=random.uniform(0.5, 3),
                    mass=random.uniform(0.1, 2.0)) for _ in range(6)]

    # Generate synthetic target similarity values (e.g., from an external oracle)
    target_sim = [random.random() for _ in eps]

    # Fit RBF surrogate on the same points (using their own features)
    features = [Morphology(e).get_geometric_properties() for e in eps]
    surrogate = RBF_Surrogate(epsilon=0.5)
    surrogate.fit(features, target_sim)

    # Random margins as would be produced by a weak learner
    margins = [random.uniform(-2, 2) for _ in eps]

    # Run hybrid pipeline
    final = hybrid_final_score(eps, surrogate, margins, reg_lambda=1.0)

    print("Surrogate similarity:", compute_surrogate_similarity(eps, surrogate))
    print("Margins:", margins)
    print("Hybrid final scores:", final)