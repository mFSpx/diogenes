# DARWIN HAMMER — match 1268, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1150_s0.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s1.py (gen3)
# born: 2026-05-29T23:35:01Z

"""Hybrid Decision–Fractional Tree Algorithm
Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hoeffding_tre_m1150_s0.py (Hoeffding‑bound driven decision tree
  with tropical max‑plus algebra for piecewise‑linear convex decision surfaces)
- hybrid_hybrid_fisher_locali_hybrid_caputo_fracti_m1185_s1.py (Fisher‑information scoring,
  SSIM similarity, and Caputo fractional decay kernel)

Mathematical bridge:
The splitting criterion of a Hoeffding tree is a statistical bound B that guarantees,
with confidence 1‑δ, that the observed attribute merit differs from the true merit by ≤B.
We embed this bound as a *scale* for a fractional decay kernel κₐ(t)=t^{α‑1}/Γ(α) that
modulates edge weights of the tree.  The tropical max‑plus algebra provides a
piecewise‑linear convex evaluation of the decision surface:
    φ(x)=⊕_i (w_i ⊗ x_i) = max_i (w_i + x_i).
The final fused cost for an edge e is

    C_e = ( Σ_i w_i·κₐ(t_i) ) · (1 – S_sim) + λ·F_score
          × (1 + B)

where S_sim is the structural‑similarity index, F_score the Fisher‑information score,
λ a regularisation constant and B the Hoeffding bound.  This code implements the
core operations and demonstrates the hybrid behaviour.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------


def hoeffding_bound(range_R: float, delta: float, n: int) -> float:
    """
    Hoeffding bound B = sqrt( R^2 * ln(1/δ) / (2n) )
    Guarantees that the true mean lies within B of the empirical mean
    with probability 1‑δ.
    """
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((range_R ** 2) * math.log(1.0 / delta) / (2.0 * n))


def tropical_max_plus(vector: np.ndarray) -> float:
    """
    Tropical max‑plus evaluation φ(x)=max_i (x_i).
    In max‑plus algebra the addition is max and multiplication is plus,
    so a linear form becomes max_i (w_i + x_i).  Here we only need the max.
    """
    if vector.size == 0:
        raise ValueError("input vector must not be empty")
    return float(np.max(vector))


def tropical_linear(weights: np.ndarray, features: np.ndarray) -> float:
    """
    Tropical linear form: max_i (w_i + x_i)
    """
    if weights.shape != features.shape:
        raise ValueError("weights and features must have the same shape")
    return float(np.max(weights + features))


def fractional_decay(alpha: float, t: float) -> float:
    """
    Caputo fractional decay kernel κₐ(t) = t^{α‑1} / Γ(α)
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if t < 0:
        raise ValueError("time t must be non‑negative")
    return (t ** (alpha - 1)) / math.gamma(alpha)


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity used by the Fisher score."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher‑information score for a scalar angle.
    Implements the formula from parent B.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Structural Similarity Index (SSIM) for 1‑D signals.
    Simplified version matching parent B's intent.
    """
    if x.shape != y.shape:
        raise ValueError("signals must have the same shape")
    if x.size == 0:
        raise ValueError("signals must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid cost functional
# ----------------------------------------------------------------------


def fused_edge_cost(weights: np.ndarray,
                    times: np.ndarray,
                    alpha: float,
                    ssim_val: float,
                    lambda_: float,
                    fisher: float,
                    hoeffding_B: float) -> float:
    """
    Compute the fused cost for a tree edge.

    Parameters
    ----------
    weights : np.ndarray
        Edge weight vector w_i.
    times   : np.ndarray
        Corresponding time stamps t_i (same shape as weights).
    alpha   : float
        Fractional order for the decay kernel.
    ssim_val: float
        Structural similarity factor in [0,1].
    lambda_ : float
        Regularisation coefficient for the Fisher term.
    fisher  : float
        Fisher‑information score for the packet/feature.
    hoeffding_B : float
        Hoeffding bound scaling factor.

    Returns
    -------
    float
        The hybrid cost C_e.
    """
    if weights.shape != times.shape:
        raise ValueError("weights and times must have the same shape")
    decay = np.vectorize(fractional_decay)(alpha, times)
    weighted_decay = np.sum(weights * decay)
    cost = weighted_decay * (1.0 - ssim_val) + lambda_ * fisher
    # Apply Hoeffding scaling (1 + B) to respect statistical confidence
    return cost * (1.0 + hoeffding_B)


# ----------------------------------------------------------------------
# Simple hybrid decision tree node
# ----------------------------------------------------------------------


class HybridNode:
    """
    Minimal node for a Hoeffding‑driven decision tree whose split evaluation
    uses tropical max‑plus algebra and whose edge costs are given by
    `fused_edge_cost`.
    """

    def __init__(self,
                 feature_idx: int,
                 split_value: float,
                 left=None,
                 right=None,
                 weight: float = 1.0):
        self.feature_idx = feature_idx
        self.split_value = split_value
        self.left = left
        self.right = right
        self.weight = weight  # base weight for this node

    def route(self, sample: np.ndarray) -> 'HybridNode':
        """
        Route a sample down the tree using a tropical decision surface:
        if max(feature - split, 0) > 0 go right, else left.
        """
        feature = sample[self.feature_idx]
        # Tropical evaluation: max(0, feature - split)
        decision = tropical_max_plus(np.array([0.0, feature - self.split_value]))
        if decision > 0:
            return self.right if self.right else self
        else:
            return self.left if self.left else self

    def compute_cost(self,
                     sample: np.ndarray,
                     times: np.ndarray,
                     alpha: float,
                     lambda_: float,
                     delta: float,
                     range_R: float,
                     fisher_params: tuple) -> float:
        """
        Compute the hybrid cost for traversing this node.

        Parameters
        ----------
        sample          : np.ndarray of feature values.
        times           : np.ndarray of timestamps aligned with `sample`.
        alpha, lambda_  : fractional and regularisation parameters.
        delta, range_R  : Hoeffding parameters.
        fisher_params  : (theta, center, width) for Fisher score.

        Returns
        -------
        float
            Hybrid cost for this node.
        """
        n = len(sample)
        B = hoeffding_bound(range_R, delta, n)

        # Fisher score based on a chosen feature (here we use the split feature)
        theta, center, width = fisher_params
        fisher = fisher_score(theta, center, width)

        # SSIM between the sample and a simple reference (here the split value repeated)
        ref = np.full_like(sample, self.split_value)
        ssim_val = ssim(sample, ref)

        # Edge weight vector is just the node weight repeated
        w_vec = np.full_like(sample, self.weight)

        return fused_edge_cost(w_vec, times, alpha, ssim_val,
                               lambda_, fisher, B)


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------


def demo_fused_cost():
    """Run a tiny example of the fused cost computation."""
    weights = np.array([0.8, 1.2, 0.5])
    times = np.array([1.0, 2.0, 3.0])
    alpha = 0.7
    ssim_val = 0.85
    lambda_ = 0.3
    fisher = fisher_score(theta=0.5, center=0.0, width=1.0)
    B = hoeffding_bound(range_R=1.0, delta=0.05, n=100)

    cost = fused_edge_cost(weights, times, alpha, ssim_val,
                           lambda_, fisher, B)
    print(f"Demo fused cost: {cost:.6f}")


def demo_tree_routing():
    """Create a tiny tree and route a random sample."""
    # Build a 2‑level tree
    leaf_left = HybridNode(feature_idx=0, split_value=0.0, weight=0.6)
    leaf_right = HybridNode(feature_idx=0, split_value=0.0, weight=0.9)
    root = HybridNode(feature_idx=1, split_value=0.5,
                      left=leaf_left, right=leaf_right, weight=1.0)

    sample = np.array([0.3, 0.7])          # two features
    times = np.array([1.0, 2.0])           # timestamps for the two features

    node = root.route(sample)
    cost = node.compute_cost(sample,
                             times,
                             alpha=0.8,
                             lambda_=0.2,
                             delta=0.01,
                             range_R=1.0,
                             fisher_params=(0.7, 0.0, 1.0))
    print(f"Routed to node with weight {node.weight}, cost = {cost:.6f}")


def demo_tropical_evaluation():
    """Show tropical max‑plus on a random vector."""
    vec = np.random.randn(5)
    result = tropical_max_plus(vec)
    print(f"Tropical max‑plus of {vec.round(3)} = {result:.3f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running hybrid algorithm smoke test...")
    demo_fused_cost()
    demo_tree_routing()
    demo_tropical_evaluation()
    print("Smoke test completed successfully.")