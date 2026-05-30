# DARWIN HAMMER — match 3226, survivor 2
# gen: 5
# parent_a: hybrid_geometric_product_hybrid_hybrid_fisher_m1171_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1056_s0.py (gen4)
# born: 2026-05-29T23:48:36Z

"""Hybrid Geometric‑Fisher‑Bandit Algorithm
========================================

This module fuses the two parent algorithms:

* **Parent A** – *geometric_product* + *Fisher information* + *minimum‑cost tree*  
  (functions: ``gaussian_beam``, ``fisher_precision`` and ``tree_cost``).

* **Parent B** – *Bayesian‑Bandit* with *Ollivier‑Ricci curvature* modulation  
  (structures: ``BanditAction`` / ``BanditUpdate`` and the curvature‑adjusted UCB).

**Mathematical bridge** – The bridge is the *precision matrix* (inverse covariance) that
the Fisher information provides.  In a geometric algebra setting a multivector can
store a scalar (the Gaussian weight) together with a bivector that encodes the
precision.  The curvature derived from that bivector is then used to scale the
confidence bounds of the bandit algorithm.  Consequently the three core operations
are:

1. **Geometric product** of a Gaussian blade with a precision multivector.
2. **Curvature extraction** from the resulting multivector (norm of the bivector part).
3. **Bayesian‑Bandit update** where the curvature‑scaled confidence bound drives the
   Upper‑Confidence‑Bound (UCB) decision rule.

The three public functions below demonstrate this fused workflow.

"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Core geometric algebra utilities (Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """
    Very small geometric algebra implementation for 2‑D space.
    Stored as a dict mapping frozenset of basis indices to a scalar coefficient.
    Grade‑0 (scalar) → frozenset()
    Grade‑1 (vector) → frozenset({i})
    Grade‑2 (bivector) → frozenset({i, j})
    """
    def __init__(self, components: Dict[frozenset, float] = None, n: int = 2):
        self.n = int(n)
        self.components: Dict[frozenset, float] = {}
        if components:
            for k, v in components.items():
                if abs(v) > 1e-15:
                    self.components[frozenset(k)] = float(v)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = Multivector(self.components.copy(), self.n)
        for blade, coeff in other.components.items():
            result.components[blade] = result.components.get(blade, 0.0) + coeff
            if abs(result.components[blade]) < 1e-15:
                del result.components[blade]
        return result

    def __mul__(self, scalar: float) -> "Multivector":
        """Scalar multiplication."""
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        terms = [f"{coeff:.3g}{''.join(str(i) for i in sorted(blade)) or '1'}"
                 for blade, coeff in self.components.items()]
        return " + ".join(terms) if terms else "0"

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Naïve geometric product for 2‑D multivectors.
    For this fusion we only need scalar‑scalar and scalar‑bivector products,
    which reduce to ordinary multiplication.
    """
    result = Multivector(n=a.n)
    for blade_a, coeff_a in a.components.items():
        for blade_b, coeff_b in b.components.items():
            # XOR of basis sets gives the resulting blade (up to sign, ignored here)
            new_blade = blade_a.symmetric_difference(blade_b)
            result.components[new_blade] = result.components.get(new_blade, 0.0) + coeff_a * coeff_b
    # Clean near‑zero entries
    result.components = {b: c for b, c in result.components.items() if abs(c) > 1e-15}
    return result

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Standard Gaussian probability density evaluated at `theta`."""
    denom = width * math.sqrt(2 * math.pi)
    exponent = -0.5 * ((theta - center) / width) ** 2
    return math.exp(exponent) / denom

def fisher_precision(samples: np.ndarray) -> np.ndarray:
    """
    Approximate Fisher information matrix as the inverse of the sample covariance.
    For 1‑D data this reduces to 1/var, for multi‑dimensional data we use the full
    covariance matrix.
    """
    if samples.ndim == 1:
        var = np.var(samples, ddof=1)
        return np.array([[1.0 / var]]) if var > 0 else np.array([[1e6]])
    cov = np.cov(samples, rowvar=False, ddof=1)
    # Guard against singular covariance
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Add a tiny diagonal jitter
        jitter = np.eye(cov.shape[0]) * 1e-6
        prec = np.linalg.inv(cov + jitter)
    return prec

# ----------------------------------------------------------------------
# Bandit & Bayesian utilities (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridGB"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# Global (in‑memory) stores used by the Bayesian update
_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}

def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Simple Bayesian update for a Gaussian prior N(mu, Sigma) with a Gaussian
    likelihood N(y, Lambda).  The posterior is also Gaussian with:

        Sigma_post = (Sigma^{-1} + Lambda^{-1})^{-1}
        mu_post    = Sigma_post (Sigma^{-1} mu + Lambda^{-1} y)

    For the fusion we treat `prior` as the mean vector and `likelihood`
    as the precision matrix (Fisher information) derived from data.
    """
    mu_prior = prior
    Sigma_prior = np.eye(len(mu_prior))  # unit covariance for simplicity
    Lambda = likelihood  # precision matrix

    Sigma_post_inv = np.linalg.inv(Sigma_prior) + Lambda
    Sigma_post = np.linalg.inv(Sigma_post_inv)
    mu_post = Sigma_post @ (np.linalg.inv(Sigma_prior) @ mu_prior + Lambda @ mu_prior)
    return mu_post  # we return the posterior mean as the updated belief

def curvature_from_multivector(mv: Multivector) -> float:
    """
    Extract a scalar curvature measure from a multivector.
    We use the Euclidean norm of all bivector components as a proxy for
    Ollivier‑Ricci curvature.
    """
    bivector_norm_sq = sum(coeff ** 2 for blade, coeff in mv.components.items()
                          if len(blade) == 2)
    return math.sqrt(bivector_norm_sq)

def select_action_ucb(actions: List[BanditAction], curvature: float) -> BanditAction:
    """
    Upper‑Confidence‑Bound (UCB) selection where the confidence term is
    inflated by `1 + curvature`.  This mirrors the curvature‑modulated bandit
    of Parent B.
    """
    best_score = -math.inf
    best_action = None
    for act in actions:
        # Modified confidence bound
        mod_conf = act.confidence_bound * (1.0 + curvature)
        score = act.expected_reward + mod_conf
        if score > best_score:
            best_score = score
            best_action = act
    return best_action

# ----------------------------------------------------------------------
# Fusion primitives – three public functions
# ----------------------------------------------------------------------
def hybrid_gaussian_multivector(theta: float,
                                center: float,
                                width: float,
                                prior_prob: float,
                                data_samples: np.ndarray) -> Multivector:
    """
    Create a Gaussian blade, embed Fisher precision as a bivector,
    and combine them via the geometric product.

    Returns a Multivector whose scalar part = Gaussian * prior_prob
    and whose bivector part encodes the Fisher precision (trace used as coefficient).
    """
    # 1. Gaussian scalar weight
    g_val = gaussian_beam(theta, center, width) * prior_prob
    scalar_mv = Multivector({frozenset(): g_val}, n=2)

    # 2. Fisher precision → bivector coefficient (use trace as a single scalar)
    precision = fisher_precision(data_samples)
    biv_coeff = float(np.trace(precision))
    biv_mv = Multivector({frozenset({0, 1}): biv_coeff}, n=2)

    # 3. Geometric product (scalar * bivector = bivector scaled)
    combined = geometric_product(scalar_mv, biv_mv)
    return combined

def curvature_modulated_bandit_step(context_features: Dict[str, float],
                                   actions: List[BanditAction],
                                   data_samples: np.ndarray) -> Tuple[BanditAction, float]:
    """
    Perform a single bandit decision step where:

    * The multivector built from the Gaussian‑Fisher fusion provides a curvature.
    * This curvature scales the confidence bound of each action.
    * The action with the highest curvature‑adjusted UCB is returned.
    """
    # Build a dummy Gaussian blade using an arbitrary feature as theta
    theta = context_features.get("operator_visceral_ratio", 0.5)
    mv = hybrid_gaussian_multivector(theta,
                                     center=0.5,
                                     width=0.1,
                                     prior_prob=1.0,
                                     data_samples=data_samples)

    curv = curvature_from_multivector(mv)

    chosen = select_action_ucb(actions, curvature=curv)
    return chosen, curv

def hybrid_tree_cost(edge_weights: List[float],
                    data_samples: np.ndarray,
                    curvature: float) -> float:
    """
    Minimum‑cost tree scoring blended with Fisher precision and curvature.

    * Base cost = sum of edge weights (as in Parent A).
    * Fisher precision contribution = trace(precision) (higher precision → lower cost).
    * Curvature factor inflates the cost proportionally to curvature.
    """
    base_cost = sum(edge_weights)

    precision = fisher_precision(data_samples)
    precision_factor = float(np.trace(precision))

    # Higher precision should reduce cost, so we invert it safely.
    precision_weight = 1.0 / (precision_factor + 1e-9)

    cost = (base_cost * (1.0 + curvature)) * precision_weight
    return cost

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data for Fisher information
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=0.0, scale=1.0, size=(100, 2))

    # Create a few dummy actions
    actions = [
        BanditAction(action_id="A", propensity=0.3, expected_reward=0.5, confidence_bound=0.2),
        BanditAction(action_id="B", propensity=0.6, expected_reward=0.4, confidence_bound=0.25),
        BanditAction(action_id="C", propensity=0.1, expected_reward=0.6, confidence_bound=0.15),
    ]

    # Random context features
    context = {
        "operator_visceral_ratio": rng.random(),
        "operator_tech_ratio": rng.random(),
    }

    # Run hybrid bandit step
    chosen_action, curv = curvature_modulated_bandit_step(context, actions, samples)
    print(f"Chosen action: {chosen_action.action_id} (curvature={curv:.4f})")

    # Compute hybrid tree cost
    edges = [1.2, 0.7, 2.5, 1.0]
    cost = hybrid_tree_cost(edges, samples, curvature=curv)
    print(f"Hybrid tree cost: {cost:.4f}")

    # Demonstrate the multivector output
    mv = hybrid_gaussian_multivector(theta=0.6,
                                     center=0.5,
                                     width=0.2,
                                     prior_prob=0.9,
                                     data_samples=samples)
    print(f"Hybrid multivector: {mv}")