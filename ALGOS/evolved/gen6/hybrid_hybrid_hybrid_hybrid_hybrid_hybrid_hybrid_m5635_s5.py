# DARWIN HAMMER — match 5635, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_krampus_brain_m617_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s2.py (gen5)
# born: 2026-05-30T00:03:40Z

"""
Hybrid Bayesian‑Krampus‑Fisher Algorithm
=======================================

Parents:
    - hybrid_hybrid_hybrid_bayes__hybrid_krampus_brain_m617_s0.py (Bayesian‑Curvature
      prior + linear bandit UCB)
    - hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1268_s2.py (Gaussian beam,
      Fisher‑information weighting, SSIM similarity, Hoeffding‑bound tree split)

Mathematical bridge
-------------------
Parent A regularises the Gram matrix **A** of a linear model with a scalar
*λ* derived from the Ollivier‑Ricci curvature *κ* of the feature set.
Parent B provides a *Fisher‑information* scalar *𝕀* for the same feature
distribution (computed via a Gaussian‑beam model).  

The hybrid therefore defines

    λ = κ · 𝕀

and uses **Â = A + λ·I** as the *curvature‑Fisher* regularised Gram matrix.
The Bayesian posterior is still

    posterior(action) ∝ prior(action)·likelihood(action)

where the prior is *κ* and the likelihood is the SSIM similarity between a
payload vector and a prototype.  The linear‑bandit UCB uses the regularised
inverse **Â⁻¹**.  Additionally, the Hoeffding bound decides whether a node in
a decision‑tree should split, with the bound scaled by the same λ.

The module implements three core hybrid functions that showcase this
integration.
"""

import sys
import random
import math
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# ----------------------------------------------------------------------
# Parent‑A utilities (features, curvature, SSIM, bandit)
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text) & 0xFFFFFFFFFFFFFFFF)
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "rainmaker_corporate_grit_tension", "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight", "telemetry_agent_symmetry"
    ]
    return {k: rnd.random() for k in keys}


def compute_ollivier_ricci_curvature(features: Dict[str, float]) -> float:
    """Simple proxy for Ollivier‑Ricci curvature: mean of feature values."""
    return float(np.mean(list(features.values())))


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size == 0:
        raise ValueError("empty signals")
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


def bandit_ucb(x: np.ndarray,
               A_inv: np.ndarray,
               theta: np.ndarray,
               alpha: float = 1.0) -> float:
    """Upper‑Confidence‑Bound for a linear bandit."""
    mean = float(theta @ x)
    var = float(x @ A_inv @ x)
    return mean + alpha * math.sqrt(var)


# ----------------------------------------------------------------------
# Parent‑B utilities (Gaussian beam, Fisher score, Hoeffding bound)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float,
                 center: float,
                 width: float,
                 eps: float = 1e-12) -> float:
    """Fisher‑information score for a scalar angle."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hoeffding_bound(mean: float,
                    range_width: float,
                    n: int,
                    delta: float) -> float:
    """Hoeffding bound ε such that P(|X‑μ|>ε) ≤ δ."""
    if n <= 0:
        raise ValueError("sample size must be positive")
    return range_width * math.sqrt(math.log(2.0 / delta) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_regularized_gram(features: Dict[str, float],
                            curvature: float,
                            fisher: float,
                            eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regularised Gram matrix Â = XᵀX + λ·I where λ = curvature·fisher.
    Returns the matrix and its inverse.
    """
    # Convert feature dict to column vector
    vec = np.array(list(features.values()), dtype=float).reshape(-1, 1)  # (d,1)
    A = vec @ vec.T                     # (d,d) rank‑1 Gram matrix
    d = A.shape[0]
    lam = curvature * fisher
    A_reg = A + lam * np.eye(d)
    # Numerical safeguard for inversion
    A_inv = np.linalg.inv(A_reg + eps * np.eye(d))
    return A_reg, A_inv


def hybrid_posterior_ucb(payload: np.ndarray,
                         prototype: np.ndarray,
                         features: Dict[str, float],
                         alpha: float = 1.0) -> float:
    """
    Compute a Bayesian‑regularised UCB score.
    - Prior: curvature κ from features.
    - Likelihood: SSIM(payload, prototype).
    - Linear model: θ = Â⁻¹ b where b = X·κ (simple proxy).
    """
    if payload.shape != prototype.shape:
        raise ValueError("payload and prototype must share shape")
    # 1️⃣ Curvature (prior)
    κ = compute_ollivier_ricci_curvature(features)

    # 2️⃣ Fisher information (used for regularisation)
    # Use arbitrary centre/width for demonstration
    θ_scalar = np.mean(payload)               # a stand‑in angle
    𝕀 = fisher_score(theta=θ_scalar, center=0.5, width=0.2)

    # 3️⃣ Regularised Gram matrix and its inverse
    _, A_inv = hybrid_regularized_gram(features, κ, 𝕀)

    # 4️⃣ Build b vector (simple linear proxy using κ)
    d = len(features)
    b = κ * np.ones(d)

    # 5️⃣ Parameter vector θ̂
    theta_hat = A_inv @ b

    # 6️⃣ SSIM as likelihood factor
    lik = ssim(payload, prototype)

    # 7️⃣ UCB on the feature space (project payload onto feature basis)
    x = np.array(list(features.values()))
    ucb = bandit_ucb(x, A_inv, theta_hat, alpha)

    # Combine Bayesian posterior (κ * lik) with exploration term
    return float(κ * lik + ucb)


def hybrid_tree_split(node_stats: Dict[str, float],
                      features: Dict[str, float],
                      delta: float = 0.05) -> bool:
    """
    Decide whether to split a decision‑tree node.
    The Hoeffding bound ε is scaled by λ = curvature·fisher.
    Split if the observed gain exceeds ε.
    """
    # Extract required statistics
    mean_gain = node_stats.get("mean_gain", 0.0)
    range_width = node_stats.get("range_width", 1.0)   # max‑min of gain
    n_samples = int(node_stats.get("n_samples", 1))

    # Curvature and Fisher for scaling
    κ = compute_ollivier_ricci_curvature(features)
    θ_scalar = np.mean(list(features.values()))
    𝕀 = fisher_score(theta=θ_scalar, center=0.5, width=0.2)

    lam = κ * 𝕀
    # Scaled Hoeffding bound
    eps = hoeffding_bound(mean_gain, range_width * lam, n_samples, delta)
    return mean_gain > eps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy textual payload and prototype
    txt = "The quick brown fox jumps over the lazy dog."
    features = extract_full_features(txt)

    # Create synthetic numeric payload / prototype
    rng = np.random.default_rng(42)
    payload = rng.integers(0, 256, size=64).astype(float)
    prototype = rng.integers(0, 256, size=64).astype(float)

    # Hybrid UCB score
    score = hybrid_posterior_ucb(payload, prototype, features, alpha=0.7)
    print(f"Hybrid posterior‑UCB score: {score:.4f}")

    # Node statistics for split decision
    node_stats = {
        "mean_gain": 0.12,
        "range_width": 0.5,
        "n_samples": 150
    }
    split = hybrid_tree_split(node_stats, features, delta=0.01)
    print(f"Tree node split decision: {'split' if split else 'do not split'}")