# DARWIN HAMMER — match 5396, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_minimum_cost__m693_s1.py (gen4)
# born: 2026-05-30T00:01:37Z

"""Hybrid Fisher-NLMS Entropic Edge Prior (HFNEE)

Parents:
- **Algorithm A** (`hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2535_s2.py`):
  Provides Shannon entropy of symbolic evidence, edge priors derived from entropy,
  a compatibility score via a matrix `P`, and an NLMS adaptive weight update.

- **Algorithm B** (`hybrid_hybrid_hybrid_fisher_hybrid_minimum_cost__m693_s1.py`):
  Supplies a Gaussian‑beam model and its Fisher information `F(θ)`.  
  The Fisher score quantifies how sharply the intensity varies with angle.

**Mathematical bridge**  
Both parents manipulate a scalar that measures *information*:  
- Entropy `H` (Algorithm A) quantifies uncertainty of discrete evidence.  
- Fisher information `F(θ)` (Algorithm B) quantifies the curvature of a continuous
  likelihood (the Gaussian beam).

In the hybrid we let the Fisher score modulate the *compatibility matrix* `P`
used by the NLMS update and also bias the edge priors.  Concretely


P(θ) = I + F(θ)·J


where `I` is the identity and `J` a simple rank‑1 matrix, so larger Fisher
information sharpens the interaction between weight vector `w` and the random
probe `m`.  Simultaneously each edge `(u,v)` receives a prior proportional to
`exp(-H·(1+0.1·i))·F(θ_uv)`, linking discrete entropy and continuous Fisher
information into a single unified prior distribution.

The resulting system adapts its weights with NLMS while respecting both
discrete evidence uncertainty and continuous measurement sensitivity.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def compute_shannon_entropy(evidence: List[str]) -> float:
    """Shannon entropy of a list of symbolic observations."""
    if not evidence:
        return 0.0
    counter = Counter(evidence)
    total = len(evidence)
    entropy = 0.0
    for cnt in counter.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def compatibility_score(v: np.ndarray, m: np.ndarray, P: np.ndarray) -> float:
    """Quadratic form vᵀ·P·m with truncation to the smallest common dimension."""
    dim = min(v.shape[0], m.shape[0], P.shape[0], P.shape[1])
    v2 = v[:dim]
    m2 = m[:dim]
    P2 = P[:dim, :dim]
    return float(v2.T @ P2 @ m2)


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    base_mu: float,
    eps: float,
    compat: float,
    entropy: float,
) -> np.ndarray:
    """Normalized LMS weight update; learning rate is attenuated by entropy."""
    if base_mu <= 0:
        raise ValueError("Base mu must be positive")
    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    mu_prime = base_mu * math.exp(-entropy)          # entropy‑driven decay
    y = float(w @ x)
    error = target - y
    power = float(x @ x) + eps
    increment = mu_prime * compat * error * x / power
    return w + increment


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I   where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    # derivative of Gaussian w.r.t θ
    dz = (theta - center) / (width * width)
    dI = -dz * intensity
    return (dI * dI) / intensity


def fisher_covariance_matrix(theta: float, center: float, width: float) -> np.ndarray:
    """
    Construct a 2×2 positive‑definite matrix using Fisher information.
    The matrix is I + F(θ)·J where J = [[1, 0.5],[0.5,1]].
    """
    F = fisher_score(theta, center, width)
    J = np.array([[1.0, 0.5], [0.5, 1.0]])
    return np.eye(2) + F * J


# ----------------------------------------------------------------------
# Hybrid primitives
# ----------------------------------------------------------------------
def hybrid_edge_priors(
    edges: List[Tuple[int, int]],
    evidence: List[str],
    theta_center: float,
    theta_width: float,
) -> Dict[Tuple[int, int], float]:
    """
    Combine discrete entropy with continuous Fisher information to produce
    edge priors.

    Prior(e) ∝ exp(-H·(1+0.1·i))·F(θ_e)
    where i is the edge index and θ_e = (u+v)/2 interpreted as an angle.
    """
    H = compute_shannon_entropy(evidence)
    if not edges:
        return {}
    raw = []
    for i, (u, v) in enumerate(edges):
        theta_e = (u + v) / 2.0
        F = fisher_score(theta_e, theta_center, theta_width)
        weight = math.exp(-H * (1 + 0.1 * i)) * (F + 1e-9)  # avoid zero
        raw.append(weight)
    total = sum(raw)
    return {e: w / total for e, w in zip(edges, raw)}


def hybrid_compatibility_matrix(
    w: np.ndarray,
    m: np.ndarray,
    theta_center: float,
    theta_width: float,
) -> np.ndarray:
    """
    Build the compatibility matrix `P` by blending a static identity with a
    Fisher‑driven covariance.  The angle used is the dot product direction
    between `w` and `m`.
    """
    # angle proxy: arccos of normalized dot product, clipped to avoid domain error
    dot = float(w @ m)
    norm = float(np.linalg.norm(w) * np.linalg.norm(m)) + 1e-12
    cos_angle = max(min(dot / norm, 1.0), -1.0)
    theta = math.acos(cos_angle)  # in radians
    return fisher_covariance_matrix(theta, theta_center, theta_width)


def hybrid_step(
    evidence: List[str],
    edges: List[Tuple[int, int]],
    x: np.ndarray,
    target: float,
    w: np.ndarray,
    theta_center: float = 0.0,
    theta_width: float = 1.0,
    base_mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
    """
    Perform one hybrid iteration:
    1. Compute entropy‑aware edge priors enriched by Fisher information.
    2. Build a Fisher‑modulated compatibility matrix.
    3. Sample a random probe `m` and evaluate the compatibility score.
    4. Update the weight vector `w` with NLMS using the entropy‑scaled learning rate.
    Returns the updated weight vector and the edge prior distribution.
    """
    # 1. priors
    priors = hybrid_edge_priors(edges, evidence, theta_center, theta_width)

    # 2. compatibility matrix
    m = np.random.rand(*w.shape)
    P = hybrid_compatibility_matrix(w, m, theta_center, theta_width)

    # 3. compatibility score
    s = compatibility_score(w, m, P)

    # 4. NLMS update (entropy already inside nlms_update)
    H = compute_shannon_entropy(evidence)
    w_new = nlms_update(w, x, target, base_mu, eps, s, H)

    return w_new, priors


def variational_free_energy(mu_q: np.ndarray, obs: np.ndarray, sigma_obs: float) -> float:
    """
    Simple Gaussian variational free energy:
        F = 0.5 * [ (μ_q - obs)ᵀ Σ⁻¹ (μ_q - obs) + log|Σ| + const ]
    With isotropic Σ = σ² I.
    """
    if sigma_obs <= 0:
        raise ValueError("sigma_obs must be positive")
    diff = mu_q - obs
    dim = diff.shape[0]
    term1 = np.dot(diff, diff) / (sigma_obs * sigma_obs)
    term2 = dim * math.log(sigma_obs * sigma_obs)
    return 0.5 * (term1 + term2)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic evidence
    evidence = ["alpha", "beta", "alpha", "gamma", "beta", "beta", "delta"]
    # simple graph edges
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # NLMS inputs
    w = np.array([0.2, -0.1])
    x = np.array([0.5, 0.4])
    target = 1.0

    # run hybrid step
    w_new, priors = hybrid_step(
        evidence,
        edges,
        x,
        target,
        w,
        theta_center=0.0,
        theta_width=1.0,
        base_mu=0.5,
        eps=1e-9,
    )

    print("Updated weights:", w_new)
    print("Edge priors:")
    for e, p in priors.items():
        print(f"  {e}: {p:.4f}")

    # demonstrate variational free energy using Fisher as sigma
    mu_q = w_new
    obs = np.array([0.6, 0.3])
    sigma = fisher_score(0.25, 0.0, 1.0) + 1e-6  # small positive variance
    F = variational_free_energy(mu_q, obs, sigma)
    print(f"Variational free energy (F): {F:.6f}")