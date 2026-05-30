# DARWIN HAMMER — match 1248, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py (gen5)
# born: 2026-05-29T23:34:51Z

"""Hybrid NLMS‑Rectified‑Flow + Physarum‑RBF Algorithm
===================================================

Parent A: *hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s2.py*  
Parent B: *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m518_s3.py*

Mathematical bridge
-------------------
- The **conductance multivector** **C** of the Physarum network encodes the edge
  conductances *g* = (g₁,…,gₙ) in its grade‑1 part.
- A **rectified‑flow interpolant**  x(t)=t·gᵏ + (1‑t)·gᵏ⁻¹ produces a feature vector
  that is fed to an **NLMS predictor**  ŷ = w·x.
- The NLMS prediction is used as a **surrogate free‑energy** 𝔈̂(**C**)≈𝔈(**C**).  
  Because ŷ = w·x is linear in *g*, its gradient w.r.t. the conductances is simply the
  weight vector *w* (restricted to the first *n* components).
- The original Physarum update  

    gᵢ ← gᵢ + η ( Φᵢ – λ ∂𝔈/∂gᵢ )

  is therefore fused by replacing the exact energy gradient with the NLMS‑derived
  gradient *w*.  The NLMS weights are continuously adapted using the true
  RBF‑based energy 𝔈(g) as the target, closing the feedback loop.

The code below implements this unified system with three core functions:
`nlms_predict`, `nlms_update` and `hybrid_step`.  A small smoke test runs the
algorithm for a few iterations."""

import sys
import math
import random
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------------
# NLMS core (from Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction ŷ = w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output (here the true RBF energy).
    mu : float
        Learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (weights, error)
    """
    e = target - nlms_predict(weights, x)
    weights += mu * e * x / (np.linalg.norm(x) ** 2 + eps)
    return weights, e


# ----------------------------------------------------------------------
# Rectified flow interpolant (from Parent A)
# ----------------------------------------------------------------------
def interpolant(x0: np.ndarray, x1: np.ndarray, t: float) -> np.ndarray:
    """Straight‑line rectified‑flow interpolant:  x(t)=t·x1+(1‑t)·x0 ."""
    return t * x1 + (1.0 - t) * x0


# ----------------------------------------------------------------------
# Multivector (geometric algebra) – simplified version of Parent B
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector limited to scalar (grade‑0) and vector (grade‑1) parts.
    Grade‑1 blades are stored with keys frozenset({i}) → conductance g_i.
    """

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # keep only non‑zero entries
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "Multivector":
        """Create a multivector whose grade‑1 part equals `vec`."""
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec)}
        return cls(comps, n=len(vec))

    def vector_part(self) -> np.ndarray:
        """Return the grade‑1 part as a dense numpy vector of length n."""
        vec = np.zeros(self.n, dtype=float)
        for blade, val in self.components.items():
            if len(blade) == 1:
                i = next(iter(blade))
                vec[i] = val
        return vec

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, if any."""
        return self.components.get(frozenset(), 0.0)

    def copy(self) -> "Multivector":
        return Multivector(dict(self.components), self.n)

    def __repr__(self) -> str:
        return f"Multivector(components={self.components})"


# ----------------------------------------------------------------------
# RBF surrogate energy (from Parent B)
# ----------------------------------------------------------------------
def rbf_energy(g: np.ndarray, centers: np.ndarray, sigma: float) -> float:
    """
    Gaussian radial‑basis‑function surrogate for the free‑energy of the network.

    E(g) = Σ_j exp( -‖g‑c_j‖² / (2σ²) )
    """
    diffs = g - centers  # shape (m, n)
    sq_norms = np.sum(diffs ** 2, axis=1)
    return float(np.sum(np.exp(-sq_norms / (2.0 * sigma ** 2))))


# ----------------------------------------------------------------------
# Physarum flux placeholder (simplified)
# ----------------------------------------------------------------------
def compute_fluxes(g: np.ndarray) -> np.ndarray:
    """
    Very rough flux model: Φ_i = g_i * (random pressure difference).
    In a full Physarum simulation this would be obtained from solving a
    linear system of pressures; here we use a stochastic proxy.
    """
    pressures = np.random.randn(len(g) + 1)  # one pressure per node (n+1 nodes)
    # assume edge i connects node i and i+1
    diffs = pressures[1:] - pressures[:-1]
    return g * diffs


# ----------------------------------------------------------------------
# Hybrid step integrating both parents
# ----------------------------------------------------------------------
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
) -> Tuple[Multivector, np.ndarray, np.ndarray]:
    """
    Perform one iteration of the fused algorithm.

    Returns
    -------
    (C_new, g_new, weights_new)
    """
    # Current conductance vector
    g = C.vector_part()

    # 1. Feature generation via rectified‑flow interpolant
    x = interpolant(prev_g, g, t)          # shape (n,)

    # 2. NLMS prediction of the surrogate energy
    pred_energy = nlms_predict(weights, x)

    # 3. True surrogate energy from the RBF model (target for NLMS)
    true_energy = rbf_energy(g, centers, sigma)

    # 4. NLMS weight adaptation
    weights, _ = nlms_update(weights, x, true_energy, mu=mu)

    # 5. Gradient of the surrogate w.r.t conductances ≈ weights (linear model)
    grad_est = weights[: len(g)]  # only the first n components affect g

    # 6. Physarum fluxes (placeholder)
    phi = compute_fluxes(g)

    # 7. Hybrid conductance update
    g_new = g + eta * (phi - lam * grad_est)

    # 8. Build new multivector
    C_new = Multivector.from_vector(g_new)

    return C_new, g_new, weights


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo(num_steps: int = 5) -> None:
    random.seed(42)
    np.random.seed(42)

    n_edges = 6                     # modest size for the demo
    # Initial conductances (positive)
    g0 = np.abs(np.random.randn(n_edges))
    C = Multivector.from_vector(g0)
    prev_g = g0.copy()

    # NLMS weight vector (size n_edges)
    weights = np.zeros(n_edges, dtype=float)

    # RBF centers: a small set of random conductance patterns
    m_centers = 4
    centers = np.abs(np.random.randn(m_centers, n_edges))

    sigma = 1.0
    eta = 0.1      # Physarum learning rate
    lam = 0.05     # Coupling between NLMS gradient and Physarum
    mu = 0.3       # NLMS adaptation rate
    t = 0.5        # Mid‑point interpolant

    print("Initial conductances:", g0)
    for step in range(num_steps):
        C, g, weights = hybrid_step(
            C,
            prev_g,
            weights,
            centers,
            sigma,
            eta,
            lam,
            mu,
            t,
        )
        print(f"Step {step+1:02d} | g = {g.round(4)} | NLMS err norm = {np.linalg.norm(weights):.4f}")
        prev_g = g.copy()


if __name__ == "__main__":
    _demo()