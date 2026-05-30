# DARWIN HAMMER — match 1736, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m982_s0.py (gen5)
# parent_b: hybrid_hybrid_bayes_claim_k_hybrid_physarum_netw_m640_s0.py (gen4)
# born: 2026-05-29T23:38:33Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_fisher_m982_s0.py (RBF surrogate + Fisher + SSIM)
- hybrid_hybrid_bayes_claim_k_hybrid_physarum_netw_m640_s0.py (Bayesian update + pruning schedule)

Mathematical Bridge:
The surrogate model provides a data‑driven estimate of the likelihood ratio
ℓ(x)=p(evidence|hypothesis) that feeds the Bayesian update.  Fisher information
computed from the Gaussian‑beam model is used to adapt the pruning probability,
making the weight of each evidence a function of both time and local
information content.  SSIM is employed as a consistency check between the
surrogate prediction and the observed evidence, optionally tempering the
likelihood ratio when similarity is low.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    if a.shape != b.shape:
        raise ValueError("vectors must have same dimension")
    return float(np.linalg.norm(a - b))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile (used for Fisher information)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (1‑D)."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    sxy = np.mean((x - mx) * (y - my))
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
class MathEvidence:
    def __init__(self, id: str, signal: np.ndarray):
        self.id = id
        self.signal = signal  # raw observation vector


class MathHypothesis:
    def __init__(self, id: str, prior: float, posterior: float,
                 evidence_ids: Tuple[str, ...] = ()):
        self.id = id
        self.prior = prior
        self.posterior = posterior
        self.evidence_ids = evidence_ids


def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Base pruning schedule (exponential decay)."""
    if t < 0:
        raise ValueError("time t must be non‑negative")
    return math.exp(-lam * (t ** alpha))


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def rbf_kernel_matrix(X: np.ndarray, C: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute the Gaussian RBF kernel matrix K_{ij}=exp(-epsilon^2 * ||x_i-c_j||^2)
    between data points X (N×d) and centres C (M×d).
    """
    N = X.shape[0]
    M = C.shape[0]
    K = np.empty((N, M), dtype=float)
    for i in range(N):
        for j in range(M):
            r = euclidean(X[i], C[j])
            K[i, j] = gaussian(r, epsilon)
    return K


def surrogate_likelihood_ratio(
    evidence_signal: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """
    Use the RBF surrogate to map an evidence signal to a likelihood ratio ℓ.
    The surrogate is a linear combination of RBFs: ℓ = w·φ(x)
    where φ(x) is the kernel vector between x and the centres.
    """
    if evidence_signal.ndim != 1:
        raise ValueError("evidence_signal must be a 1‑D vector")
    phi = rbf_kernel_matrix(evidence_signal[None, :], centers, epsilon).ravel()
    ell = float(np.dot(weights, phi))
    # Clamp to a sensible positive range to keep Bayesian odds well‑behaved
    return max(1e-8, ell)


def hybrid_prune_probability(
    t: float,
    lam: float,
    alpha: float,
    theta: float,
    center: float,
    width: float,
    beta: float = 0.5,
) -> float:
    """
    Extend the base pruning schedule with Fisher information.
    High Fisher information (sharp features) reduces pruning, i.e.
    p_prune = base * (1 - beta * normalized_fisher).
    """
    base = prune_probability(t, lam, alpha)
    fisher = fisher_score(theta, center, width)
    # Normalise fisher to [0,1] using a simple logistic squash
    norm_fisher = 1.0 / (1.0 + math.exp(-fisher))
    return base * (1.0 - beta * norm_fisher)


def hybrid_update_hypothesis(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    centers: np.ndarray,
    weights: np.ndarray,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    epsilon: float = 1.0,
    theta: float = 0.0,
    center_fisher: float = 0.0,
    width_fisher: float = 1.0,
    ssim_thresh: float = 0.6,
) -> MathHypothesis:
    """
    Perform a Bayesian update where:
      * The likelihood ratio is supplied by the RBF surrogate.
      * The pruning probability is modulated by Fisher information.
      * If SSIM between surrogate prediction and the raw evidence falls
        below `ssim_thresh`, the likelihood ratio is attenuated to reflect
        low structural similarity.
    """
    # 1. Surrogate estimate of likelihood
    ell = surrogate_likelihood_ratio(evidence.signal, centers, weights, epsilon)

    # 2. SSIM consistency check
    surrogate_pred = rbf_kernel_matrix(evidence.signal[None, :], centers, epsilon).ravel() @ weights
    sim = ssim(evidence.signal, surrogate_pred * np.ones_like(evidence.signal))
    if sim < ssim_thresh:
        # damp the likelihood proportionally to the deficit
        ell *= sim / ssim_thresh

    # 3. Adjust pruning probability with Fisher information
    p_prune = hybrid_prune_probability(t, lam, alpha, theta, center_fisher, width_fisher)

    # 4. Effective likelihood after pruning (weights evidence)
    effective_ell = ell * (1.0 - p_prune)

    # 5. Standard Bayesian odds update
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or effective_ell == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * effective_ell
        posterior = new_odds / (1.0 + new_odds)

    posterior = max(0.0, min(1.0, posterior))

    # Record evidence id
    new_ids = hypothesis.evidence_ids + (evidence.id,)

    return MathHypothesis(
        id=hypothesis.id,
        prior=hypothesis.posterior,
        posterior=posterior,
        evidence_ids=new_ids,
    )


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def generate_random_rbf_model(d: int, m: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce random RBF centres (m×d) and weights (m) for testing.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(m, d))
    weights = rng.normal(loc=1.0, scale=0.5, size=m)  # bias towards positive values
    return centers, weights


def simulate_evidence(d: int, noise_level: float = 0.1, seed: int = 0) -> MathEvidence:
    """
    Generate a synthetic evidence vector with optional Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(loc=0.0, scale=1.0, size=d)
    signal += rng.normal(scale=noise_level, size=d)
    return MathEvidence(id=f"e{seed}", signal=signal)


def demo_hybrid_cycle():
    """
    Run a short chain of hybrid updates on a dummy hypothesis.
    """
    dim = 5
    centers, weights = generate_random_rbf_model(d=dim, m=8)

    # Initialise a hypothesis with a vague prior
    hyp = MathHypothesis(id="H0", prior=0.5, posterior=0.5)

    # Simulate a few evidences at increasing time steps
    for t in range(5):
        ev = simulate_evidence(dim, noise_level=0.2, seed=t)
        hyp = hybrid_update_hypothesis(
            hypothesis=hyp,
            evidence=ev,
            centers=centers,
            weights=weights,
            t=float(t),
            lam=0.8,
            alpha=0.3,
            epsilon=1.2,
            theta=ev.signal.mean(),
            center_fisher=0.0,
            width_fisher=1.0,
            ssim_thresh=0.5,
        )
        print(f"t={t}, posterior={hyp.posterior:.4f}, evidences={len(hyp.evidence_ids)}")


if __name__ == "__main__":
    demo_hybrid_cycle()