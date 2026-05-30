# DARWIN HAMMER — match 5444, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1736_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1586_s0.py (gen5)
# born: 2026-05-30T00:01:53Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1736_s4.py
  (RBF surrogate, Gaussian beam Fisher information, SSIM consistency)
- Parent B: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1586_s0.py
  (Signature similarity, ternary decision weighting, expected‑value actions)

Mathematical Bridge:
The surrogate likelihood ℓ(x)=exp(-ε²‖x−c‖²) from Parent A supplies a
data‑driven likelihood term for a Bayesian update.  Fisher information
I(θ) derived from the Gaussian‑beam model weights the contribution of each
likelihood by the local information content.  SSIM(x, y) acts as a
tempering factor that down‑weights ℓ when the surrogate prediction deviates
from the observed evidence.

From Parent B the expected edge length 𝔼[ℓₑ] (computed from signature
similarities) is used to scale the posterior probabilities before they are
applied to the ternary‑decision actions (MathAction).  The final hybrid
decision value for an action a is

    V(a) = posterior_weight · 𝔼[ℓₑ] · a.expected_value

where posterior_weight = normalize( prior·ℓ·I·SSIM ).  This fuses the
probabilistic core of Parent A with the regret‑weighted ternary framework of
Parent B into a single unified system.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Parent A core utilities (adapted)
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
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


def ssim_1d(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
            k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified 1‑D Structural Similarity Index."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Parent B core utilities (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action with an intrinsic expected value."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(np.uint8(data), dtype=np.uint8).tobytes(), "big"
    )  # simple deterministic hash substitute


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two min‑hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def calculate_expected_value(action: MathAction, edge_length: float) -> float:
    """Scale the action's intrinsic value by an edge length factor."""
    return action.expected_value * edge_length


# ----------------------------------------------------------------------
# Hybrid Functions (the new unified system)
# ----------------------------------------------------------------------
def rbf_likelihood(sample: np.ndarray,
                   centers: np.ndarray,
                   epsilon: float = 1.0) -> np.ndarray:
    """
    Compute a vector of surrogate likelihoods ℓ_i = exp(-ε²‖sample−center_i‖²)
    for each centre in `centers`.
    """
    if centers.ndim != 2:
        raise ValueError("centers must be a 2‑D array (n_centers, dim)")
    distances = np.linalg.norm(centers - sample, axis=1)
    return np.vectorize(gaussian_rbf)(distances, epsilon)


def posterior_update(prior: np.ndarray,
                     likelihood: np.ndarray,
                     theta: float,
                     beam_center: float,
                     beam_width: float,
                     obs: np.ndarray,
                     pred: np.ndarray) -> np.ndarray:
    """
    Bayesian update that incorporates:
      * Fisher information I(θ) as a multiplicative weight,
      * SSIM between observation and surrogate prediction as a tempering factor.
    Returns a normalized posterior distribution.
    """
    # Fisher weighting
    fisher = fisher_score(theta, beam_center, beam_width)
    weighted_likelihood = likelihood * fisher

    # SSIM tempering (clipped to [0,1])
    ssim_val = ssim_1d(obs, pred)
    tempered = weighted_likelihood * max(0.0, min(1.0, ssim_val))

    unnorm_posterior = prior * tempered
    if unnorm_posterior.sum() == 0:
        # avoid division by zero – fall back to prior
        return prior / prior.sum()
    return unnorm_posterior / unnorm_posterior.sum()


def hybrid_action_values(actions: List[MathAction],
                         edge_lengths: List[float],
                         posterior: np.ndarray) -> Dict[str, float]:
    """
    Fuse the posterior probabilities (from Parent A) with the expected‑value
    actions (from Parent B).  The edge_lengths are derived from signature
    similarities and act as scaling factors.
    """
    if len(actions) != len(edge_lengths):
        raise ValueError("actions and edge_lengths must be of equal length")
    if posterior.shape[0] != len(actions):
        raise ValueError("posterior length must match number of actions")

    # Normalise posterior to act as weights for actions
    posterior_weights = posterior / posterior.sum()

    values = {}
    for i, (act, edge_len) in enumerate(zip(actions, edge_lengths)):
        exp_val = calculate_expected_value(act, edge_len)
        values[act.id] = posterior_weights[i] * exp_val
    return values


def expected_edge_length(tokens_a: List[str],
                         tokens_b: List[str],
                         k: int = 128) -> float:
    """
    Compute an expected edge length as the similarity between two token
    signatures.  The result lies in [0,1] and is used to scale action values.
    """
    sig_a = signature(tokens_a, k)
    sig_b = signature(tokens_b, k)
    return similarity(sig_a, sig_b)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for surrogate
    np.random.seed(0)
    sample = np.random.rand(5)
    centers = np.random.rand(3, 5)
    prior = np.array([0.3, 0.4, 0.3])

    # Compute likelihoods via RBF surrogate
    lik = rbf_likelihood(sample, centers, epsilon=0.8)

    # Observation vs prediction for SSIM
    obs = np.random.rand(100)
    pred = np.random.rand(100)

    # Bayesian posterior with Fisher and SSIM
    post = posterior_update(
        prior=prior,
        likelihood=lik,
        theta=0.5,
        beam_center=0.0,
        beam_width=1.0,
        obs=obs,
        pred=pred,
    )

    # Define actions and edge lengths (similarity based)
    actions = [
        MathAction(id="A", expected_value=10.0),
        MathAction(id="B", expected_value=20.0),
        MathAction(id="C", expected_value=15.0),
    ]
    edge_len = expected_edge_length(
        tokens_a=["alpha", "beta", "gamma"], tokens_b=["beta", "delta"]
    )
    edge_lengths = [edge_len] * len(actions)  # uniform for demo

    # Hybrid decision values
    values = hybrid_action_values(actions, edge_lengths, post)

    print("Posterior:", post)
    print("Edge similarity (expected length):", edge_len)
    print("Hybrid action values:")
    for aid, val in values.items():
        print(f"  {aid}: {val:.4f}")

    # Simple sanity check: values sum should be <= max expected value
    total = sum(values.values())
    assert total <= max(a.expected_value for a in actions) * edge_len + 1e-6
    print("Smoke test passed.")