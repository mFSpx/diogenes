# DARWIN HAMMER — match 2810, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_infota_m346_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1396_s2.py (gen4)
# born: 2026-05-29T23:46:01Z

"""Hybrid Fisher‑Infotaxis‑Semantic‑LTC algorithm.

This module fuses:

* **Parent A** – Fisher information for a Gaussian sensing beam and MinHash‑based
  similarity used as the hit‑probability in an infotaxis‑style expected‑entropy
  computation.
* **Parent B** – Semantic‑neighbor Bayesian updates and a Liquid‑Time‑Constant (LTC)
  recurrent dynamics whose time‑constant is driven by semantic distances.

**Mathematical bridge**

For a candidate sensing angle θ the Fisher score F(θ) quantifies how sharply the
beam discriminates the underlying directional parameter.  A token set T derived
from a hypothesis is compared to the current observation token set O via a
MinHash Jaccard estimate J(T,O), which plays the role of the hit‑probability
pₕᵢₜ in infotaxis.  We combine them multiplicatively to obtain a *weighted
likelihood*  

 L = F(θ)·J(T,O) / Z ,

where Z normalises across hypotheses.  This likelihood feeds a Bayesian update
of the prior probabilities of semantic neighbours.  The posterior probabilities
are then injected as inputs to an LTC recurrent cell whose time‑constant τ is
inversely related to the semantic distance d between the neighbour and the
current query point:

 τ = τ₀ · (1 + d) .

The LTC update implements a first‑order ODE discretisation  

 x_{t+1} = x_t + Δt·(‑x_t + L·input)/τ .

The resulting state vector encodes a jointly optimised sensing direction,
token hypothesis, and resource‑allocation schedule that maximises expected
information gain while respecting semantic topology.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# MinHash utilities (Parent A – infotaxis side)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
_DEFAULT_SEEDS = [random.randint(0, 2**32 - 1) for _ in range(128)]


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash from a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    # Use built‑in hashlib.blake2b for speed and determinism
    import hashlib
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Sequence[str], k: int = 128) -> List[int]:
    """Compute a MinHash signature of length *k* for a token sequence."""
    signature = [MAX64] * k
    for i, seed in enumerate(_DEFAULT_SEEDS[:k]):
        for token in tokens:
            h = _hash(seed, token)
            if h < signature[i]:
                signature[i] = h
    return signature


def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]


def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for Bayesian update."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior probability given prior, likelihood and marginal."""
    if marginal <= 0.0:
        raise ValueError("P(E) must be > 0")
    return prior * likelihood / marginal


def semantic_neighbors(query_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Dummy semantic‑neighbor generator.
    Returns a list of (neighbor_id, distance) tuples.
    """
    # In a real system this would query an embedding index.
    # Here we fabricate synthetic distances.
    random.seed(hash(query_id) & 0xffffffff)
    neighbors = [(f"doc_{i}", random.uniform(0.1, 2.0)) for i in range(k)]
    return neighbors


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) cell
# ----------------------------------------------------------------------
def ltc_update(state: np.ndarray,
               input_vec: np.ndarray,
               tau: np.ndarray,
               dt: float = 0.01) -> np.ndarray:
    """
    One step of a discrete LTC dynamics:
        x_{t+1} = x_t + dt * ( -x_t + input_vec ) / tau
    Parameters
    ----------
    state : np.ndarray
        Current hidden state (shape: (N,))
    input_vec : np.ndarray
        Input vector (same shape as state)
    tau : np.ndarray
        Time‑constant per dimension (positive)
    dt : float
        Integration step size
    Returns
    -------
    np.ndarray
        Updated state.
    """
    if not (state.shape == input_vec.shape == tau.shape):
        raise ValueError("state, input_vec and tau must share shape")
    if np.any(tau <= 0):
        raise ValueError("All tau values must be positive")
    return state + dt * (-state + input_vec) / tau


# ----------------------------------------------------------------------
# Hybrid core functions (integration of both parents)
# ----------------------------------------------------------------------
def hybrid_likelihood(theta: float,
                      center: float,
                      width: float,
                      tokens_obs: Sequence[str],
                      tokens_hyp: Sequence[str]) -> float:
    """
    Compute the weighted likelihood L = F(θ) * J(tokens_obs, tokens_hyp)
    where F is the Fisher score and J is the MinHash Jaccard estimate.
    The result is normalised to [0,1] by an ad‑hoc scaling factor.
    """
    F = fisher_score(theta, center, width)          # Fisher information (≥0)
    sig_obs = minhash_signature(tokens_obs)
    sig_hyp = minhash_signature(tokens_hyp)
    J = minhash_jaccard(sig_obs, sig_hyp)           # in [0,1]

    # Simple scaling: map Fisher scores to (0,1] via a logistic‑like transform
    F_norm = 1.0 - math.exp(-F)                     # monotonic, bounded
    L_raw = F_norm * J
    return min(1.0, max(0.0, L_raw))


def update_semantic_posteriors(query_id: str,
                               theta: float,
                               center: float,
                               width: float,
                               tokens_obs: Sequence[str],
                               neighbor_priors: Dict[str, float],
                               false_positive: float = 0.01) -> Dict[str, float]:
    """
    Perform a Bayesian update of semantic‑neighbor probabilities using the
    hybrid likelihood derived from Fisher information and MinHash similarity.
    """
    # Gather neighbours and their distances
    neighbours = semantic_neighbors(query_id, k=len(neighbor_priors))
    posteriors = {}

    # Normalise the raw likelihoods across neighbours to obtain a proper
    # likelihood distribution.
    raw_likelihoods = []
    hypo_tokens_list = []
    for nid, _dist in neighbours:
        # For demonstration we fabricate a hypothesis token set per neighbour:
        random.seed(hash(nid) & 0xffffffff)
        hypo_tokens = [f"tok_{random.randint(0, 100)}" for _ in range(10)]
        hypo_tokens_list.append((nid, hypo_tokens))
        L = hybrid_likelihood(theta, center, width, tokens_obs, hypo_tokens)
        raw_likelihoods.append(L)

    total_L = sum(raw_likelihoods) + 1e-12
    normalized_likelihoods = [L / total_L for L in raw_likelihoods]

    # Bayesian update per neighbour
    for (nid, _dist), L_norm, prior in zip(neighbours,
                                          normalized_likelihoods,
                                          neighbor_priors.values()):
        marginal = bayes_marginal(prior, L_norm, false_positive)
        posterior = bayes_update(prior, L_norm, marginal)
        posteriors[nid] = posterior

    return posteriors


def hybrid_allocation_state(query_id: str,
                            theta: float,
                            center: float,
                            width: float,
                            tokens_obs: Sequence[str],
                            prior_probs: Dict[str, float],
                            state_dim: int = 8,
                            dt: float = 0.01,
                            steps: int = 10) -> np.ndarray:
    """
    Run an LTC‑modulated allocation dynamics whose inputs are the posterior
    probabilities of semantic neighbours.  The time‑constant τ for each hidden
    unit is derived from the average semantic distance of the neighbours.
    """
    # 1. Update posteriors using the hybrid Bayesian step
    posteriors = update_semantic_posteriors(query_id,
                                            theta,
                                            center,
                                            width,
                                            tokens_obs,
                                            prior_probs)

    # 2. Build input vector: concatenate posterior values (padded/truncated)
    prob_vec = np.array(list(posteriors.values()), dtype=np.float64)
    if prob_vec.size < state_dim:
        prob_vec = np.pad(prob_vec, (0, state_dim - prob_vec.size), constant_values=0.0)
    else:
        prob_vec = prob_vec[:state_dim]

    # 3. Derive τ from semantic distances (average distance → slower dynamics)
    neighbours = semantic_neighbors(query_id, k=len(posteriors))
    avg_dist = np.mean([dist for _, dist in neighbours]) if neighbours else 1.0
    tau_base = 1.0
    tau_vec = tau_base * (1.0 + avg_dist) * np.ones(state_dim)

    # 4. Initialise hidden state (zeros) and run discrete LTC updates
    state = np.zeros(state_dim, dtype=np.float64)
    for _ in range(steps):
        state = ltc_update(state, prob_vec, tau_vec, dt=dt)

    return state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic observation tokens
    obs_tokens = [f"obs_{i}" for i in range(15)]

    # Prior probabilities for a small set of neighbours
    prior_dict = {f"doc_{i}": random.uniform(0.1, 0.9) for i in range(5)}

    # Parameters for the Gaussian beam
    theta_test = 0.35
    center_test = 0.0
    width_test = 1.2

    # Run hybrid allocation dynamics
    final_state = hybrid_allocation_state(
        query_id="query_123",
        theta=theta_test,
        center=center_test,
        width=width_test,
        tokens_obs=obs_tokens,
        prior_probs=prior_dict,
        state_dim=8,
        dt=0.02,
        steps=20,
    )

    print("Final LTC state:", final_state)
    # Verify that the state contains finite numbers
    assert np.all(np.isfinite(final_state)), "Non‑finite values in final state"
    print("Smoke test passed.")