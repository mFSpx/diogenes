# DARWIN HAMMER — match 98, survivor 2
# gen: 4
# parent_a: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# parent_b: hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s0.py (gen3)
# born: 2026-05-29T23:26:49Z

"""Hybrid Sketch‑Bayesian‑RLCT Module.

Parents:
- hybrid_sketches_rlct_grokking_m5_s1.py (sketch primitives + RLCT asymptotics)
- hybrid_hybrid_bayes_update__hybrid_hybrid_ternar_m150_s0.py (Bayesian update + SSIM‑guided bandit with Ollivier‑Ricci curvature)

Mathematical bridge:
Both parents operate on *log‑probability* quantities.  
The sketch suite provides an inexpensive estimator of the empirical log‑likelihood
∑_i log p(x_i|θ) via count‑min frequencies and a HyperLogLog estimate of the
effective number of distinct activation patterns.  Those quantities feed a
Gaussian conjugate Bayesian update (prior → posterior) where the likelihood
term is replaced by the sketch‑derived log‑likelihood.  The resulting posterior
covariance is then used as the “dimension m” in the RLCT asymptotic
formula λ·log n − (m−1)·loglog n.  Finally, MinHash signatures give a Jaccard‑based
similarity matrix that approximates Ollivier‑Ricci curvature; this curvature
modulates a multi‑armed bandit selection rule (SSIM‑inspired) for choosing
which sketch‑derived observations to incorporate next.

The module implements three core hybrid operations:
1. ``build_hybrid_sketch`` – builds Count‑Min, HyperLogLog, and MinHash structures.
2. ``bayesian_sketch_update`` – performs a conjugate Gaussian update using
   sketch‑derived log‑likelihoods and returns posterior parameters.
3. ``hybrid_rlct_estimate`` – computes an RLCT estimate from the posterior and
   sketch statistics, optionally using curvature‑aware bandit sampling. """

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives (adapted from parent A)
# ----------------------------------------------------------------------


def _hash(item: str, seed: int) -> int:
    """Deterministic integer hash for a given seed."""
    h = hashlib.blake2b(digest_size=8)
    h.update(item.encode("utf-8"))
    h.update(seed.to_bytes(2, "little"))
    return int.from_bytes(h.digest(), "little")


def count_min_sketch(
    items: Iterable[str], width: int = 128, depth: int = 5
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = _hash(item, d) % width
            table[d][idx] += 1
    return table


def hyperloglog_cardinality(items: Iterable[str], b: int = 10) -> int:
    """
    Very light HyperLogLog estimator.
    b is the number of bits for the register index (2**b registers).
    """
    m = 1 << b
    registers = [0] * m
    for item in items:
        x = _hash(item, 0)
        j = x & (m - 1)                # register index
        w = x >> b                     # remaining bits
        rank = (w.bit_length() - w.bit_length() + 1) if w != 0 else 1
        # count leading zeros + 1
        lz = (w.bit_length() - len(bin(w))) + 1 if w != 0 else b + 1
        rank = lz
        registers[j] = max(registers[j], rank)
    alpha_m = 0.7213 / (1 + 1.079 / m)
    Z = 1.0 / sum(2.0 ** -r for r in registers)
    E = alpha_m * m * m * Z
    # Small range correction
    if E <= 2.5 * m:
        V = registers.count(0)
        if V != 0:
            E = m * math.log(m / V)
    return int(E)


def minhash_signature(items: Iterable[str], num_perm: int = 64) -> np.ndarray:
    """Compute MinHash signature (Jaccard estimator) for a set of items."""
    sig = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for item in items:
        for i in range(num_perm):
            h = _hash(item, i)
            if h < sig[i]:
                sig[i] = h
    return sig


# ----------------------------------------------------------------------
# Bayesian update utilities (adapted from parent B)
# ----------------------------------------------------------------------


def gaussian_conjugate_update(
    prior_mu: np.ndarray,
    prior_sigma2: np.ndarray,
    observations: np.ndarray,
    obs_variance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a conjugate Gaussian update.
    prior_mu, prior_sigma2 : (d,) vectors (mean, variance per dimension)
    observations           : (n, d) matrix of observed values
    obs_variance           : scalar observation variance (assumed same for all dims)
    Returns posterior mean and variance.
    """
    precision_prior = 1.0 / prior_sigma2
    precision_lik = 1.0 / obs_variance
    # Aggregate sufficient statistics
    sum_obs = observations.sum(axis=0)
    n = observations.shape[0]
    posterior_precision = precision_prior + n * precision_lik
    posterior_sigma2 = 1.0 / posterior_precision
    posterior_mu = (precision_prior * prior_mu + precision_lik * sum_obs) / posterior_precision
    return posterior_mu, posterior_sigma2


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def build_hybrid_sketch(corpus: Iterable[str]) -> dict:
    """
    Build all sketch structures from a corpus.
    Returns a dict with keys:
        'cm'  : Count‑Min table
        'hll' : HyperLogLog cardinality estimate
        'mh'  : MinHash signature (numpy array)
    """
    items = list(corpus)
    cm = count_min_sketch(items)
    hll = hyperloglog_cardinality(items)
    mh = minhash_signature(items)
    return {"cm": cm, "hll": hll, "mh": mh, "total": len(items)}


def sketch_log_likelihood(
    cm: List[List[int]], query: str, width: int = 128, depth: int = 5
) -> float:
    """
    Approximate log‑likelihood of a single item using Count‑Min frequencies.
    Uses the minimum count across hash rows (standard CM estimator).
    """
    mins = []
    for d in range(depth):
        idx = _hash(query, d) % width
        mins.append(cm[d][idx])
    est_count = min(mins)
    # Laplace smoothing to avoid log(0)
    return math.log(est_count + 1)


def bayesian_sketch_update(
    prior_mu: np.ndarray,
    prior_sigma2: np.ndarray,
    sketch: dict,
    sample_items: Iterable[str],
    obs_variance: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use sketch‑derived log‑likelihoods of ``sample_items`` as observations
    to update a Gaussian prior.
    Returns posterior (mu, sigma2).
    """
    cm = sketch["cm"]
    width = len(cm[0])
    depth = len(cm)
    # Build observation matrix: each row is the vector of log‑likelihoods
    # for all dimensions (here we reuse the same scalar across dimensions for simplicity)
    obs = np.array(
        [sketch_log_likelihood(cm, itm, width, depth) for itm in sample_items],
        dtype=np.float64,
    )
    # Replicate scalar across dimensions to match prior shape
    obs_matrix = np.tile(obs[:, None], (1, prior_mu.shape[0]))
    posterior_mu, posterior_sigma2 = gaussian_conjugate_update(
        prior_mu, prior_sigma2, obs_matrix, obs_variance
    )
    return posterior_mu, posterior_sigma2


def ollivier_ricci_curvature(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature κ_{ij} = 1 - d_{ij} / diam,
    where d_{ij} = 1 - similarity (here Jaccard from MinHash).
    Returns a symmetric matrix of curvatures.
    """
    # Convert similarity to distance
    dist = 1.0 - sim_matrix
    diam = np.max(dist)
    # Avoid division by zero
    if diam == 0:
        return np.ones_like(dist)
    curvature = 1.0 - dist / diam
    return curvature


def minhash_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return np.mean(sig_a == sig_b)


def curvature_aware_bandit(
    sketches: List[dict],
    prior_mu: np.ndarray,
    prior_sigma2: np.ndarray,
    horizon: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a simple bandit loop: at each step select the sketch whose
    curvature‑weighted similarity to the current posterior mean is maximal,
    draw a small random sample from its corpus (simulated here), and perform
    a Bayesian sketch update.
    """
    # Pre‑compute pairwise similarities between MinHash signatures
    n = len(sketches)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = minhash_jaccard(sketches[i]["mh"], sketches[j]["mh"])
            sim[i, j] = sim[j, i] = s

    curvature = ollivier_ricci_curvature(sim)

    mu, sigma2 = prior_mu.copy(), prior_sigma2.copy()
    for t in range(horizon):
        # Score each sketch: curvature·(inverse variance)·similarity to mu
        scores = []
        for idx, sk in enumerate(sketches):
            # Use the HyperLogLog cardinality as a proxy for the amount of information
            info_weight = math.log(sk["hll"] + 1)
            score = curvature[idx].sum() * info_weight / (sigma2.mean() + 1e-9)
            scores.append(score)
        chosen = int(np.argmax(scores))
        # Simulate a tiny sample (5 random items) from the chosen sketch's corpus size
        sample = [f"item_{random.randint(0, sketches[chosen]['total'])}" for _ in range(5)]
        mu, sigma2 = bayesian_sketch_update(mu, sigma2, sketches[chosen], sample)
    return mu, sigma2


def hybrid_rlct_estimate(
    posterior_mu: np.ndarray,
    posterior_sigma2: np.ndarray,
    sketch: dict,
    n_samples: int = 1000,
) -> float:
    """
    Compute an RLCT estimate using the hybrid quantities.
    λ is approximated by the effective dimensionality: d_eff = trace(cov) / max variance.
    n is the total count from the Count‑Min sketch (sum of minima for a dummy item).
    m is the number of posterior dimensions (len(posterior_mu)).
    RLCT ≈ λ·log n − (m−1)·loglog n .
    """
    # Estimate total count n via a dummy query that hits every bucket
    width = len(sketch["cm"][0])
    depth = len(sketch["cm"])
    dummy = "___DUMMY___"
    n_est = math.exp(sketch_log_likelihood(sketch["cm"], dummy, width, depth)) - 1
    n_est = max(n_est, 1.0)

    cov = np.diag(posterior_sigma2)
    max_var = np.max(posterior_sigma2)
    lambda_eff = np.trace(cov) / (max_var + 1e-12)

    m = posterior_mu.shape[0]
    rlct = lambda_eff * math.log(n_est) - (m - 1) * math.log(math.log(n_est) + 1e-12)
    return rlct


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic corpora
    corpora = [
        [f"doc_{i}_topicA" for i in range(500)],
        [f"doc_{i}_topicB" for i in range(300)],
        [f"doc_{i}_topicC" for i in range(800)],
    ]

    # Build sketches for each corpus
    sketches = [build_hybrid_sketch(c) for c in corpora]

    # Prior over a 5‑dimensional latent vector (mirroring PROTOTYPE_VECTOR size)
    prior_mu = np.zeros(5)
    prior_sigma2 = np.full(5, 1.0)

    # Run curvature‑aware bandit to obtain posterior
    post_mu, post_sigma2 = curvature_aware_bandit(sketches, prior_mu, prior_sigma2, horizon=4)

    # Estimate RLCT from the resulting posterior and the first sketch
    rlct_value = hybrid_rlct_estimate(post_mu, post_sigma2, sketches[0])

    print("Posterior mean :", post_mu)
    print("Posterior var  :", post_sigma2)
    print("Hybrid RLCT    :", rlct_value)