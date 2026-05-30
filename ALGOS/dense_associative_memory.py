#!/usr/bin/env python3
"""
Dense Associative Memory — Modern Hopfield Networks.

Classical Hopfield networks store patterns as fixed points of a Hebbian weight
matrix.  Capacity is O(d) patterns in d dimensions, and retrieval is a linear
threshold update.

Modern Hopfield networks (Ramsauer et al. 2020) replace the binary threshold
with a softmax (Boltzmann distribution over stored patterns) and derive it from
an energy function:

    E(xi) = -beta^{-1} log( sum_i exp(beta * M_i . xi) )
            + 0.5 * xi . xi
            + lse_const

where M is the N x d memory matrix (N patterns, d dimensions) and beta is the
inverse temperature.

Properties:

  - As beta -> infinity: energy wells become exponentially deep and narrow.
    The softmax collapses onto the nearest pattern; one-step retrieval suffices.

  - Memory capacity: O(exp(d/2)) at high beta versus O(d) for classical nets.
    You can store exponentially many patterns in d dimensions.

  - Update rule: xi_new = M^T softmax(beta * M xi)
    This is exactly the Transformer attention mechanism.  Set beta = 1/sqrt(d),
    M = keys, xi = query: you get standard scaled dot-product attention.

The softmax IS the Boltzmann distribution over memory patterns at inverse
temperature beta.  Higher beta -> sharper retrieval -> lower temperature ->
more certain memory.  The Transformer was reverse-engineered from this energy.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "energy",
    "update_rule",
    "retrieve",
    "capacity_estimate",
    "attention_as_hopfield",
    "store_patterns",
]


def _softmax(z):
    """Numerically stable softmax over 1-D array z."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _lse(z):
    """log-sum-exp of 1-D array z (numerically stable)."""
    m = z.max()
    return m + np.log(np.exp(z - m).sum())


def energy(xi, M, beta=1.0):
    """Compute the Dense AM energy E(xi).

    Parameters
    ----------
    xi : array shape (d,)
        Query / current state vector.
    M : array shape (N, d)
        Memory matrix — each row is one stored pattern.
    beta : float
        Inverse temperature. Higher -> sharper energy wells.

    Returns
    -------
    float
        Scalar energy value. Fixed-point attractors are local minima.
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    # similarity scores: shape (N,)
    scores = beta * (M @ xi)
    lse_term = _lse(scores) / beta
    quadratic_term = 0.5 * np.dot(xi, xi)
    return -lse_term + quadratic_term


def update_rule(xi, M, beta=1.0):
    """One Dense AM retrieval step.

    xi_new = M^T softmax(beta * M xi)

    Each step descends the energy landscape.  At high beta the softmax peaks
    sharply on the closest memory pattern, so retrieval often converges in one
    or two steps.

    Parameters
    ----------
    xi : array shape (d,)
    M  : array shape (N, d)
    beta : float

    Returns
    -------
    xi_new : array shape (d,)
    """
    xi = np.asarray(xi, dtype=float)
    M = np.asarray(M, dtype=float)
    scores = beta * (M @ xi)          # shape (N,)
    weights = _softmax(scores)        # shape (N,)
    return M.T @ weights              # shape (d,)


def retrieve(xi, M, beta=1.0, steps=10, tol=1e-6):
    """Iterate update_rule until convergence or max steps.

    Parameters
    ----------
    xi    : array shape (d,)   — initial (possibly corrupted) query
    M     : array shape (N, d)
    beta  : float
    steps : int                — maximum iterations
    tol   : float              — L-inf norm change threshold for convergence

    Returns
    -------
    xi_final  : array shape (d,)
    n_steps   : int   — number of update steps taken
    converged : bool
    """
    xi = np.asarray(xi, dtype=float).copy()
    M = np.asarray(M, dtype=float)
    for step in range(1, steps + 1):
        xi_new = update_rule(xi, M, beta)
        delta = np.max(np.abs(xi_new - xi))
        xi = xi_new
        if delta < tol:
            return xi, step, True
    return xi, steps, False


def capacity_estimate(d, beta=1.0):
    """Approximate maximum storable patterns before retrieval degrades.

    Classical Hopfield: C ~ 0.138 * d  (Hopfield 1982 / Amit et al. 1985).
    Dense AM at high beta: C ~ exp(d / 2) by energy-landscape counting
    (Ramsauer et al. 2020, Theorem 1 and Corollary 2).

    At finite beta the effective capacity interpolates; this returns the
    high-beta exponential bound as the headline number.

    Parameters
    ----------
    d    : int   — pattern dimensionality
    beta : float — included for future finite-beta correction terms

    Returns
    -------
    float — estimated capacity
    """
    d = float(d)
    # Classical bound shown for reference (not returned but instructive).
    # classical = 0.138 * d
    return float(np.exp(d / 2.0))


def attention_as_hopfield(Q, K, V, beta=None):
    """Scaled dot-product attention IS Dense AM retrieval.

    For each query row q in Q, the output is:

        out = V^T softmax(beta * K q)

    with M = K (keys as memories) and reading out V instead of K.
    When V = K this is exactly update_rule.  The general V case is the
    standard value-projection used in Transformers.

    Set beta = 1/sqrt(d) (the canonical Transformer scaling) and this gives
    IDENTICAL output to the standard softmax attention formula:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

    Parameters
    ----------
    Q    : array shape (n_queries, d)
    K    : array shape (n_keys, d)
    V    : array shape (n_keys, d)
    beta : float or None — defaults to 1/sqrt(d)

    Returns
    -------
    out : array shape (n_queries, d)
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    d = Q.shape[1]
    if beta is None:
        beta = 1.0 / np.sqrt(d)
    # scores shape (n_queries, n_keys)
    scores = beta * (Q @ K.T)
    # softmax over keys axis for each query
    scores_stable = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores_stable)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ V


def store_patterns(patterns, noise_level=0.1, rng=None):
    """Add Gaussian noise to patterns to create corrupted queries.

    Parameters
    ----------
    patterns    : array shape (N, d)
    noise_level : float — std-dev of additive Gaussian noise
    rng         : numpy.random.Generator or None

    Returns
    -------
    corrupted : array shape (N, d)
    """
    patterns = np.asarray(patterns, dtype=float)
    if rng is None:
        rng = np.random.default_rng(seed=42)
    noise = rng.normal(0.0, noise_level, size=patterns.shape)
    return patterns + noise


if __name__ == "__main__":
    import sys

    rng = np.random.default_rng(seed=0)

    # ------------------------------------------------------------------ #
    # Part 1: retrieval from a corrupted query
    # ------------------------------------------------------------------ #
    N, d = 5, 16
    beta = 4.0  # high-ish temperature for sharp wells

    # Orthonormal-ish patterns: random unit vectors
    raw = rng.standard_normal((N, d))
    M = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    target_idx = 2
    xi0 = store_patterns(M, noise_level=0.4, rng=rng)[target_idx]

    print("=== Dense Associative Memory — retrieval demo ===")
    print(f"  Stored {N} patterns in {d}-dim space  (beta={beta})")
    print(f"  Target pattern index: {target_idx}")
    print(f"  Initial query cosine sim to target: "
          f"{np.dot(xi0, M[target_idx]):.4f}")

    # Show energy descending step by step
    xi = xi0.copy()
    print("\n  Step | Energy          | cos-sim to target")
    print("  " + "-" * 46)
    for step in range(8):
        e = energy(xi, M, beta)
        sim = np.dot(xi / (np.linalg.norm(xi) + 1e-12), M[target_idx])
        print(f"  {step:4d} | {e:15.6f} | {sim:.6f}")
        xi = update_rule(xi, M, beta)

    xi_final, n_steps, converged = retrieve(xi0, M, beta=beta, steps=50, tol=1e-8)
    retrieved_idx = int(np.argmax(M @ xi_final))
    print(f"\n  Converged: {converged}  after {n_steps} steps")
    print(f"  Retrieved pattern index: {retrieved_idx}  "
          f"(correct: {retrieved_idx == target_idx})")

    # ------------------------------------------------------------------ #
    # Part 2: capacity estimate
    # ------------------------------------------------------------------ #
    print("\n=== Capacity estimates ===")
    for dim in [8, 16, 32, 64]:
        cap_dense = capacity_estimate(dim)
        cap_classical = 0.138 * dim
        print(f"  d={dim:3d}  classical~{cap_classical:6.1f}  "
              f"dense~{cap_dense:.3e}")

    # ------------------------------------------------------------------ #
    # Part 3: attention == hopfield (the big reveal)
    # ------------------------------------------------------------------ #
    print("\n=== Attention as Hopfield ===")
    n_q, n_k, d_attn = 4, 6, 8

    Q = rng.standard_normal((n_q, d_attn))
    K = rng.standard_normal((n_k, d_attn))
    V = rng.standard_normal((n_k, d_attn))
    beta_attn = 1.0 / np.sqrt(d_attn)

    # Standard numpy softmax attention
    scores_std = (Q @ K.T) / np.sqrt(d_attn)
    scores_std -= scores_std.max(axis=1, keepdims=True)
    w_std = np.exp(scores_std)
    w_std /= w_std.sum(axis=1, keepdims=True)
    out_std = w_std @ V

    # Dense AM retrieval via attention_as_hopfield
    out_hop = attention_as_hopfield(Q, K, V, beta=beta_attn)

    max_diff = float(np.max(np.abs(out_std - out_hop)))
    print(f"  Max absolute difference (standard vs hopfield): {max_diff:.2e}")
    assert max_diff < 1e-12, f"mismatch: {max_diff}"
    print("  IDENTICAL output confirmed — softmax attention IS dense AM retrieval.")

    sys.exit(0)
