# DARWIN HAMMER — match 5323, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hoeffding_tre_m44_s0.py (gen3)
# born: 2026-05-30T00:01:13Z

"""Hybrid Endpoint‑Memory‑Similarity Module

Parents:
    • hybrid_hybrid_hybrid_caputo_serpentina_self_righ_m444_s0.py (Caputo fractional
      derivative + MinHash similarity on Voronoi points)
    • hybrid_hybrid_endpoint_circ_state_space_duality_m1_s2.py (Health‑score SSM
      + Hoeffding‑bound decision)

Mathematical bridge:
    – The Caputo fractional derivative provides a power‑law memory kernel that can be
      applied to the time‑evolution of endpoint health scores.
    – MinHash yields an unbiased estimator of the Jaccard similarity between the
      tokenised feature sets of endpoints; this similarity can be interpreted as a
      geometric distance in the same Clifford‑algebra space used by the Caputo kernel.
    – The Hoeffding bound supplies a statistical guarantee for switching decisions.
      By feeding the memory‑aware (fractional) health derivatives together with the
      similarity‑weighted scores into the Hoeffding bound we obtain a single unified
      decision rule that respects both long‑range memory and probabilistic confidence.

The module therefore:
    1. Builds MinHash signatures for each endpoint.
    2. Computes a Caputo fractional derivative of the health‑score time series.
    3. Uses similarity‑weighted health scores together with a Hoeffding bound to
       decide whether to switch to a better endpoint.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Gamma(z) for z > 0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    g = 7
    z += g + 0.5
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    a = p[0]
    for i in range(1, len(p)):
        a += p[i] / (z - i)
    t = z + 0.5
    return math.sqrt(2 * math.pi) * t ** (z - 0.5) * math.exp(-t) * a


def caputo_derivative(f: np.ndarray, t: np.ndarray, alpha: float) -> float:
    """
    Approximate the Caputo fractional derivative of order ``alpha`` at the last time point.

    Parameters
    ----------
    f : np.ndarray
        Function values sampled at times ``t`` (1‑D array).
    t : np.ndarray
        Monotonically increasing time samples (1‑D array).
    alpha : float
        Fractional order, 0 < alpha < 1.

    Returns
    -------
    float
        Approximation of D^alpha f(t[-1]).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if f.shape != t.shape:
        raise ValueError("f and t must have the same shape")
    dt = np.diff(t)
    df = np.diff(f)
    # Kernel (t_n - s)^{-alpha} discretised by left‑point rule
    kernel = (t[-1] - t[:-1]) ** (-alpha)
    integral = np.sum(df * kernel) / lanczos_gamma(1 - alpha)
    return integral


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash combining a seed and a token."""
    data = f"{seed}{token}"
    h = 0
    for ch in data:
        h = (h * 31 + ord(ch)) & MAX64
    return h


def minhash_signature(tokens: Iterable[str], k: int, seed: int = 0) -> List[int]:
    """
    Compute a MinHash signature of length ``k`` for a set of string tokens.

    Parameters
    ----------
    tokens : iterable of str
        The token set representing an endpoint.
    k : int
        Signature length (number of hash permutations).
    seed : int, optional
        Base seed for hash functions.

    Returns
    -------
    list[int]
        List of k hash minima.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    mins = [MAX64] * k
    for i in range(k):
        perm_seed = seed + i * 0x9e3779b97f4a7c15  # large odd constant for mixing
        for token in tokens:
            h = _hash(perm_seed, token)
            if h < mins[i]:
                mins[i] = h
    return mins


def jaccard_estimate(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have equal length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
@dataclass
class Endpoint:
    """State of a service endpoint."""
    id: Any
    health_score: float
    failure_rate: float
    recovery_priority: float
    token_set: List[str]  # feature tokens for MinHash


def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """
    Hoeffding bound for a bounded random variable in [0, r].

    Parameters
    ----------
    r : float
        Range of the variable (max - min), must be > 0.
    delta : float
        Desired failure probability (0 < delta < 1).
    n : int
        Number of independent observations, must be > 0.

    Returns
    -------
    float
        Upper confidence bound.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("n must be positive")
    return r * math.sqrt(math.log(2 / delta) / (2 * n))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def compute_endpoint_signatures(endpoints: List[Endpoint], k: int = 64, seed: int = 0) -> Dict[Any, List[int]]:
    """
    Build MinHash signatures for all endpoints.

    Returns
    -------
    dict mapping endpoint.id -> signature list
    """
    signatures = {}
    for ep in endpoints:
        signatures[ep.id] = minhash_signature(ep.token_set, k=k, seed=seed)
    return signatures


def fractional_health_derivatives(
    times: np.ndarray,
    health_series: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Compute the Caputo fractional derivative of each endpoint's health series.

    Parameters
    ----------
    times : np.ndarray, shape (T,)
        Monotonically increasing timestamps.
    health_series : np.ndarray, shape (E, T)
        Health scores for E endpoints over T time steps.
    alpha : float, optional
        Fractional order (default 0.5).

    Returns
    -------
    np.ndarray, shape (E,)
        Approximate derivative at the last timestamp for each endpoint.
    """
    derivatives = np.empty(health_series.shape[0], dtype=float)
    for i, f in enumerate(health_series):
        derivatives[i] = caputo_derivative(f, times, alpha)
    return derivatives


def maybe_switch_endpoint(
    current: Endpoint,
    candidates: List[Endpoint],
    times: np.ndarray,
    health_series: np.ndarray,
    alpha: float = 0.5,
    delta: float = 0.05,
    k: int = 64,
) -> Endpoint:
    """
    Decide whether to switch from ``current`` to a better candidate.

    The decision uses:
        * Fractional derivative of health scores (memory‑aware trend).
        * MinHash‑based similarity weighting between endpoints.
        * Hoeffding bound to guarantee statistical confidence.

    Returns the chosen endpoint (may be ``current`` if no switch is warranted).
    """
    # 1. Derivative (trend) for all endpoints, ordered as [current] + candidates
    all_eps = [current] + candidates
    health_matrix = np.vstack([ep.health_score for ep in all_eps])  # shape (E, 1)
    # Expand to match the provided time series shape (E, T)
    # Assume ``health_series`` rows correspond to the same ordering.
    derivs = fractional_health_derivatives(times, health_series, alpha)

    # 2. Similarity weighting via MinHash
    signatures = compute_endpoint_signatures(all_eps, k=k)
    base_sig = signatures[current.id]
    weighted_scores = []
    for ep in all_eps:
        sim = jaccard_estimate(base_sig, signatures[ep.id])
        # Weight health by (1 + similarity) to favour similar (geometrically close) endpoints
        weighted = ep.health_score * (1.0 + sim) + derivs[all_eps.index(ep)]
        weighted_scores.append(weighted)

    # 3. Hoeffding bound to test if any candidate statistically exceeds current
    n_obs = len(times)
    bound = hoeffding_bound(r=1.0, delta=delta, n=n_obs)

    current_score = weighted_scores[0]
    best_candidate = current
    best_score = current_score

    for idx, ep in enumerate(candidates, start=1):
        if weighted_scores[idx] - bound > best_score:
            best_candidate = ep
            best_score = weighted_scores[idx]

    return best_candidate


def hybrid_compute_health_scores(endpoints: List[Endpoint]) -> List[float]:
    """
    Example helper that combines raw health_score with a recovery priority term
    and returns a normalized list.
    """
    raw = np.array([ep.health_score + 0.5 * ep.recovery_priority for ep in endpoints])
    if raw.max() == raw.min():
        return [1.0] * len(endpoints)
    normalized = (raw - raw.min()) / (raw.max() - raw.min())
    return normalized.tolist()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic endpoints
    eps = [
        Endpoint(id="A", health_score=0.7, failure_rate=0.02, recovery_priority=0.3,
                 token_set=["cpu", "linux", "v1"]),
        Endpoint(id="B", health_score=0.6, failure_rate=0.01, recovery_priority=0.5,
                 token_set=["cpu", "windows", "v2"]),
        Endpoint(id="C", health_score=0.8, failure_rate=0.03, recovery_priority=0.2,
                 token_set=["gpu", "linux", "v1"]),
    ]

    # Simulated time series (5 timestamps)
    times = np.arange(5, dtype=float)
    # For simplicity, repeat each endpoint's health_score over time with small noise
    health_series = np.array([
        eps[0].health_score + 0.01 * np.random.randn(5),
        eps[1].health_score + 0.01 * np.random.randn(5),
        eps[2].health_score + 0.01 * np.random.randn(5),
    ])

    # Choose current endpoint and candidates
    current = eps[0]
    candidates = eps[1:]

    # Run hybrid decision
    chosen = maybe_switch_endpoint(
        current=current,
        candidates=candidates,
        times=times,
        health_series=health_series,
        alpha=0.6,
        delta=0.05,
        k=64,
    )
    print(f"Current endpoint: {current.id}")
    print(f"Chosen endpoint after hybrid decision: {chosen.id}")
    # Also demonstrate health score normalization
    print("Normalized hybrid health scores:", hybrid_compute_health_scores(eps))