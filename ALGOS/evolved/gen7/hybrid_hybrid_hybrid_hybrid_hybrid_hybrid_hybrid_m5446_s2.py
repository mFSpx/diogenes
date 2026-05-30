# DARWIN HAMMER — match 5446, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1235_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m397_s1.py (gen4)
# born: 2026-05-30T00:01:54Z

"""Hybrid Algorithm: MinHash‑RBF‑Regret‑Hoeffding Fusion

This module combines the probabilistic Hoeffding‑bound machinery of
Parent A with the MinHash‑based similarity and radial‑basis‑function (RBF)
surrogate model of Parent B.

Mathematical bridge
-------------------
* The *observed gain* in the Hoeffding bound is replaced by a
  **tropical‑regret‑aware gain** derived from the MinHash Jaccard‑like
  similarity between two token sets.
* A set of reference MinHash signatures is fed to a Gaussian RBF kernel;
  the resulting kernel matrix is used as the **weight matrix** in the
  Hoeffding computation.
* The *regret* value from the regret‑based strategy is added to the
  observed gain before the bound is evaluated, yielding a
  **regret‑aware Hoeffding bound**.

The resulting hybrid can be used wherever a confidence bound on a
similarity‑driven gain is required while still accounting for past regret
and adaptive weighting via the RBF surrogate.

Author: computational‑physicist‑AI‑architect
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase."""
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))


def regret_aware_hoeffding_bound(
    regret: float,
    observed_gain: float,
    delta: float,
    n: int,
    weight_matrix: np.ndarray,
) -> float:
    """
    Regret‑aware Hoeffding bound.

    The classic Hoeffding term is computed on (observed_gain + regret) and
    the L2‑norm of the weight matrix is added as a regularisation term.
    """
    if n <= 0:
        raise ValueError('sample count n must be positive')
    if not (0.0 < delta < 1.0):
        raise ValueError('delta must be in (0,1)')
    effective_gain = observed_gain + regret
    bound = math.sqrt(
        (effective_gain * math.log(2.0 / delta)) / (2.0 * n)
    )
    return bound + np.linalg.norm(weight_matrix)


# ----------------------------------------------------------------------
# Parent B primitives
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Return a deterministic 64‑bit integer hash for a token + seed."""
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    """
    Compute a deterministic MinHash signature for a token list.
    """
    signature = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1  # max unsigned 64‑bit int
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity based on identical hash positions."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def rbf_kernel(vec1: np.ndarray, vec2: np.ndarray, sigma: float) -> float:
    """
    Gaussian radial‑basis kernel.
    k(x, y) = exp( -||x‑y||² / (2σ²) )
    """
    if sigma <= 0.0:
        raise ValueError('sigma must be positive')
    diff = vec1 - vec2
    sq_norm = np.dot(diff, diff)
    return math.exp(-sq_norm / (2.0 * sigma * sigma))


def build_weight_matrix(
    signature: List[int],
    reference_signatures: List[List[int]],
    sigma: float,
) -> np.ndarray:
    """
    Construct a weight matrix whose columns are RBF kernel evaluations
    between the given signature and each reference signature.
    """
    vec = np.array(signature, dtype=np.float64)
    cols = []
    for ref in reference_signatures:
        ref_vec = np.array(ref, dtype=np.float64)
        cols.append(rbf_kernel(vec, ref_vec, sigma))
    # The matrix is shaped (len(signature), n_refs) – we repeat the kernel
    # values across rows to obtain a full matrix compatible with the
    # Hoeffding norm term.
    if not cols:
        raise ValueError('reference_signatures must contain at least one signature')
    weight_matrix = np.tile(np.array(cols, dtype=np.float64), (len(signature), 1))
    return weight_matrix


# ----------------------------------------------------------------------
# Hybrid core functions (at least three)
# ----------------------------------------------------------------------
def hybrid_observed_gain(
    tokens_a: List[str],
    tokens_b: List[str],
    num_hash_functions: int,
    sigma: float,
    reference_signatures: List[List[int]],
) -> Tuple[float, np.ndarray]:
    """
    Compute the observed gain (MinHash similarity) and the associated
    weight matrix derived from the RBF surrogate model.
    """
    sig_a = minhash_signature(tokens_a, num_hash_functions)
    sig_b = minhash_signature(tokens_b, num_hash_functions)
    similarity = minhash_similarity(sig_a, sig_b)

    # Use the similarity as a gain proxy; scale to [0, 1] (already in that range)
    observed_gain = similarity

    # Build a weight matrix using *both* signatures as references.
    weight_matrix = build_weight_matrix(sig_a, reference_signatures, sigma)
    return observed_gain, weight_matrix


def hybrid_regret_aware_bound(
    regret: float,
    tokens_a: List[str],
    tokens_b: List[str],
    delta: float,
    n_samples: int,
    num_hash_functions: int,
    sigma: float,
    reference_signatures: List[List[int]],
) -> float:
    """
    End‑to‑end hybrid bound: combine regret, MinHash similarity, and
    RBF‑derived weight matrix inside the Hoeffding bound.
    """
    observed_gain, weight_matrix = hybrid_observed_gain(
        tokens_a,
        tokens_b,
        num_hash_functions,
        sigma,
        reference_signatures,
    )
    return regret_aware_hoeffding_bound(
        regret,
        observed_gain,
        delta,
        n_samples,
        weight_matrix,
    )


def hybrid_broadcast_decision(
    total_phases: int,
    current_phase: int,
    regret: float,
    tokens_a: List[str],
    tokens_b: List[str],
    delta: float,
    n_samples: int,
    num_hash_functions: int,
    sigma: float,
    reference_signatures: List[List[int]],
) -> bool:
    """
    Decide whether to broadcast in the current phase.

    The decision is stochastic: we broadcast if a uniform random draw
    falls below the product of the broadcast probability and a
    confidence factor derived from the hybrid Hoeffding bound.
    """
    prob = broadcast_probability(total_phases, current_phase)
    bound = hybrid_regret_aware_bound(
        regret,
        tokens_a,
        tokens_b,
        delta,
        n_samples,
        num_hash_functions,
        sigma,
        reference_signatures,
    )
    # Transform bound into a confidence factor in [0,1] (larger bound → smaller factor)
    confidence = max(0.0, 1.0 - bound)  # simple linear mapping
    decision_threshold = prob * confidence
    return random.random() < decision_threshold


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple reproducible test
    random.seed(42)

    tokens_set_1 = ["alpha", "beta", "gamma", "delta"]
    tokens_set_2 = ["beta", "epsilon", "zeta", "alpha"]

    # Generate a few reference signatures (could be from a training corpus)
    reference_sigs = [
        minhash_signature(tokens_set_1, 64),
        minhash_signature(tokens_set_2, 64),
        minhash_signature(["theta", "iota", "kappa"], 64),
    ]

    bound = hybrid_regret_aware_bound(
        regret=0.05,
        tokens_a=tokens_set_1,
        tokens_b=tokens_set_2,
        delta=0.01,
        n_samples=100,
        num_hash_functions=64,
        sigma=10.0,
        reference_signatures=reference_sigs,
    )
    print(f"Hybrid regret‑aware Hoeffding bound: {bound:.6f}")

    decision = hybrid_broadcast_decision(
        total_phases=10,
        current_phase=3,
        regret=0.05,
        tokens_a=tokens_set_1,
        tokens_b=tokens_set_2,
        delta=0.01,
        n_samples=100,
        num_hash_functions=64,
        sigma=10.0,
        reference_signatures=reference_sigs,
    )
    print(f"Broadcast decision (True=send, False=skip): {decision}")