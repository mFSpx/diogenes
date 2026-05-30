# DARWIN HAMMER — match 5, survivor 1
# gen: 1
# parent_a: sketches.py (gen0)
# parent_b: rlct_grokking.py (gen0)
# born: 2026-05-29T23:15:04Z

"""Hybrid sketch‑RLCT module.

This module fuses the *sketch* primitives from ``sketches.py`` (Count‑Min,
HyperLogLog‑lite, MinHash LSH) with the *singular‑learning‑theory* utilities
from ``rlct_grokking.py`` (BIC, WAIC, RLCT estimation, free‑energy asymptotic).

Mathematical bridge
-------------------
Both families rely on **log‑count statistics**:

* Count‑Min sketch stores a linear combination of item frequencies
  ``c_{d,i} = Σ_j 1_{h_d(item_j)=i}``.  Summing the sketch rows yields an
  estimate of the total count Σ_j 1, i.e. the (log‑)likelihood contribution
  of a dataset.

* The RLCT formulas contain terms ``λ·log(n)`` and ``−(m−1)·loglog(n)`` that
  are functions of the *logarithm of a cardinality* (dataset size, number of
  distinct activation patterns, etc.).

The hybrid therefore uses a Count‑Min sketch to **approximate the empirical
log‑likelihood sum** required by WAIC/BIC, while the HyperLogLog estimate of
distinct tokens provides a cheap proxy for the effective number of activation
patterns that influences the RLCT λ.  The combined quantities feed the
free‑energy asymptotic and the RLCT regression.

The public API offers three representative hybrid operations:

1. ``build_hybrid_sketch`` – builds a Count‑Min sketch, a HyperLogLog cardinality,
   and a MinHash LSH index from a corpus.
2. ``approximate_log_likelihoods`` – uses the sketch to estimate per‑sample
   log‑likelihoods for WAIC‑style evaluation.
3. ``hybrid_rlct_estimate`` – derives an RLCT estimate from the sketch‑based
   loss curve and evaluates the asymptotic free energy.

All functions are pure Python and depend only on the allowed standard‑library
modules and NumPy.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives (from sketches.py)
# ----------------------------------------------------------------------


def count_min_sketch(
    items: Iterable[str], width: int = 64, depth: int = 4
) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(
                hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16
            ) % width
            table[d][idx] += 1
    return table


def hyperloglog_cardinality(items: Iterable[str]) -> int:
    """Very lightweight distinct‑count estimator (exact for small sets)."""
    return len(set(items))


def minhash_lsh_index(docs: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """MinHash‑based LSH bucket index."""
    buckets: defaultdict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        key = min(
            (
                hashlib.sha1(s.encode()).hexdigest()[:6]
                for s in shingles
            ),
            default="empty",
        )
        buckets[key].append(doc_id)
    return dict(buckets)


# ----------------------------------------------------------------------
# Singular‑learning‑theory utilities (from rlct_grokking.py)
# ----------------------------------------------------------------------


def bayesian_information_criterion(
    log_likelihood: float, n_params: int, n_samples: int
) -> float:
    """Standard BIC."""
    return -2.0 * float(log_likelihood) + float(n_params) * np.log(float(n_samples))


def waic_estimate(log_likelihoods_per_sample: np.ndarray) -> float:
    """Widely Applicable Information Criterion."""
    ll = np.asarray(log_likelihoods_per_sample, dtype=np.float64)  # (n, J)
    max_ll = ll.max(axis=1, keepdims=True)
    lppd = np.log(np.exp(ll - max_ll).mean(axis=1)) + max_ll.squeeze(axis=1)
    p_waic = ll.var(axis=1)
    return float(-2.0 * (lppd.sum() - p_waic.sum()))


def estimate_rlct_from_losses(
    train_losses_per_n: Iterable[float], n_values: Iterable[int]
) -> float:
    """Linear regression of log(loss) vs log(log(n))."""
    losses = np.asarray(list(train_losses_per_n), dtype=np.float64)
    ns = np.asarray(list(n_values), dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def free_energy_asymptotic(
    n: float, L0: float, lambda_rlct: float, m: int = 1
) -> float:
    """Asymptotic Bayesian free energy."""
    return n * L0 + lambda_rlct * math.log(n) - (m - 1) * math.log(math.log(n))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def build_hybrid_sketch(
    docs: Dict[str, Set[str]],
    width: int = 128,
    depth: int = 5,
) -> Tuple[List[List[int]], int, Dict[str, List[str]]]:
    """
    Construct the three sketch structures from a corpus.

    Returns
    -------
    cms : Count‑Min sketch of all tokens (flattened across documents)
    hll_est : HyperLogLog estimate of distinct tokens
    lsh   : MinHash LSH index mapping bucket keys to document IDs
    """
    # Flatten all tokens for the count‑min sketch and HLL
    all_tokens = (token for shingles in docs.values() for token in shingles)
    cms = count_min_sketch(all_tokens, width=width, depth=depth)

    # Re‑iterate tokens for distinct count (set conversion is cheap for test sizes)
    distinct_tokens = set()
    for shingles in docs.values():
        distinct_tokens.update(shingles)
    hll_est = hyperloglog_cardinality(distinct_tokens)

    # LSH index
    lsh = minhash_lsh_index(docs)

    return cms, hll_est, lsh


def _sketch_estimated_counts(
    cms: List[List[int]], width: int, depth: int
) -> np.ndarray:
    """
    Recover an approximate count vector from a Count‑Min sketch by taking the
    element‑wise minimum across rows (the standard estimator).
    """
    arr = np.array(cms, dtype=np.int64)  # shape (depth, width)
    return arr.min(axis=0)


def approximate_log_likelihoods(
    cms: List[List[int]],
    token_logprob: Callable[[str], float],
    token_list: List[str],
) -> np.ndarray:
    """
    Approximate per‑sample log‑likelihoods using the sketch.

    Parameters
    ----------
    cms : Count‑Min sketch built over the tokens.
    token_logprob : Callable mapping a token string to its log‑probability under
        the model (e.g. -log frequency, language model score, etc.).
    token_list : Ordered list of tokens that correspond to the sketch columns.
        The length must equal ``width`` used when building the sketch.

    Returns
    -------
    log_liks : ndarray of shape (width,) containing the estimated total
        log‑likelihood contributed by each column (i.e. bucket).
    """
    width = len(cms[0])
    depth = len(cms)
    # Estimate raw frequencies per column
    freq_est = _sketch_estimated_counts(cms, width, depth).astype(np.float64) + 1e-12
    # Convert frequencies to probabilities (normalised)
    prob_est = freq_est / freq_est.sum()
    # Apply the model's log‑probability function token‑wise
    log_probs = np.array([token_logprob(tok) for tok in token_list], dtype=np.float64)
    # Weighted sum gives an estimate of the total log‑likelihood per column
    return prob_est * log_probs


def hybrid_rlct_estimate(
    docs: Dict[str, Set[str]],
    token_logprob: Callable[[str], float],
    n_values: List[int],
    width: int = 128,
    depth: int = 5,
) -> Tuple[float, float]:
    """
    End‑to‑end RLCT estimation using sketch‑based loss approximations.

    Steps
    -----
    1. Build a Count‑Min sketch of the corpus.
    2. For each ``n`` in ``n_values`` take a random subset of ``n`` documents,
       recompute the sketch on that subset, and estimate the total negative log‑loss.
    3. Fit the RLCT regression on the (n, loss) pairs.
    4. Return the estimated ``lambda`` and the free‑energy value at the largest ``n``.

    Returns
    -------
    lambda_hat : float
        Estimated RLCT.
    free_energy : float
        Asymptotic free energy evaluated at the largest ``n`` using the estimated
        lambda and a naive ``L0`` taken as the average per‑sample loss.
    """
    # Helper to compute loss for a given subset of docs
    def subset_loss(subset_docs: Dict[str, Set[str]]) -> float:
        # Build sketch for the subset
        cms, _, _ = build_hybrid_sketch(subset_docs, width=width, depth=depth)
        # Build the token list that matches sketch columns (deterministic order)
        # We use the first ``width`` distinct tokens from the subset (fallback to padding)
        all_tokens = list(
            {tok for shingles in subset_docs.values() for tok in shingles}
        )
        token_list = all_tokens[:width] + ["<pad>"] * max(0, width - len(all_tokens))
        # Approximate log‑likelihoods per column
        col_ll = approximate_log_likelihoods(cms, token_logprob, token_list)
        # Total negative log‑likelihood (loss) is -sum(col_ll)
        return -float(col_ll.sum())

    losses: List[float] = []
    doc_items = list(docs.items())
    for n in n_values:
        if n > len(doc_items):
            raise ValueError(f"Requested n={n} exceeds corpus size {len(doc_items)}")
        subset = dict(random.sample(doc_items, n))
        losses.append(subset_loss(subset))

    lambda_hat = estimate_rlct_from_losses(losses, n_values)

    # Estimate L0 as average per‑sample loss at the largest n
    avg_loss = losses[-1] / n_values[-1]
    free_energy = free_energy_asymptotic(
        n=n_values[-1], L0=avg_loss, lambda_rlct=lambda_hat, m=1
    )
    return lambda_hat, free_energy


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Small synthetic corpus
    corpus = {
        "doc1": {"the", "quick", "brown", "fox"},
        "doc2": {"jumps", "over", "the", "lazy", "dog"},
        "doc3": {"lorem", "ipsum", "dolor", "sit", "amet"},
        "doc4": {"consectetur", "adipiscing", "elit"},
    }

    # Dummy log‑probability function: -log(1 + frequency of token in whole corpus)
    all_tokens = [t for s in corpus.values() for t in s]
    freq = defaultdict(int)
    for t in all_tokens:
        freq[t] += 1

    def token_logprob(tok: str) -> float:
        # Simple smoothed negative log‑frequency
        return -math.log(1 + freq.get(tok, 0))

    n_vals = [2, 3, 4]

    lam, fe = hybrid_rlct_estimate(
        docs=corpus,
        token_logprob=token_logprob,
        n_values=n_vals,
        width=64,
        depth=4,
    )
    print(f"Estimated RLCT (lambda): {lam:.4f}")
    print(f"Asymptotic free energy at n={n_vals[-1]}: {fe:.4f}")

    # Demonstrate that the sketch utilities still work independently
    cms, hll, lsh = build_hybrid_sketch(corpus, width=32, depth=3)
    print(f"Count‑Min sketch shape: {len(cms)}×{len(cms[0])}")
    print(f"HyperLogLog distinct token estimate: {hll}")
    print(f"MinHash LSH buckets (sample): {list(lsh.items())[:2]}")