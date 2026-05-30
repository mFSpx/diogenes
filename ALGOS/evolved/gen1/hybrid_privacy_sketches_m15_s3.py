# DARWIN HAMMER — match 15, survivor 3
# gen: 1
# parent_a: privacy.py (gen0)
# parent_b: sketches.py (gen0)
# born: 2026-05-29T23:25:05Z

"""Hybrid privacy‑sketch module.

This module fuses the core mathematics of *privacy.py* and *sketches.py*:

* From **privacy.py** we take the reconstruction‑risk ratio  
  `risk = unique_quasi_identifiers / total_records` and the DP‑aggregate
  concept of adding Laplace noise with scale `sensitivity/epsilon`.

* From **sketches.py** we take the Count‑Min sketch matrix `C[d, w]`,
  the per‑column minimum estimator `min_d C[d, w]` for frequency,
  and the MinHash‑LSH bucket construction.

The mathematical bridge is the *frequency matrix* `C`.  
We first populate `C` with hashed quasi‑identifier strings, then inject
Laplace noise (DP) directly into the matrix.  The noisy matrix yields a
noisy estimate of the number of distinct quasi‑identifiers via
`|{ w : min_d C_noisy[d, w] > 0 }|`.  Plugging this estimate into the
risk formula gives a differentially‑private reconstruction‑risk score.
The same noisy sketch can be reused for downstream approximate counting
or similarity (MinHash‑LSH) while preserving the privacy budget.

The module provides three high‑level hybrid functions demonstrating this
integration.
"""

import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

# ----------------------------------------------------------------------
# Core privacy utilities (adapted from privacy.py)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                             total_records: int) -> float:
    """Deterministic risk ratio clipped to [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def anonymize_record(record: Dict[str, Any],
                    redact_keys: Set[str] | None = None) -> Dict[str, Any]:
    """Redact sensitive keys; case‑insensitive match."""
    default_redact = {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    redact = {k.lower() for k in (redact_keys or default_redact)}
    return {
        k: ('<redacted>' if k.lower() in redact else v)
        for k, v in record.items()
    }


def dp_laplace_noise(scale: float) -> float:
    """Draw a single Laplace(0, scale) sample using numpy."""
    return float(np.random.laplace(0.0, scale))


# ----------------------------------------------------------------------
# Core sketch utilities (adapted from sketches.py)
# ----------------------------------------------------------------------
def _hash_item(item: str, seed: int) -> int:
    """Deterministic integer hash based on SHA‑256 and a seed."""
    h = hashlib.sha256(f'{seed}:{item}'.encode()).hexdigest()
    return int(h, 16)


def build_count_min_sketch(items: Iterable[str],
                           width: int = 64,
                           depth: int = 4) -> np.ndarray:
    """
    Populate a Count‑Min sketch matrix C ∈ ℕ^{depth×width}.
    Each cell counts occurrences of hashed items.
    """
    C = np.zeros((depth, width), dtype=np.int64)
    for item in items:
        for d in range(depth):
            col = _hash_item(item, d) % width
            C[d, col] += 1
    return C


def noisy_count_min_sketch(C: np.ndarray,
                           epsilon: float,
                           sensitivity: float = 1.0) -> np.ndarray:
    """
    Add independent Laplace noise to each cell of the sketch.
    Scale = sensitivity / epsilon.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale, size=C.shape)
    C_noisy = C.astype(np.float64) + noise
    # Clip to non‑negative values because counts cannot be negative
    np.maximum(C_noisy, 0.0, out=C_noisy)
    return C_noisy


def estimate_distinct_from_sketch(C: np.ndarray) -> int:
    """
    Classic Count‑Min distinct estimator:
    count columns whose minimum across rows is > 0.
    """
    min_across_rows = C.min(axis=0)
    return int(np.count_nonzero(min_across_rows > 0))


def minhash_lsh_index(docs: Dict[str, Set[str]],
                     num_perm: int = 128) -> Dict[str, List[str]]:
    """
    Very lightweight MinHash LSH: each document is assigned to a bucket
    keyed by the smallest hex prefix among its shingles.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        # Compute a deterministic permutation‑independent hash per shingle
        hashes = (
            hashlib.sha1(s.encode()).hexdigest()[:6]
            for s in shingles
        )
        bucket_key = min(hashes, default='empty')
        buckets[bucket_key].append(doc_id)
    return dict(buckets)


def dp_noise_bucket_sizes(buckets: Dict[str, List[str]],
                         epsilon: float,
                         sensitivity: float = 1.0) -> Dict[str, float]:
    """
    Apply Laplace noise to each bucket size, returning a mapping
    bucket_key → noisy size.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = sensitivity / epsilon
    noisy = {
        key: max(0.0, len(members) + np.random.laplace(0.0, scale))
        for key, members in buckets.items()
    }
    return noisy


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_privacy_risk(records: Iterable[Dict[str, Any]],
                        quasi_keys: Set[str],
                        epsilon: float,
                        width: int = 64,
                        depth: int = 4) -> float:
    """
    Compute a differentially‑private reconstruction risk score.

    Steps:
    1. Anonymize each record (redaction).
    2. Extract the concatenated quasi‑identifier string.
    3. Build a Count‑Min sketch over those strings.
    4. Add Laplace noise to the sketch (DP).
    5. Estimate the number of distinct quasi‑identifiers from the noisy sketch.
    6. Plug the estimate into the risk ratio.
    """
    # 1‑2. Prepare quasi‑identifier strings
    qi_strings = []
    for rec in records:
        anon = anonymize_record(rec)
        # Concatenate values of the selected quasi keys in a deterministic order
        parts = [str(anon.get(k, '')) for k in sorted(quasi_keys)]
        qi_strings.append('||'.join(parts))

    total = len(qi_strings)
    if total == 0:
        return 0.0

    # 3‑4. Sketch + DP noise
    C = build_count_min_sketch(qi_strings, width=width, depth=depth)
    C_noisy = noisy_count_min_sketch(C, epsilon=epsilon)

    # 5. Estimate distinct quasi‑identifiers
    distinct_est = estimate_distinct_from_sketch(C_noisy)

    # 6. Risk score
    return reconstruction_risk_score(distinct_est, total)


def hybrid_dp_cardinality(items: Iterable[str],
                          epsilon: float) -> float:
    """
    Differentially‑private cardinality estimate using a noisy Count‑Min sketch.
    Returns a float (noisy distinct count) rather than an integer.
    """
    C = build_count_min_sketch(items)
    C_noisy = noisy_count_min_sketch(C, epsilon=epsilon)
    # Use the same estimator as before, but treat the minimum as a real value
    min_across = C_noisy.min(axis=0)
    distinct_est = float(np.count_nonzero(min_across > 0))
    # Add an extra Laplace draw to the final count for a second layer of DP
    distinct_est += dp_laplace_noise(scale=1.0 / epsilon)
    return max(0.0, distinct_est)


def hybrid_lsh_with_privacy(docs: Dict[str, Set[str]],
                            epsilon: float) -> Dict[str, List[str]]:
    """
    Build MinHash LSH buckets, then publish only those buckets whose
    noisy size exceeds a threshold (e.g., 1).  This provides DP protection
    on bucket membership counts.
    """
    buckets = minhash_lsh_index(docs)
    noisy_sizes = dp_noise_bucket_sizes(buckets, epsilon=epsilon)

    # Keep buckets with noisy size >= 1 (i.e., likely non‑empty after noise)
    protected: Dict[str, List[str]] = {
        key: members
        for key, members in buckets.items()
        if noisy_sizes.get(key, 0.0) >= 1.0
    }
    return protected


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic dataset
    sample_records = [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "zip": "12345"},
        {"id": 2, "name": "Bob",   "email": "bob@example.com",   "zip": "12345"},
        {"id": 3, "name": "Carol", "email": "carol@example.com", "zip": "67890"},
    ]

    quasi = {"name", "zip"}
    eps = 1.0

    risk = hybrid_privacy_risk(sample_records, quasi, epsilon=eps)
    print(f"DP reconstruction risk: {risk:.4f}")

    items = [rec["zip"] for rec in sample_records]
    card = hybrid_dp_cardinality(items, epsilon=eps)
    print(f"DP cardinality estimate for ZIP codes: {card:.2f}")

    docs = {
        "doc1": {"apple", "banana", "cherry"},
        "doc2": {"banana", "date", "fig"},
        "doc3": {"apple", "fig", "grape"},
    }
    lsh = hybrid_lsh_with_privacy(docs, epsilon=eps)
    print(f"DP‑protected LSH buckets: {lsh}")