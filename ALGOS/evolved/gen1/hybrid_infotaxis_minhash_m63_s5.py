# DARWIN HAMMER — match 63, survivor 5
# gen: 1
# parent_a: infotaxis.py (gen0)
# parent_b: minhash.py (gen0)
# born: 2026-05-29T23:24:20Z

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

MAX64 = (1 << 64) - 1
DEFAULT_K = 128
_EPS = 1e-12


# ----------------------------------------------------------------------
# MinHash core
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash for a (seed, token) pair."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = DEFAULT_K) -> List[int]:
    """
    Compute a MinHash signature of length *k* for the given token collection.

    An empty token set yields a signature consisting of the maximal hash value,
    which represents the absence of any information.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    token_set: Set[str] = {t for t in tokens if t}
    if not token_set:
        return [MAX64] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """
    Approximate Jaccard similarity via the fraction of equal components
    in two MinHash signatures.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Entropy utilities
# ----------------------------------------------------------------------
def _normalize(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total <= 0:
        raise ValueError("probability mass must be positive")
    return [p / total for p in probs]


def entropy_from_counts(counts: List[int]) -> float:
    """
    Shannon entropy of a discrete distribution defined by raw counts.
    The counts are normalised internally.
    """
    if not counts:
        raise ValueError("counts must not be empty")
    probs = _normalize([float(c) for c in counts])
    return -sum(p * math.log(max(p, _EPS)) for p in probs)


def signature_entropy(sig: List[int]) -> float:
    """
    Treat a MinHash signature as a multiset of bucket identifiers and compute
    the Shannon entropy of the induced empirical distribution.
    """
    if not sig:
        raise ValueError("signature must not be empty")
    bucket_counts = list(Counter(sig).values())
    return entropy_from_counts(bucket_counts)


def expected_signature_entropy(p_hit: float, h_entropy: float, m_entropy: float) -> float:
    """
    Expected entropy after a probabilistic observation.

    *p_hit* – probability that the addition is observed (0 ≤ p_hit ≤ 1)  
    *h_entropy* – entropy of the signature after the addition (hit)  
    *m_entropy* – entropy of the signature if the addition is missed (miss)
    """
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * h_entropy + (1.0 - p_hit) * m_entropy


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def _probability_of_hit(sig_cur: List[int], sig_hit: List[int]) -> float:
    """
    Use MinHash similarity as a proxy for the probability that a candidate token
    will be observed (i.e., that the underlying set indeed contains it).
    """
    return similarity(sig_cur, sig_hit)


def hybrid_expected_entropy_for_addition(
    current_tokens: Iterable[str],
    token: str,
    k: int = DEFAULT_K,
) -> float:
    """
    Compute the expected entropy of the MinHash signature after *token* is
    (probabilistically) added to *current_tokens*.
    """
    cur_set = set(current_tokens)
    sig_cur = signature(cur_set, k=k)

    # Hit state: token is added
    hit_set = cur_set | {token}
    sig_hit = signature(hit_set, k=k)

    # Miss state: token is not added (signature unchanged)
    sig_miss = sig_cur

    p_hit = _probability_of_hit(sig_cur, sig_hit)

    h_ent = signature_entropy(sig_hit)
    m_ent = signature_entropy(sig_miss)

    return expected_signature_entropy(p_hit, h_ent, m_ent)


def hybrid_best_addition(
    current_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    k: int = DEFAULT_K,
) -> str:
    """
    Select the token from *candidate_tokens* that yields the smallest expected
    signature entropy after a potential addition.
    """
    cur_set = set(current_tokens)
    sig_cur = signature(cur_set, k=k)

    best_token: str | None = None
    best_exp_entropy = math.inf

    for token in candidate_tokens:
        # Compute hit signature once per candidate
        hit_set = cur_set | {token}
        sig_hit = signature(hit_set, k=k)

        p_hit = _probability_of_hit(sig_cur, sig_hit)

        h_ent = signature_entropy(sig_hit)
        m_ent = signature_entropy(sig_cur)  # miss state = current

        exp_ent = expected_signature_entropy(p_hit, h_ent, m_ent)

        if exp_ent < best_exp_entropy - 1e-12:  # strict improvement with tolerance
            best_exp_entropy = exp_ent
            best_token = token

    if best_token is None:
        raise ValueError("candidate_tokens must contain at least one element")
    return best_token


def hybrid_entropy_of_tokens(tokens: Iterable[str], k: int = DEFAULT_K) -> float:
    """
    Convenience wrapper: compute the Shannon entropy of the MinHash signature
    of *tokens*.
    """
    return signature_entropy(signature(tokens, k=k))


# ----------------------------------------------------------------------
# Simple smoke test (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)

    base = ["alpha", "beta", "gamma"]
    candidates = ["delta", "epsilon", "zeta", "alpha"]  # includes a duplicate

    print("Current tokens:", base)
    print("Candidate tokens:", candidates)

    best = hybrid_best_addition(base, candidates, k=64)
    print("\nBest token to add (minimises expected entropy):", best)

    exp_ent = hybrid_expected_entropy_for_addition(base, best, k=64)
    print(f"Expected entropy after adding '{best}': {exp_ent:.6f}")

    cur_ent = hybrid_entropy_of_tokens(base, k=64)
    print(f"Current signature entropy: {cur_ent:.6f}")

    assert exp_ent <= cur_ent + 1e-9, "Expected entropy should not exceed current entropy"

    print("\nSmoke test passed.")