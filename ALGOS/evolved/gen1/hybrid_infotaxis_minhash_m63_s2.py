# DARWIN HAMMER — match 63, survivor 2
# gen: 1
# parent_a: infotaxis.py (gen0)
# parent_b: minhash.py (gen0)
# born: 2026-05-29T23:24:19Z

"""Hybrid Infotaxis‑MinHash module.

This module fuses the entropy‑driven decision logic of *infotaxis.py* with the
set‑similarity machinery of *minhash.py*.  The mathematical bridge is the
interpretation of a MinHash signature as a discrete probability distribution
over hash buckets.  The Shannon entropy of that distribution quantifies the
uncertainty of the underlying token set.  When considering a candidate token
addition we treat the Jaccard‑like similarity between the current and the
hypothetical “hit” signature as the probability `p_hit` of observing the
addition.  The expected post‑action entropy is then

    E[H] = p_hit * H(sig_hit) + (1‑p_hit) * H(sig_miss)

which is exactly the `expected_entropy` formula from *infotaxis.py*.  The
action (token) minimising this expectation is selected, yielding a
single unified algorithm that chooses tokens to reduce signature entropy the
most while being guided by MinHash similarity.

The implementation provides three public hybrid functions plus a small
smoke‑test.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# MinHash core (parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Entropy core (parent A)
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution given by *probabilities*."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted average entropy of two possible states."""
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, List[float], List[float]]]) -> str:
    """Select the action with minimal expected entropy."""
    if not actions:
        raise ValueError("actions required")
    return min(actions, key=lambda a: (expected_entropy(*actions[a]), a))


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def signature_entropy(sig: List[int]) -> float:
    """Entropy of a MinHash signature treated as a discrete distribution."""
    if not sig:
        raise ValueError("signature must not be empty")
    counts = Counter(sig)
    probs = list(counts.values())
    return entropy(probs)


def expected_signature_entropy(
    p_hit: float,
    sig_hit: List[int],
    sig_miss: List[int],
) -> float:
    """Expected entropy after a probabilistic observation of a token addition."""
    return expected_entropy(p_hit, [signature_entropy(sig_hit)], [signature_entropy(sig_miss)])


def hybrid_expected_entropy_for_addition(
    current_tokens: Iterable[str],
    token: str,
    k: int = 128,
) -> float:
    """
    Compute the expected signature entropy if *token* were added to *current_tokens*.

    *p_hit* is approximated by the MinHash similarity between the current signature
    and the signature that would result after the addition.
    """
    current_set = set(current_tokens)
    sig_current = signature(current_set, k=k)

    # "hit" state: token is added
    hit_set = current_set | {token}
    sig_hit = signature(hit_set, k=k)

    # "miss" state: token is not added (remains the same)
    sig_miss = sig_current

    # Use similarity as a proxy for probability that the addition is observed.
    p_hit = similarity(sig_current, sig_hit)

    # Expected entropy using the hybrid bridge.
    return expected_signature_entropy(p_hit, sig_hit, sig_miss)


def hybrid_best_addition(
    current_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    k: int = 128,
) -> str:
    """
    Choose the token from *candidate_tokens* that minimises the expected
    signature entropy after a potential addition.
    """
    actions: Dict[str, Tuple[float, List[int], List[int]]] = {}
    for token in candidate_tokens:
        # compute hit and miss signatures once
        cur_set = set(current_tokens)
        sig_miss = signature(cur_set, k=k)

        hit_set = cur_set | {token}
        sig_hit = signature(hit_set, k=k)

        p_hit = similarity(sig_miss, sig_hit)

        # Store raw states for the generic `expected_entropy` function
        actions[token] = (p_hit, sig_hit, sig_miss)

    return best_action(actions)


def hybrid_entropy_of_tokens(tokens: Iterable[str], k: int = 128) -> float:
    """
    Convenience wrapper: compute the Shannon entropy of the MinHash signature
    of *tokens*.
    """
    return signature_entropy(signature(tokens, k=k))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple reproducible example
    random.seed(0)

    base_tokens = ["alpha", "beta", "gamma"]
    candidates = ["delta", "epsilon", "zeta", "alpha"]  # note duplicate

    print("Current tokens:", base_tokens)
    print("Candidate tokens:", candidates)

    best = hybrid_best_addition(base_tokens, candidates, k=64)
    print("\nBest token to add (minimises expected entropy):", best)

    exp_ent = hybrid_expected_entropy_for_addition(base_tokens, best, k=64)
    print("Expected entropy after adding '{}': {:.6f}".format(best, exp_ent))

    cur_entropy = hybrid_entropy_of_tokens(base_tokens, k=64)
    print("Current signature entropy: {:.6f}".format(cur_entropy))

    # Verify that the expected entropy is not larger than the current entropy
    assert exp_ent <= cur_entropy + 1e-9, "Expected entropy should not increase"

    print("\nSmoke test completed successfully.")