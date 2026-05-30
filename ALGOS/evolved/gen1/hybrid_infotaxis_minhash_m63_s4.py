# DARWIN HAMMER — match 63, survivor 4
# gen: 1
# parent_a: infotaxis.py (gen0)
# parent_b: minhash.py (gen0)
# born: 2026-05-29T23:24:19Z

import hashlib
import math
import random
from collections import Counter
from typing import Iterable, List, Set, Tuple, Dict

MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0
    )


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    if not 0 <= p_hit <= 1:
        raise ValueError("p_hit must be in [0,1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


def best_action(actions: Dict[str, Tuple[float, List[float], List[float]]]) -> str:
    if not actions:
        raise ValueError("actions required")
    return min(actions, key=lambda a: (expected_entropy(*actions[a]), a))


def signature_entropy(sig: List[int]) -> float:
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
    return expected_entropy(p_hit, [signature_entropy(sig_hit)], [signature_entropy(sig_miss)])


def hybrid_expected_entropy_for_addition(
    current_tokens: Iterable[str],
    token: str,
    k: int = 128,
) -> float:
    current_set = set(current_tokens)
    sig_current = signature(current_set, k=k)

    hit_set = current_set | {token}
    sig_hit = signature(hit_set, k=k)

    sig_miss = sig_current

    p_hit = similarity(sig_current, sig_hit)

    return expected_signature_entropy(p_hit, sig_hit, sig_miss)


def hybrid_best_addition(
    current_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    k: int = 128,
) -> str:
    actions: Dict[str, Tuple[float, List[int], List[int]]] = {}
    for token in candidate_tokens:
        cur_set = set(current_tokens)
        sig_miss = signature(cur_set, k=k)

        hit_set = cur_set | {token}
        sig_hit = signature(hit_set, k=k)

        p_hit = similarity(sig_miss, sig_hit)

        actions[token] = (p_hit, sig_hit, sig_miss)

    return best_action(actions)


def hybrid_entropy_of_tokens(tokens: Iterable[str], k: int = 128) -> float:
    return signature_entropy(signature(tokens, k=k))


def improved_hybrid_best_addition(
    current_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    k: int = 128,
) -> str:
    def KL_divergence(p: List[float], q: List[float]) -> float:
        eps = 1e-12
        return sum(p_i * math.log(max(p_i, eps) / max(q_i, eps)) for p_i, q_i in zip(p, q) if p_i > 0)

    actions: Dict[str, Tuple[float, List[int], List[int]]] = {}
    for token in candidate_tokens:
        cur_set = set(current_tokens)
        sig_miss = signature(cur_set, k=k)

        hit_set = cur_set | {token}
        sig_hit = signature(hit_set, k=k)

        p_hit = similarity(sig_miss, sig_hit)

        counts_hit = Counter(sig_hit)
        probs_hit = [counts_hit[i] / len(sig_hit) for i in range(k)]
        counts_miss = Counter(sig_miss)
        probs_miss = [counts_miss[i] / len(sig_miss) for i in range(k)]

        KL_hit = KL_divergence(probs_hit, probs_miss)
        KL_miss = KL_divergence(probs_miss, probs_hit)

        actions[token] = (p_hit, [KL_hit], [KL_miss])

    return best_action(actions)


def improved_hybrid_expected_entropy_for_addition(
    current_tokens: Iterable[str],
    token: str,
    k: int = 128,
) -> float:
    current_set = set(current_tokens)
    sig_current = signature(current_set, k=k)

    hit_set = current_set | {token}
    sig_hit = signature(hit_set, k=k)

    sig_miss = sig_current

    p_hit = similarity(sig_current, sig_hit)

    counts_hit = Counter(sig_hit)
    probs_hit = [counts_hit[i] / len(sig_hit) for i in range(k)]
    counts_miss = Counter(sig_miss)
    probs_miss = [counts_miss[i] / len(sig_miss) for i in range(k)]

    KL_hit = sum(p_i * math.log(max(p_i, 1e-12) / max(q_i, 1e-12)) for p_i, q_i in zip(probs_hit, probs_miss) if p_i > 0)
    KL_miss = sum(p_i * math.log(max(p_i, 1e-12) / max(q_i, 1e-12)) for p_i, q_i in zip(probs_miss, probs_hit) if p_i > 0)

    return p_hit * KL_hit + (1 - p_hit) * KL_miss


if __name__ == "__main__":
    random.seed(0)

    base_tokens = ["alpha", "beta", "gamma"]
    candidates = ["delta", "epsilon", "zeta", "alpha"]

    print("Current tokens:", base_tokens)
    print("Candidate tokens:", candidates)

    best = improved_hybrid_best_addition(base_tokens, candidates, k=64)
    print("\nBest token to add (minimises expected entropy):", best)

    exp_ent = improved_hybrid_expected_entropy_for_addition(base_tokens, best, k=64)
    print("Expected entropy after adding '{}': {:.6f}".format(best, exp_ent))

    cur_entropy = hybrid_entropy_of_tokens(base_tokens, k=64)
    print("Current signature entropy: {:.6f}".format(cur_entropy))

    assert exp_ent <= cur_entropy + 1e-9, "Expected entropy should not increase"

    print("\nSmoke test completed successfully.")