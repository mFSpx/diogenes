# DARWIN HAMMER — match 5019, survivor 0
# gen: 7
# parent_a: hybrid_infotaxis_minhash_m63_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1927_s1.py (gen6)
# born: 2026-05-29T23:59:23Z

"""Hybrid Infotaxis‑MinHash + Doomsday‑Gini‑Fisher Module.

Parents:
- **hybrid_infotaxis_minhash_m63_s2.py** – entropy‑driven infotaxis built on
  MinHash signatures.  Core equations: Shannon entropy *H(sig)* and expected
  entropy `E[H] = p_hit·H(sig_hit) + (1‑p_hit)·H(sig_miss)`.
- **hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1927_s1.py** – Doomsday
  weekday calculation, Gini coefficient of token frequencies, and Fisher‑style
  information weighting of stylometry‑like features.

Mathematical bridge:
The MinHash signature is interpreted as a discrete probability distribution
over hash buckets.  Its Shannon entropy *H* quantifies uncertainty, while the
Gini coefficient *G* of the underlying token frequencies quantifies inequality
in that distribution.  We fuse them into a composite uncertainty

    U = α·H + β·G ,

with α,β∈[0,1] (α+β≈1).  A temporal modulation derived from the Doomsday
weekday *w* (0=Sunday … 6=Saturday) is applied as a sinusoidal factor

    U_w = U·(1 + sin(2π·w/7)) .

When evaluating a candidate token addition we compute the probability
`p_hit = similarity(sig_current, sig_candidate)` (MinHash Jaccard estimate) and
the expected composite uncertainty

    E[U] = p_hit·U_w(hit) + (1‑p_hit)·U_w(miss) .

The token minimising `E[U]` is selected.  For routing decisions we treat the
signature‑derived probability vector *p* and use its log‑probability variance as
a proxy for Fisher information *I*, yielding a routing cost `C = I·distance`.

The module provides three public hybrid functions demonstrating this unified
system.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple, Dict

import numpy as np

MAX64 = (1 << 64) - 1


# ----------------------------------------------------------------------
# MinHash core (from Parent A)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length *k* for the given token set."""
    toks: Set[str] = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Approximate Jaccard similarity via MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ----------------------------------------------------------------------
# Entropy & Gini (fusion of Parent A & B)
# ----------------------------------------------------------------------
def shannon_entropy(sig: List[int]) -> float:
    """Shannon entropy of the discrete distribution induced by a signature."""
    # Count occurrences of each hash bucket value
    cnt = Counter(sig)
    total = len(sig)
    ent = 0.0
    for c in cnt.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient of a non‑negative value list."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values")
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return gini


# ----------------------------------------------------------------------
# Doomsday weekday (Parent B)
# ----------------------------------------------------------------------
def doomsday_weekday(year: int, month: int, day: int) -> int:
    """Return weekday (0=Sunday … 6=Saturday) using the Doomsday algorithm."""
    # Anchor day for century
    century = year // 100
    anchor = (5 * (century % 4) + 2) % 7  # 0=Sunday
    # Year’s doomsday
    y = year % 100
    doomsday = (y // 12 + (y % 12) + (y % 12) // 4 + anchor) % 7
    # Month’s doomsday table (non‑leap year)
    month_doom = [0, 3, 28, 14, 4, 9, 6, 11, 8, 5, 10, 7, 12]
    # Adjust for leap year
    if month > 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        offset = 1
    else:
        offset = 0
    diff = (day - month_doom[month] - offset) % 7
    return (doomsday + diff) % 7


# ----------------------------------------------------------------------
# Fisher‑style information from a probability vector
# ----------------------------------------------------------------------
def fisher_information(prob_vec: np.ndarray) -> float:
    """Proxy Fisher information: variance of log‑probabilities (ignoring zeros)."""
    eps = np.finfo(float).eps
    safe = np.clip(prob_vec, eps, 1.0)
    logp = np.log(safe)
    return float(np.var(logp))


# ----------------------------------------------------------------------
# Hybrid expected composite uncertainty (core of new algorithm)
# ----------------------------------------------------------------------
def expected_composite_uncertainty(
    current_tokens: Set[str],
    candidate_token: str,
    all_tokens: Set[str],
    k: int = 128,
    alpha: float = 0.7,
    beta: float = 0.3,
    date: Tuple[int, int, int] | None = None,
) -> float:
    """
    Compute the expected composite uncertainty after adding *candidate_token*.

    The composite uncertainty combines Shannon entropy of the MinHash signature
    and Gini coefficient of token frequencies, temporally modulated by the
    Doomsday weekday (if *date* is given).

    Returns the scalar expectation E[U].
    """
    # Base signature for current token set
    sig_cur = minhash_signature(current_tokens, k=k)
    # Signature if the candidate were added (hit)
    sig_hit = minhash_signature(current_tokens | {candidate_token}, k=k)
    # Signature if the candidate were ignored (miss) – identical to cur
    sig_miss = sig_cur

    # Probability of observing the candidate as a “hit”
    p_hit = minhash_similarity(sig_cur, sig_hit)

    # Helper to compute composite uncertainty for a signature
    def composite(sig: List[int], token_set: Set[str]) -> float:
        H = shannon_entropy(sig)
        # Gini over token frequencies (simple count of each token)
        freq = Counter(token_set).values()
        G = gini_coefficient(freq)
        U = alpha * H + beta * G
        if date is not None:
            w = doomsday_weekday(*date)
            U *= 1.0 + math.sin(2 * math.pi * w / 7.0)
        return U

    U_hit = composite(sig_hit, current_tokens | {candidate_token})
    U_miss = composite(sig_miss, current_tokens)

    return p_hit * U_hit + (1.0 - p_hit) * U_miss


def hybrid_select_token(
    current_tokens: Set[str],
    candidate_pool: Iterable[str],
    all_tokens: Set[str],
    k: int = 128,
    alpha: float = 0.7,
    beta: float = 0.3,
    date: Tuple[int, int, int] | None = None,
) -> str:
    """
    Choose the token from *candidate_pool* that minimises the expected composite
    uncertainty defined in :func:`expected_composite_uncertainty`.
    """
    best_token = None
    best_score = float("inf")
    for cand in candidate_pool:
        if cand in current_tokens:
            continue
        score = expected_composite_uncertainty(
            current_tokens,
            cand,
            all_tokens,
            k=k,
            alpha=alpha,
            beta=beta,
            date=date,
        )
        if score < best_score:
            best_score = score
            best_token = cand
    if best_token is None:
        raise ValueError("No valid candidate token found")
    return best_token


# ----------------------------------------------------------------------
# Entity routing using Fisher information (demonstrates the other side of Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    tokens: Tuple[str, ...]  # underlying token set describing the entity
    lat: float
    lon: float
    category: str


def signature_vector(entity: Entity, k: int = 128) -> np.ndarray:
    """
    Produce a normalised probability vector from the MinHash signature of an
    entity's token set.
    """
    sig = minhash_signature(entity.tokens, k=k)
    cnt = Counter(sig)
    total = sum(cnt.values())
    prob = np.array([cnt[i] / total for i in range(k)], dtype=float)
    return prob


def routing_cost(source: Entity, target: Entity, k: int = 128) -> float:
    """
    Compute a routing cost between *source* and *target*.

    Cost = Fisher information of source's signature vector multiplied by the
    Euclidean geographic distance.  This mirrors the Fisher‑SSIM routing idea.
    """
    p_src = signature_vector(source, k=k)
    I = fisher_information(p_src)
    # Haversine distance (km)
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [source.lat, source.lon, target.lat, target.lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return I * distance


# ----------------------------------------------------------------------
# Public API (three demonstrative functions)
# ----------------------------------------------------------------------
def hybrid_entropy_gini(tokens: Iterable[str], k: int = 128) -> Tuple[float, float]:
    """
    Return the pair (Shannon entropy, Gini coefficient) for the token set.
    """
    sig = minhash_signature(tokens, k=k)
    H = shannon_entropy(sig)
    G = gini_coefficient(Counter(tokens).values())
    return H, G


def hybrid_optimal_token(
    current: Set[str],
    candidates: List[str],
    universe: Set[str],
    date: Tuple[int, int, int] | None = None,
) -> str:
    """
    Wrapper around :func:`hybrid_select_token` with default parameters.
    """
    return hybrid_select_token(
        current_tokens=current,
        candidate_pool=candidates,
        all_tokens=universe,
        k=128,
        alpha=0.6,
        beta=0.4,
        date=date,
    )


def hybrid_route(source: Entity, target: Entity) -> float:
    """
    Compute routing cost between two entities using the hybrid Fisher‑SSIM metric.
    """
    return routing_cost(source, target, k=128)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Token universe and initial set
    universe = {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima"
    }
    current_set = {"alpha", "bravo", "charlie"}
    candidate_tokens = ["delta", "echo", "foxtrot", "golf"]

    # Choose token based on hybrid expected uncertainty (today's date)
    today = (2026, 5, 29)
    chosen = hybrid_optimal_token(current_set, candidate_tokens, universe, date=today)
    print(f"Chosen token: {chosen}")

    # Show entropy & Gini for the new set
    new_set = current_set | {chosen}
    H, G = hybrid_entropy_gini(new_set)
    print(f"After addition -> Entropy: {H:.4f}, Gini: {G:.4f}")

    # Create two dummy entities and compute routing cost
    e1 = Entity(id="E1", tokens=tuple(new_set), lat=37.7749, lon=-122.4194, category="A")
    e2 = Entity(id="E2", tokens=("delta", "echo", "foxtrot"), lat=40.7128, lon=-74.0060, category="B")
    cost = hybrid_route(e1, e2)
    print(f"Routing cost from {e1.id} to {e2.id}: {cost:.6f}")

    sys.exit(0)