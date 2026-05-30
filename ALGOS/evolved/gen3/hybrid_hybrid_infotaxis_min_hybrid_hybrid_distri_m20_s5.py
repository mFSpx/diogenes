# DARWIN HAMMER — match 20, survivor 5
# gen: 3
# parent_a: hybrid_infotaxis_minhash_m63_s0.py (gen1)
# parent_b: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s1.py (gen2)
# born: 2026-05-29T23:25:08Z

"""Hybrid Entropic MinHash‑Strike (HEMS)

This module fuses the two parent algorithms:

* **Parent A** – Entropic MinHash (infotaxis + minhash).  It provides
  - entropy of a probability distribution,
  - a MinHash signature for a list of probabilities,
  - Jaccard‑like similarity between signatures.

* **Parent B** – Chelydrid ambush‑strike physics.  It provides
  - a binary “d‑hash” derived from a numeric sequence,
  - Hamming distance as a similarity measure,
  - a drag‑limited integration of a force series (the “strike” dynamics).

**Mathematical bridge**

A MinHash signature is a vector of integers `s = (s₁,…,s_k)`.  
From this vector we construct a binary differential hash


d_i = 1  if s_i > s_{i+1}
     0  otherwise


which yields a `k‑1`‑bit integer.  The Hamming distance between two
such bit‑patterns is a metric on the space of signatures.  We treat the
(normalised) Hamming similarity as a *force* that drives a search
agent through the entropy landscape of the underlying probability
distributions.  The agent’s motion obeys the drag equation used in the
original strike model, thus coupling the entropy‑based information gain
with the physics‑based cost model.

The three core functions below illustrate this hybrid operation:

1. `entropic_minhash` – builds a MinHash signature from a probability
   distribution.
2. `signature_hamming_force` – converts two signatures to d‑hashes,
   computes normalised Hamming similarity and scales it by the expected
   entropy reduction.
3. `strike_from_distributions` – runs the drag‑limited integration using
   the force from (2) and returns the final `StrikeState`.

The result is a unified system that can be used for information‑theoretic
search, clustering, or any task where similarity of probability
distributions must be evaluated under a physical cost model.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
import pathlib
from collections.abc import Iterable, Mapping, Hashable
from dataclasses import dataclass
from typing import List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – entropy and MinHash utilities
# ----------------------------------------------------------------------


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a (non‑negative) probability mass vector."""
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0.0
    )


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Standard MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def entropic_minhash(probabilities: List[float], k: int = 128) -> List[int]:
    """Generate a MinHash signature directly from a probability distribution."""
    tokens = [f"{p:.12g}" for p in probabilities]  # deterministic string tokens
    return signature(tokens, k)


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """
    Expected entropy after a binary observation.

    p_hit  – probability that the observation is a “hit”.
    hit_state – probability vector if the observation is a hit.
    miss_state – probability vector if the observation is a miss.
    """
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)


# ----------------------------------------------------------------------
# Parent B – binary hash, Hamming distance and strike dynamics
# ----------------------------------------------------------------------


def compute_dhash(values: List[float]) -> int:
    """Differential hash: bit i = 1 if values[i] > values[i+1]."""
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def normalised_hamming_similarity(a: int, b: int, length: int) -> float:
    """Similarity in [0,1] derived from Hamming distance."""
    if length <= 0:
        raise ValueError("length must be positive")
    return 1.0 - hamming_distance(a, b) / length


@dataclass(frozen=True)
class StrikeState:
    """Result of integrating a force series under quadratic drag."""
    velocity: float
    distance: float
    peak_velocity: float


def integrate_strike(
    force_series: Iterable[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> StrikeState:
    """Quadratic‑drag integration (Chelydrid ambush‑strike model)."""
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        drag = drag_cd * fluid_density * area * v * abs(v) / (2.0 * m_head)
        acc = force / m_head - drag
        v = max(0.0, v + acc * dt)
        x += v * dt
        peak = max(peak, v)
    return StrikeState(v, x, peak)


def pulse_force(peak_force: float, steps: int) -> List[float]:
    """Triangular pulse of length `steps` with given peak amplitude."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [
        peak_force * max(0.0, 1.0 - abs(i - mid) / denom) for i in range(steps)
    ]


# ----------------------------------------------------------------------
# Hybrid layer – coupling entropy/minhash with strike dynamics
# ----------------------------------------------------------------------


def signature_hamming_force(
    prob_a: List[float],
    prob_b: List[float],
    k: int = 128,
    dt: float = 0.01,
    m_head: float = 0.01,
) -> StrikeState:
    """
    Compute a strike state whose driving force is the entropy‑weighted
    Hamming similarity of the MinHash signatures of two probability
    distributions.

    Steps
    -----
    1. Build MinHash signatures `s_a`, `s_b`.
    2. Convert each signature to a differential hash integer `d_a`,
       `d_b`.
    3. Normalise Hamming similarity `sim ∈ [0,1]`.
    4. Estimate the expected entropy reduction if we were to choose the
       higher‑entropy distribution as the “hit”.
    5. Use `force = sim * (entropy_a - entropy_b)` as the peak force of a
       triangular pulse and integrate the strike dynamics.
    """
    # 1. MinHash signatures
    s_a = entropic_minhash(prob_a, k)
    s_b = entropic_minhash(prob_b, k)

    # 2. Differential hashes (k‑1 bits)
    d_a = compute_dhash(s_a)
    d_b = compute_dhash(s_b)

    # 3. Normalised Hamming similarity
    sim = normalised_hamming_similarity(d_a, d_b, length=k - 1)

    # 4. Expected entropy reduction (simple proxy)
    ent_a = entropy(prob_a)
    ent_b = entropy(prob_b)
    # probability that distribution A is “better” (lower entropy)
    p_hit = 0.5 * (1.0 + (ent_b - ent_a) / max(ent_a + ent_b, 1e-12))
    p_hit = max(0.0, min(1.0, p_hit))
    exp_ent = expected_entropy(p_hit, prob_a, prob_b)
    # reduction relative to the current average entropy
    avg_ent = 0.5 * (ent_a + ent_b)
    entropy_gain = max(0.0, avg_ent - exp_ent)

    # 5. Force series: similarity modulated by entropy gain
    peak_force = sim * entropy_gain * 10.0  # scaling factor for visibility
    force_series = pulse_force(peak_force, steps=12)

    return integrate_strike(force_series, dt=dt, m_head=m_head)


def hybrid_similarity(
    prob_a: List[float],
    prob_b: List[float],
    k: int = 128,
) -> float:
    """
    Return a composite similarity score that blends
    - Jaccard‑like MinHash similarity,
    - Normalised Hamming similarity of the differential hashes,
    - Entropy‑based weighting.
    """
    # MinHash Jaccard similarity
    sig_a = entropic_minhash(prob_a, k)
    sig_b = entropic_minhash(prob_b, k)
    jaccard = sum(1 for a, b in zip(sig_a, sig_b) if a == b) / k

    # Hamming similarity on differential hashes
    dh_a = compute_dhash(sig_a)
    dh_b = compute_dhash(sig_b)
    ham_sim = normalised_hamming_similarity(dh_a, dh_b, length=k - 1)

    # Entropy weighting (higher entropy → lower confidence)
    ent_a = entropy(prob_a)
    ent_b = entropy(prob_b)
    weight = 1.0 / (1.0 + 0.5 * (ent_a + ent_b))

    # Blend the three components
    return weight * (0.6 * jaccard + 0.4 * ham_sim)


def cluster_representative(
    distributions: List[List[float]],
    k: int = 128,
) -> List[float]:
    """
    Choose a representative probability distribution from a cluster
    using the ambush‑strike cost model.

    For each candidate `d_i` we compute the total strike cost to all
    other members.  The candidate with the minimal accumulated cost is
    returned.
    """
    n = len(distributions)
    if n == 0:
        raise ValueError("empty cluster")
    costs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            state = signature_hamming_force(
                distributions[i], distributions[j], k=k, dt=0.01, m_head=0.01
            )
            # Use distance travelled as a proxy for cost
            costs[i] += state.distance
    best_idx = int(np.argmin(costs))
    return distributions[best_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate two random probability vectors
    rng = np.random.default_rng(42)
    a = rng.random(10).tolist()
    b = rng.random(10).tolist()

    # Normalise to proper distributions
    a = [x / sum(a) for x in a]
    b = [x / sum(b) for x in b]

    # Hybrid similarity
    sim = hybrid_similarity(a, b)
    print(f"Hybrid similarity: {sim:.4f}")

    # Strike dynamics driven by the two distributions
    strike = signature_hamming_force(a, b)
    print(f"Strike result – velocity: {strike.velocity:.4f}, distance: {strike.distance:.4f}, peak: {strike.peak_velocity:.4f}")

    # Cluster representative demo
    cluster = [a, b, [0.1] * 10, [0.9] + [0.01] * 9]
    rep = cluster_representative(cluster)
    print(f"Representative distribution (first 5 entries): {rep[:5]}")