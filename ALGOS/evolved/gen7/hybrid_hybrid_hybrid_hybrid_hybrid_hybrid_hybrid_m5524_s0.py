# DARWIN HAMMER — match 5524, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_infota_hybrid_hybrid_hybrid_m1526_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1365_s1.py (gen6)
# born: 2026-05-30T00:02:31Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_infota_hybrid_hybrid_hybrid_m1526_s2.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1365_s1.py

Mathematical Bridge
-------------------
Both parents rely on a MinHash signature as the core representation of a token set.
Parent A extracts uncertainty via Shannon entropy of the signature distribution and
uses the similarity as a “force magnitude” in a drag‑based cost model.
Parent B evaluates uncertainty with a Hoeffding bound on the same signature and
uses it inside a regret‑weighted decision rule.

The hybrid fuses these two uncertainty measures into a single scalar
`U(sig) = α·Ê(sig) + (1‑α)·Ĥ(sig)`,
where `Ê` is the normalized entropy of the signature histogram and `Ĥ` is the
Hoeffding bound normalised to the same range.  `U(sig)` then modulates both the
drag‑based cost and the regret‑weighted strategy, providing a unified pipeline
that respects the physics‑inspired cost model of Parent A and the statistical
confidence model of Parent B.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared deterministic 64‑bit hash
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


MAX64 = (1 << 64) - 1

# ----------------------------------------------------------------------
# Core MinHash utilities (present in both parents)
# ----------------------------------------------------------------------
def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """Compute the MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Uncertainty measures from the two parents
# ----------------------------------------------------------------------
def signature_entropy(sig: List[int], bins: int = 256, eps: float = 1e-12) -> float:
    """Shannon entropy of the histogram of a MinHash signature."""
    hist, _ = np.histogram(sig, bins=bins, range=(0, MAX64))
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]  # drop zeros to avoid log(0)
    return -np.sum(probs * np.log(np.maximum(probs, eps)))


def hoeffding_bound(sig: List[int], n_samples: int, delta: float = 0.05) -> float:
    """
    Hoeffding bound on the mean of hash values.
    Values lie in [0, MAX64]; the bound on the deviation of the empirical mean
    from the true mean is:
        ε = sqrt( ( (b‑a)^2 / (2·n) ) * ln(2/δ) )
    where a=0, b=MAX64.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    range_sq = (MAX64) ** 2
    epsilon = math.sqrt((range_sq / (2 * n_samples)) * math.log(2 / delta))
    # Normalise by the maximum possible deviation (MAX64) to obtain a [0,1] scale
    return min(epsilon / MAX64, 1.0)


def combined_uncertainty(
    sig: List[int],
    n_samples: int,
    delta: float = 0.05,
    alpha: float = 0.5,
    bins: int = 256,
) -> float:
    """
    Fuse entropy and Hoeffding bound into a single scalar.
    Both components are normalised to [0,1] before blending.
    """
    # Normalised entropy: max entropy occurs for uniform histogram
    max_entropy = math.log(bins)
    ent = signature_entropy(sig, bins=bins) / max_entropy if max_entropy > 0 else 0.0

    hoeff = hoeffding_bound(sig, n_samples, delta=delta)

    return alpha * ent + (1.0 - alpha) * hoeff


# ----------------------------------------------------------------------
# Physics‑inspired cost model (Parent A) enriched by hybrid uncertainty
# ----------------------------------------------------------------------
def drag_cost(
    sig_a: List[int],
    sig_b: List[int],
    rho: float = 1.225,
    Cd: float = 0.47,
    area: float = 1.0,
    velocity: float = 1.0,
) -> float:
    """
    Compute a drag‑based cost where the similarity between signatures acts as a
    dimensionless force coefficient.
    Cost = 0.5 * rho * Cd * A * v² * similarity
    """
    sim = minhash_similarity(sig_a, sig_b)
    return 0.5 * rho * Cd * area * (velocity ** 2) * sim


def vram_allocation(probabilities: List[float], base_mem: float = 256.0) -> List[float]:
    """
    Allocate VRAM (in MB) proportionally to the entropy‑scaled probabilities.
    The higher the entropy of the distribution, the more even the allocation.
    """
    ent = -sum(p * math.log(p + 1e-12) for p in probabilities if p > 0)
    max_ent = math.log(len(probabilities)) if probabilities else 1.0
    scale = (ent / max_ent) if max_ent > 0 else 0.0
    total = sum(probabilities)
    if total == 0:
        return [0.0] * len(probabilities)
    normalized = [p / total for p in probabilities]
    return [base_mem * scale * n for n in normalized]


# ----------------------------------------------------------------------
# Regret‑weighted decision rule (Parent B) with hybrid uncertainty
# ----------------------------------------------------------------------
def regret_weighted_strategy(
    actions: List[str],
    sigs: Dict[str, List[int]],
    regrets: Dict[str, float],
    counts: Dict[str, int],
    n_samples: int,
    delta: float = 0.05,
    alpha: float = 0.5,
) -> Tuple[str, float]:
    """
    Choose an action minimizing the uncertainty‑adjusted regret.

    For each action a:
        U_a = combined_uncertainty(sig_a, n_samples, delta, alpha)
        score_a = regrets[a] * (1 + U_a) / max(1, counts.get(a, 1))

    Returns the action with the lowest score and the associated score.
    """
    best_action = None
    best_score = float("inf")
    for a in actions:
        sig = sigs.get(a)
        if sig is None:
            continue
        U = combined_uncertainty(sig, n_samples, delta=delta, alpha=alpha)
        regret = regrets.get(a, 0.0)
        cnt = max(1, counts.get(a, 1))
        score = regret * (1.0 + U) / cnt
        if score < best_score:
            best_score = score
            best_action = a
    return best_action, best_score


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two simple token sets
    tokens_a = ["alpha", "beta", "gamma", "delta"]
    tokens_b = ["beta", "epsilon", "zeta", "eta"]

    # Compute MinHash signatures
    sig_a = minhash_signature(tokens_a, k=128)
    sig_b = minhash_signature(tokens_b, k=128)

    # Demonstrate similarity and drag cost
    sim = minhash_similarity(sig_a, sig_b)
    cost = drag_cost(sig_a, sig_b, velocity=2.0)
    print(f"Similarity: {sim:.4f}")
    print(f"Drag‑based cost (v=2.0): {cost:.4f}")

    # Uncertainty measures
    ent = signature_entropy(sig_a)
    ho = hoeffding_bound(sig_a, n_samples=500)
    combined = combined_uncertainty(sig_a, n_samples=500)
    print(f"Entropy: {ent:.4f}")
    print(f"Hoeffding bound (norm): {ho:.4f}")
    print(f"Combined uncertainty: {combined:.4f}")

    # VRAM allocation example
    probs = [0.2, 0.5, 0.3]
    vram = vram_allocation(probs, base_mem=512)
    print(f"VRAM allocation (MB): {vram}")

    # Regret‑weighted strategy demo
    actions = ["A", "B"]
    sigs = {"A": sig_a, "B": sig_b}
    regrets = {"A": 0.7, "B": 0.4}
    counts = {"A": 3, "B": 5}
    chosen, score = regret_weighted_strategy(
        actions, sigs, regrets, counts, n_samples=500
    )
    print(f"Chosen action: {chosen} with adjusted regret score {score:.4f}")

    sys.exit(0)