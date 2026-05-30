# DARWIN HAMMER — match 5656, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1682_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m1571_s1.py (gen6)
# born: 2026-05-30T00:03:58Z

import math
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np


def gaussian_beam(theta: float, center: float, width: float, amplitude: float = 1.0) -> float:
    """Standard Gaussian beam with configurable amplitude."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return amplitude * math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, amplitude: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam (derivative squared divided by intensity)."""
    intensity = max(gaussian_beam(theta, center, width, amplitude), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Count‑Min Sketch utilities
# ----------------------------------------------------------------------
def _hash(item: Any, seed: int, width: int) -> int:
    """Deterministic hash for Count‑Min Sketch."""
    h = hashlib.sha256(f"{seed}:{item}".encode()).digest()
    return int.from_bytes(h[:8], "big") % width


def count_min_sketch(items: List[Any], width: int = 128, depth: int = 5) -> Tuple[np.ndarray, List[int]]:
    """
    Build a Count‑Min Sketch and return the table together with the
    per‑item minimum count estimates.
    """
    table = np.zeros((depth, width), dtype=np.int64)
    min_counts = []
    for item in items:
        counts = []
        for d in range(depth):
            idx = _hash(item, d, width)
            table[d, idx] += 1
            counts.append(table[d, idx])
        min_counts.append(min(counts))
    return table, min_counts


def estimate_frequency(item: Any, table: np.ndarray) -> int:
    """Estimate frequency of *item* from an existing sketch."""
    depth, width = table.shape
    return min(_hash(item, d, width) for d in range(depth))


# ----------------------------------------------------------------------
# HyperLogLog cardinality estimator (very lightweight version)
# ----------------------------------------------------------------------
def hyperloglog_cardinality(items: List[Any], p: int = 4) -> float:
    """
    Approximate distinct count using a tiny HyperLogLog sketch.
    p determines the number of registers (m = 2**p).
    """
    m = 1 << p
    registers = np.zeros(m, dtype=np.int8)

    for item in items:
        h = int(hashlib.sha256(str(item).encode()).hexdigest(), 16)
        idx = h & (m - 1)                     # low‑p bits as register index
        w = h >> p                            # remaining bits
        leading_zeros = (w.bit_length() - w.bit_length() + 1) if w else 0
        rho = 1
        while w & (1 << (64 - rho)) == 0 and rho < 64:
            rho += 1
        registers[idx] = max(registers[idx], rho)

    alpha_m = {
        16: 0.673,
        32: 0.697,
        64: 0.709,
    }.get(m, 0.7213 / (1 + 1.079 / m))

    Z = 1.0 / np.sum(2.0 ** -registers)
    raw_est = alpha_m * m * m * Z

    # Small range correction
    if raw_est <= (5 / 2) * m:
        V = np.count_nonzero(registers == 0)
        if V != 0:
            raw_est = m * math.log(m / V)

    return raw_est


# ----------------------------------------------------------------------
# Fisher information aggregation
# ----------------------------------------------------------------------
def calculate_fisher_info(items: List[Any],
                          theta: float,
                          center: float,
                          width: float) -> float:
    """
    Compute an aggregated Fisher information score for *items*.
    Each item is mapped to a pseudo‑theta via a hash function.
    The final amplitude is the mean Fisher score across the collection.
    """
    if not items:
        return 1.0

    scores = []
    for item in items:
        # Map item to a deterministic pseudo‑theta in the interval [center-3*width, center+3*width]
        h = int(hashlib.sha256(str(item).encode()).hexdigest(), 16)
        pseudo_theta = center + ( (h % 1000) / 1000.0 - 0.5 ) * 6 * width
        score = fisher_score(pseudo_theta, center, width, amplitude=1.0)
        scores.append(score)

    return float(np.mean(scores))


# ----------------------------------------------------------------------
# Bayesian update with log‑odds to avoid numerical issues
# ----------------------------------------------------------------------
def update_hypothesis(hypothesis: Dict[str, Any],
                      evidence: Dict[str, Any],
                      log_likelihood_ratio: float) -> Dict[str, Any]:
    """
    Perform a Bayesian update using log‑odds.
    hypothesis must contain keys: 'id', 'posterior', 'evidence_ids'.
    """
    if not isinstance(log_likelihood_ratio, float):
        raise TypeError("log_likelihood_ratio must be a float")

    p = max(0.0, min(1.0, hypothesis["posterior"]))
    if p in (0.0, 1.0):
        posterior = p
    else:
        log_odds = math.log(p / (1.0 - p)) + log_likelihood_ratio
        odds = math.exp(log_odds)
        posterior = odds / (1.0 + odds)

    posterior = max(0.0, min(1.0, posterior))
    return {
        "id": hypothesis["id"],
        "prior": hypothesis["posterior"],
        "posterior": posterior,
        "evidence_ids": hypothesis["evidence_ids"] + [evidence["id"]],
    }


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_update_hypothesis(hypothesis: Dict[str, Any],
                             evidence: Dict[str, Any],
                             items: List[Any]) -> Dict[str, Any]:
    """
    Combine Gaussian intensity, Fisher information and Bayesian update.
    The likelihood ratio is accumulated in log‑space.
    """
    # Build a lightweight sketch once – reused for frequency‑based pruning later
    sketch, _ = count_min_sketch(items)

    log_likelihood = 0.0
    for item in items:
        # Frequency‑aware amplitude: higher frequency → slightly damped amplitude
        freq_est = estimate_frequency(item, sketch) + 1e-9
        amp = 1.0 / math.log1p(freq_est)

        intensity = gaussian_beam(
            theta=evidence["theta"],
            center=evidence["center"],
            width=evidence["width"],
            amplitude=amp,
        )
        # Guard against zero intensity
        if intensity <= 0:
            continue
        log_likelihood += math.log(intensity)

    return update_hypothesis(hypothesis, evidence, log_likelihood)


def prune_probability(t: int, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Sigmoid‑shaped probability that grows with the global step *t*."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError("t, lam, and alpha must be non‑negative")
    return 1.0 / (1.0 + math.exp(-lam * (t - alpha)))


def hybrid_prune(items: List[Any],
                 lam: float = 1.0,
                 alpha: float = 0.2,
                 freq_threshold: float = 0.5) -> List[Any]:
    """
    Remove items whose estimated frequency is above a dynamic threshold.
    The threshold adapts with the current list length via *prune_probability*.
    """
    if not items:
        return []

    t = len(items)
    prob = prune_probability(t, lam, alpha)

    # Build sketch once
    sketch, _ = count_min_sketch(items)

    kept = []
    for item in items:
        freq = estimate_frequency(item, sketch)
        # Normalise frequency by current list size
        norm_freq = freq / t
        if norm_freq < prob * freq_threshold:
            kept.append(item)
    return kept


def hybrid_gaussian_beam(theta: float,
                         center: float,
                         width: float,
                         items: List[Any]) -> float:
    """
    Gaussian beam whose amplitude is driven by aggregated Fisher information
    computed from *items*.
    """
    amp = calculate_fisher_info(items, theta, center, width)
    return gaussian_beam(theta, center, width, amplitude=amp)


# ----------------------------------------------------------------------
# Example driver (can be removed or adapted)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hypothesis = {"id": 1, "prior": 0.5, "posterior": 0.5, "evidence_ids": []}
    evidence = {"id": 101, "theta": 1.0, "center": 0.0, "width": 1.0}
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    new_hypothesis = hybrid_update_hypothesis(hypothesis, evidence, items)
    print("Updated hypothesis:", new_hypothesis)

    pruned_items = hybrid_prune(items, lam=1.2, alpha=0.3)
    print("Pruned items:", pruned_items)

    beam_val = hybrid_gaussian_beam(1.0, 0.0, 1.0, items)
    print("Hybrid Gaussian beam value:", beam_val)