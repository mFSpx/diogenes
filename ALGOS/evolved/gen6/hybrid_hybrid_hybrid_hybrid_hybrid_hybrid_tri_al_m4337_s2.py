# DARWIN HAMMER — match 4337, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s0.py (gen5)
# parent_b: hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s0.py (gen5)
# born: 2026-05-29T23:54:58Z

"""
Hybrid Module: hybrid_hybrid_hybrid_hybrid_hoeffding_tree_m1996_s0.py + hybrid_hybrid_tri_algo_cond_hybrid_hybrid_hybrid_m1626_s0.py
The mathematical bridge is the use of probability distributions to drive decision making.
Algorithm A provides a deterministic, hash‑seeded stylometric feature vector which we
interpret as a raw “pheromone” weight vector.  Algorithm B defines signal/noise scores
and an entropy‑based regularisation for a pheromone distribution.  By applying a soft‑max
to the stylometric vector we obtain a probability distribution (pheromone levels).  The
entropy of this distribution is combined with the signal/noise scores to form a
single information‑gain‑like metric.  This metric is then fed to a Hoeffding‑bound
test (the core of Algorithm A’s Hoeffding tree) to decide whether enough evidence
exists to trigger a split/action.  The resulting hybrid therefore fuses the streaming
decision criterion of a Hoeffding tree with the signal‑entropy regularisation of the
tri‑algo/pheromone model.
"""

import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A (stylometry / deterministic hash)
# ----------------------------------------------------------------------
def _deterministic_hash(text: str) -> int:
    """Deterministic 64‑bit hash based on SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def words(text: str) -> List[str]:
    """Simple tokeniser – lower‑case alphabetic words only."""
    return [w for w in text.lower().split() if w.isalpha()]


def stylometry_features(text: str, dim: int = 96) -> np.ndarray:
    """
    Produce a reproducible random feature vector from a text string.
    The vector is seeded by a deterministic hash of the whole text, mimicking
    the probabilistic linguistic model of Parent A.
    """
    seed = _deterministic_hash(text)
    rng = np.random.default_rng(seed)
    # Normalised to unit L2 norm so it can be interpreted as a probability
    # weight vector after a soft‑max later.
    vec = rng.normal(loc=0.0, scale=1.0, size=dim)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


# ----------------------------------------------------------------------
# Utilities from Parent B (signal scores, entropy, pheromone regularisation)
# ----------------------------------------------------------------------
def shannon_entropy(sequence: List[int]) -> float:
    """Classic Shannon entropy in bits."""
    if not sequence:
        return 0.0
    length = len(sequence)
    freq: Dict[int, int] = {}
    for item in sequence:
        freq[item] = freq.get(item, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log(p, 2)
    return entropy


def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    """Entropy of the first *sample* bytes, normalised to [0,1]."""
    if not data:
        return 0.0
    chunk = data[:sample]
    # Convert bytes to list of ints for shannon_entropy
    return shannon_entropy(list(chunk)) / 8.0  # divide by 8 to map to [0,1]


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    """
    Compute a pair (signal_score, noise_score) following Parent B.
    The formulation mirrors the original but is self‑contained.
    """
    size = len(data)
    entropy = _byte_entropy(data)

    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = 0.12 if any(tok in (mime or "").lower() for tok in ("html", "json", "xml")) else 0.0

    keyword_bonus = 0.05 * min(keyword_hits, 5)
    link_bonus = 0.03 * min(structural_links, 10)

    # Base signal is larger for larger payloads and higher entropy (more information)
    signal = (math.log1p(size) * (1 + entropy)) + status_bonus + mime_bonus + keyword_bonus + link_bonus

    # Noise is modelled as the inverse of signal with a small random jitter
    rng = np.random.default_rng(_deterministic_hash(mime + str(status_code)))
    jitter = rng.normal(loc=0.0, scale=0.02)
    noise = max(0.0, 1.0 / (signal + 1e-6) + jitter)

    return float(signal), float(noise)


def pheromone_distribution(features: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Convert raw stylometric features into a probability distribution (pheromone levels)
    using a temperature‑controlled soft‑max.
    """
    scaled = features / max(temperature, 1e-12)
    exps = np.exp(scaled - np.max(scaled))  # stability shift
    probs = exps / (exps.sum() + 1e-12)
    return probs


def expected_entropy(dist: np.ndarray) -> float:
    """Entropy of a discrete probability distribution (in bits)."""
    # Guard against zero entries
    safe_dist = np.clip(dist, 1e-12, 1.0)
    return -np.sum(safe_dist * np.log2(safe_dist))


# ----------------------------------------------------------------------
# Hybrid core – combines both parent logics
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConduitDecision:
    action: str
    confidence_gap: float
    epsilon: float
    combined_metric: float
    signal_score: float
    pheromone_entropy: float
    reason: str


def combined_metric(
    data: bytes,
    text: str,
    *,
    delta: float = 1e-5,
    temperature: float = 0.5,
    pheromone_signal_reg: float = 0.42,
) -> Tuple[float, float]:
    """
    Compute the hybrid metric that will be fed to a Hoeffding bound.
    Returns (metric, epsilon) where epsilon is the Hoeffding bound for the
    current number of observed instances (here approximated by data length).
    """
    # 1. Stylometric features → pheromone distribution
    feats = stylometry_features(text)
    pher_dist = pheromone_distribution(feats, temperature=temperature)

    # 2. Entropy of the pheromone distribution
    pher_entropy = expected_entropy(pher_dist)  # in bits

    # 3. Signal / noise scores from Parent B
    signal, noise = signal_scores(data)

    # 4. Regularised hybrid score:
    #    high signal + high pheromone entropy is desirable.
    #    We penalise high noise.
    hybrid_score = (
        signal * (1 + pheromone_signal_reg * pher_entropy) - noise
    )

    # 5. Hoeffding bound (range R≈1 for a normalised score)
    n = max(len(data), 1)
    R = 1.0
    epsilon = math.sqrt((R * R * math.log(1.0 / delta)) / (2.0 * n))

    return float(hybrid_score), float(epsilon)


def hybrid_hoeffding_decision(
    data: bytes,
    text: str,
    *,
    delta: float = 1e-5,
    temperature: float = 0.5,
    pheromone_signal_reg: float = 0.42,
) -> ConduitDecision:
    """
    Apply a Hoeffding‑tree‑style decision test on the hybrid metric.
    If the metric exceeds epsilon we deem the evidence sufficient to trigger an
    'ACT' action; otherwise we 'WAIT'.
    """
    metric, epsilon = combined_metric(
        data,
        text,
        delta=delta,
        temperature=temperature,
        pheromone_signal_reg=pheromone_signal_reg,
    )

    # Confidence gap is the amount by which metric surpasses the bound
    confidence_gap = metric - epsilon

    if confidence_gap > 0:
        action = "ACT"
        reason = "Metric exceeds Hoeffding bound"
    else:
        action = "WAIT"
        reason = "Insufficient evidence"

    return ConduitDecision(
        action=action,
        confidence_gap=confidence_gap,
        epsilon=epsilon,
        combined_metric=metric,
        signal_score=signal_scores(data)[0],
        pheromone_entropy=expected_entropy(pheromone_distribution(stylometry_features(text))),
        reason=reason,
    )


def batch_hybrid_process(
    payloads: List[bytes],
    texts: List[str],
) -> List[ConduitDecision]:
    """
    Process a batch of (data, text) pairs, returning a list of decisions.
    Demonstrates the hybrid algorithm operating over multiple instances.
    """
    if len(payloads) != len(texts):
        raise ValueError("payloads and texts must have the same length")
    decisions: List[ConduitDecision] = []
    for d, t in zip(payloads, texts):
        decisions.append(hybrid_hoeffding_decision(d, t))
    return decisions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_data = b'{"status":200,"msg":"OK","payload":"Hello world!"}'
    decision = hybrid_hoeffding_decision(sample_data, sample_text)
    print("Single decision:", decision)

    batch = batch_hybrid_process(
        [sample_data, b"<html><body>Test</body></html>", b"\x00\xff\xaa\xbb"],
        [
            sample_text,
            "HTML content with <b>tags</b> and some text.",
            "Binary blob with no readable words.",
        ],
    )
    for i, d in enumerate(batch, 1):
        print(f"Batch {i} decision:", d)