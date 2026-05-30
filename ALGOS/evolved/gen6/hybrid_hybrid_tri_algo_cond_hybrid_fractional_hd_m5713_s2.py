# DARWIN HAMMER — match 5713, survivor 2
# gen: 6
# parent_a: hybrid_tri_algo_conduit_hybrid_cockpit_metri_m68_s2.py (gen2)
# parent_b: hybrid_fractional_hdc_hybrid_hybrid_hybrid_m629_s0.py (gen5)
# born: 2026-05-30T00:04:30Z

import math
import random
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – signal / noise utilities
# ----------------------------------------------------------------------

def _byte_entropy(data: bytes, sample: int = 8192) -> float:
    if not data:
        return 0.0
    chunk = data[:sample]
    return shannon_entropy(list(chunk)) / 8.0


def shannon_entropy(sequence: List[int]) -> float:
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0
    freq = {}
    for item in sequence:
        freq[item] = freq.get(item, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / seq_len
        entropy -= p * math.log(p, 2)
    return entropy


def signal_scores(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    size = len(data)
    entropy = _byte_entropy(data)
    status_bonus = 0.18 if status_code and 200 <= status_code < 300 else -0.10
    mime_bonus = (
        0.12
        if any(x in (mime or "").lower() for x in ("html", "json", "text", "xml"))
        else 0.02
    )
    size_bonus = min(0.22, math.log1p(size) / 60.0)
    keyword_bonus = min(0.20, keyword_hits * 0.05)
    structure_bonus = min(0.16, structural_links * 0.01)

    signal = max(
        0.0,
        min(
            1.0,
            0.20
            + status_bonus
            + mime_bonus
            + size_bonus
            + keyword_bonus
            + structure_bonus
            + 0.12 * entropy,
        ),
    )
    noise = max(
        0.0,
        min(
            1.0,
            0.58
            - 0.22 * entropy
            - keyword_bonus
            - structure_bonus
            + (0.12 if size < 64 else 0.0),
        ),
    )
    return signal, noise


# ----------------------------------------------------------------------
# Parent B – hyperdimensional utilities
# ----------------------------------------------------------------------

def random_hv(dim: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "complex":
        angles = rng.random(dim) * 2 * math.pi
        hv = np.exp(1j * angles)  
        return hv
    elif kind == "binary":
        hv = rng.choice([-1, 1], size=dim)
        return hv.astype(float)
    else:
        raise ValueError(f"Unsupported kind: {kind}")


def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Vectors must have the same shape for convolution")
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return np.real(np.fft.ifft(fa * fb))


def hoeffding_epsilon(n: int, delta: float = 0.05) -> float:
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))


def tropical_gain(vec: np.ndarray) -> float:
    return float(np.max(np.real(vec)))


# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ConduitDecision:
    action: str
    confidence_gap: float
    epsilon: float
    signal_score: float
    noise_score: float
    dormancy_probability: float
    recovery_priority: float
    reason: str


# ----------------------------------------------------------------------
# Hybrid core functions (minimum three)
# ----------------------------------------------------------------------

def compute_signal_noise(
    data: bytes,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Tuple[float, float]:
    return signal_scores(data, status_code, mime, keyword_hits, structural_links)


def bind_signal_to_concept(
    signal: float, concept: str, dim: int = 1024, seed: int | None = None
) -> np.ndarray:
    hv_concept = random_hv(dim=dim, kind="complex", seed=seed or hash(concept) & 0xFFFFFFFF)
    hv_signal = np.full(dim, signal, dtype=complex)
    bound = circular_convolution(hv_concept, hv_signal)
    return bound


def hybrid_decision(
    data: bytes,
    concept: str,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> ConduitDecision:
    signal, noise = compute_signal_noise(
        data, status_code, mime, keyword_hits, structural_links
    )
    confidence_gap = signal - noise

    bound_vec = bind_signal_to_concept(signal, concept, dim=2048)

    gain = tropical_gain(bound_vec)

    epsilon = hoeffding_epsilon(len(data))

    regret = max(0.0, epsilon - gain)

    action = "accept" if confidence_gap > 0 else "reject"
    reason = (
        "Signal dominates noise and regret is low"
        if confidence_gap > 0 and regret < epsilon * 0.5
        else "Signal weak or regret high"
    )

    return ConduitDecision(
        action=action,
        confidence_gap=confidence_gap,
        epsilon=epsilon,
        signal_score=signal,
        noise_score=noise,
        dormancy_probability=regret,
        recovery_priority=gain,
        reason=reason,
    )


def evaluate_batch(
    batch: List[bytes],
    concept: str,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> List[ConduitDecision]:
    decisions = []
    for data in batch:
        decisions.append(hybrid_decision(data, concept, status_code, mime, keyword_hits, structural_links))
    return decisions

def improved_hybrid_decision(
    data: bytes,
    concept: str,
    status_code: int | None = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    delta: float = 0.01,
) -> ConduitDecision:
    signal, noise = compute_signal_noise(
        data, status_code, mime, keyword_hits, structural_links
    )
    confidence_gap = signal - noise

    dim = int(math.log2(len(data))) + 1
    if dim < 10:
        dim = 10
    bound_vec = bind_signal_to_concept(signal, concept, dim=dim)

    gain = tropical_gain(bound_vec)

    epsilon = hoeffding_epsilon(len(data), delta=delta)

    regret = max(0.0, epsilon - gain)

    action = "accept" if confidence_gap > 0 else "reject"
    reason = (
        "Signal dominates noise and regret is low"
        if confidence_gap > 0 and regret < epsilon * 0.5
        else "Signal weak or regret high"
    )

    return ConduitDecision(
        action=action,
        confidence_gap=confidence_gap,
        epsilon=epsilon,
        signal_score=signal,
        noise_score=noise,
        dormancy_probability=regret,
        recovery_priority=gain,
        reason=reason,
    )

def main():
    concept = "test_concept"
    batch = [b"test_data"]
    decisions = evaluate_batch(batch, concept)
    improved_decisions = [improved_hybrid_decision(data, concept) for data in batch]
    for i in range(len(decisions)):
        print(f"Original Decision: {decisions[i]}")
        print(f"Improved Decision: {improved_decisions[i]}")

if __name__ == "__main__":
    main()