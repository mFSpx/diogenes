# DARWIN HAMMER — match 959, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s6.py (gen3)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s0.py (gen3)
# born: 2026-05-29T23:32:02Z

import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Any

import numpy as np

MAX64 = (1 << 64) - 1

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def morphology_vector(m: Morphology) -> np.ndarray:
    return np.array([m.length, m.width, m.height, m.mass], dtype=float).reshape(-1, 1)

def ssim_like_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    C1 = 1e-4
    C2 = 9e-4

    mu1 = np.mean(v1)
    mu2 = np.mean(v2)
    sigma1_sq = np.var(v1)
    sigma2_sq = np.var(v2)
    sigma12 = np.cov(v1.squeeze(), v2.squeeze())[0, 1]

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(numerator / denominator)

def token_frequencies(logs: Iterable[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    token_pattern = re.compile(r"\b\w+\b")
    for line in logs:
        for tok in token_pattern.findall(line.lower()):
            freq[tok] = freq.get(tok, 0) + 1
    return freq

def shannon_entropy_from_counts(counts: Dict[str, int], eps: float = 1e-12) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log(max(c / total, eps)) for c in counts.values())
    return ent / math.log(len(counts) or 1)

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(np.uint64(hash(data) & MAX64).tobytes(), dtype=">u8"), ">u8"
    )

def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [MAX64] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]

def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def signature_entropy(sig: List[int], bins: int = 256) -> float:
    if not sig:
        return 0.0
    hist, _ = np.histogram(sig, bins=bins, range=(0, MAX64))
    probs = hist.astype(float).tolist()
    return entropy(probs)

def reconstruction_risk_score(unique_quasi_ids: int, total_records: int) -> float:
    if total_records <= 0:
        return 1.0
    ratio = unique_quasi_ids / total_records
    return min(max(ratio, 0.0), 1.0)

def hybrid_recovery_score(
    morph_a: Morphology,
    morph_b: Morphology,
    logs_a: Iterable[str],
    logs_b: Iterable[str],
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    unique_quasi_ids: int,
    total_records: int,
    recovery_a: float,
    recovery_b: float,
    α: float = 0.5,
    β: float = 0.3,
    γ: float = 0.6,
) -> float:
    v1 = morphology_vector(morph_a)
    v2 = morphology_vector(morph_b)
    ssim_sim = ssim_like_similarity(v1, v2)

    sig_a = minhash_signature(tokens_a)
    sig_b = minhash_signature(tokens_b)
    mh_sim = minhash_similarity(sig_a, sig_b)

    S = γ * ssim_sim + (1 - γ) * mh_sim

    R = (recovery_a + recovery_b) / 2.0

    freq_a = token_frequencies(logs_a)
    freq_b = token_frequencies(logs_b)
    combined_freq = {k: freq_a.get(k, 0) + freq_b.get(k, 0) for k in set(list(freq_a.keys()) + list(freq_b.keys()))}

    H_t = shannon_entropy_from_counts(combined_freq)
    H_s = (signature_entropy(sig_a) + signature_entropy(sig_b)) / 2.0
    H = (H_t + H_s) / 2.0

    ρ = reconstruction_risk_score(unique_quasi_ids, total_records)

    Ψ = (α * S + (1 - α) * R) * (1 - β * H) * (1 - ρ)
    return Ψ

def improved_hybrid_recovery_score(
    morph_a: Morphology,
    morph_b: Morphology,
    logs_a: Iterable[str],
    logs_b: Iterable[str],
    tokens_a: Iterable[str],
    tokens_b: Iterable[str],
    unique_quasi_ids: int,
    total_records: int,
    recovery_a: float,
    recovery_b: float,
    α: float = 0.5,
    β: float = 0.3,
    γ: float = 0.6,
) -> float:
    v1 = morphology_vector(morph_a)
    v2 = morphology_vector(morph_b)
    ssim_sim = ssim_like_similarity(v1, v2)

    sig_a = minhash_signature(tokens_a)
    sig_b = minhash_signature(tokens_b)
    mh_sim = minhash_similarity(sig_a, sig_b)

    S = γ * ssim_sim + (1 - γ) * mh_sim

    R = (recovery_a + recovery_b) / 2.0

    freq_a = token_frequencies(logs_a)
    freq_b = token_frequencies(logs_b)
    combined_freq = {k: freq_a.get(k, 0) + freq_b.get(k, 0) for k in set(list(freq_a.keys()) + list(freq_b.keys()))}

    H_t = shannon_entropy_from_counts(combined_freq)
    H_s = (signature_entropy(sig_a) + signature_entropy(sig_b)) / 2.0
    H = (H_t + H_s) / 2.0

    ρ = reconstruction_risk_score(unique_quasi_ids, total_records)

    # Introduce a new term to account for the difference in recovery priorities
    ΔR = abs(recovery_a - recovery_b)
    Ψ = (α * S + (1 - α) * R) * (1 - β * H) * (1 - ρ) * (1 - 0.5 * ΔR)
    return Ψ

# Example usage
morph_a = Morphology(1.0, 2.0, 3.0, 4.0)
morph_b = Morphology(5.0, 6.0, 7.0, 8.0)
logs_a = ["log1", "log2"]
logs_b = ["log3", "log4"]
tokens_a = ["token1", "token2"]
tokens_b = ["token3", "token4"]
unique_quasi_ids = 10
total_records = 100
recovery_a = 0.5
recovery_b = 0.7

score = improved_hybrid_recovery_score(
    morph_a,
    morph_b,
    logs_a,
    logs_b,
    tokens_a,
    tokens_b,
    unique_quasi_ids,
    total_records,
    recovery_a,
    recovery_b,
)
print(score)