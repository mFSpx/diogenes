# DARWIN HAMMER — match 959, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s6.py (gen3)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s0.py (gen3)
# born: 2026-05-29T23:32:02Z

"""Hybrid Endpoint‑Similarity & Model‑Pooling Fusion

This module merges the core mathematics of two parent algorithms:

* **Parent A** – endpoint morphology vectors, SSIM‑based similarity, and a
  decision‑hygiene entropy over token categories.
* **Parent B** – MinHash signatures for token sets, signature‑based Jaccard
  similarity, and an entropy function over arbitrary probability vectors
  (used for infotaxis).

**Mathematical bridge**  
Both parents operate on *probability‑like* quantities and employ information‑
theoretic measures (Shannon entropy).  The fusion therefore:

1. Computes an SSIM‑style similarity `S₁` between two 4‑D morphology vectors.
2. Computes a MinHash‑style similarity `S₂` between two token sets.
3. Blends `S₁` and `S₂` with a weighting `γ`.
4. Builds two entropy terms:
   * `Hₜ` – entropy of categorical token frequencies extracted from free‑form
     logs (Parent A).
   * `Hₛ` – entropy of the distribution of MinHash signature values
     (Parent B).
5. Merges the entropies into `H = (Hₜ + Hₛ)/2`.
6. Incorporates a reconstruction‑risk score `ρ` derived from quasi‑identifier
   statistics (Parent B).
7. Combines the blended similarity, a recovery‑priority average `R`, the
   merged entropy, and the risk into a unified **Hybrid Recovery Score** `Ψ`:


Ψ = (α·S + (1‑α)·R) · (1‑β·H) · (1‑ρ)
where   S = γ·S₁ + (1‑γ)·S₂


The resulting score can drive endpoint circuit‑breakers and guide model‑pool
eviction decisions.

Only the Python standard library and NumPy are used.
"""

import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Decision Hygiene
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def morphology_vector(m: Morphology) -> np.ndarray:
    """Return a 4‑D NumPy column vector for a Morphology instance."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float).reshape(-1, 1)


def ssim_like_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    SSIM‑style similarity for equal‑length vectors.

    Implements the classic structural similarity index using mean,
    variance and covariance:
        S = (2 μ₁ μ₂ + C₁)(2 σ₁₂ + C₂) / ((μ₁²+μ₂²+C₁)(σ₁²+σ₂²+C₂))
    where C₁, C₂ are small stabilisers.
    """
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
    """Extract word‑like tokens from logs and count frequencies."""
    freq: Dict[str, int] = {}
    token_pattern = re.compile(r"\b\w+\b")
    for line in logs:
        for tok in token_pattern.findall(line.lower()):
            freq[tok] = freq.get(tok, 0) + 1
    return freq


def shannon_entropy_from_counts(counts: Dict[str, int], eps: float = 1e-12) -> float:
    """Normalized Shannon entropy given a token‑count dictionary."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = -sum((c / total) * math.log(max(c / total, eps)) for c in counts.values())
    return ent / math.log(len(counts) or 1)  # normalised to [0,1]


# ----------------------------------------------------------------------
# Parent B – MinHash & Infotaxis Entropy
# ----------------------------------------------------------------------


MAX64 = (1 << 64) - 1


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(np.uint64(hash(data) & MAX64).tobytes(), dtype=">u8"), ">u8"
    )


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Compute a MinHash signature of length *k* for the supplied token set."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [MAX64] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability vector (unnormalised accepted)."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError("positive probability mass required")
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)


def signature_entropy(sig: List[int], bins: int = 256) -> float:
    """
    Entropy of the distribution of signature values.
    Values are binned into *bins* equally‑spaced intervals.
    """
    if not sig:
        return 0.0
    hist, _ = np.histogram(sig, bins=bins, range=(0, MAX64))
    probs = hist.astype(float).tolist()
    return entropy(probs)


def reconstruction_risk_score(unique_quasi_ids: int, total_records: int) -> float:
    """
    Simple reconstruction‑risk estimator.
    Returns a value in [0,1] where higher means higher risk.
    """
    if total_records <= 0:
        return 1.0
    ratio = unique_quasi_ids / total_records
    # More unique identifiers → higher risk, capped at 1.
    return min(max(ratio, 0.0), 1.0)


# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------


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
    """
    Compute the unified Hybrid Recovery Score Ψ.

    Parameters
    ----------
    morph_a, morph_b : Morphology
        Endpoint morphology descriptors.
    logs_a, logs_b : Iterable[str]
        Free‑form logs for each endpoint (used for token‑frequency entropy).
    tokens_a, tokens_b : Iterable[str]
        Token sets for MinHash signature generation.
    unique_quasi_ids, total_records : int
        Statistics for reconstruction‑risk estimation.
    recovery_a, recovery_b : float
        Individual recovery priority scores (e.g. 0‑1).
    α, β, γ : float
        Blending factors in [0,1].

    Returns
    -------
    Ψ : float
        The hybrid recovery score; lower values indicate higher failure risk.
    """
    # 1️⃣ Morphology vectors & SSIM‑like similarity
    v1 = morphology_vector(morph_a)
    v2 = morphology_vector(morph_b)
    ssim_sim = ssim_like_similarity(v1, v2)

    # 2️⃣ MinHash signatures & similarity
    sig_a = minhash_signature(tokens_a)
    sig_b = minhash_signature(tokens_b)
    mh_sim = minhash_similarity(sig_a, sig_b)

    # 3️⃣ Blend the two similarity measures
    S = γ * ssim_sim + (1 - γ) * mh_sim

    # 4️⃣ Recovery priority average
    R = (recovery_a + recovery_b) / 2.0

    # 5️⃣ Entropy from logs (decision hygiene)
    freq_a = token_frequencies(logs_a)
    freq_b = token_frequencies(logs_b)
    combined_freq = {}
    for d in (freq_a, freq_b):
        for k, v in d.items():
            combined_freq[k] = combined_freq.get(k, 0) + v
    H_token = shannon_entropy_from_counts(combined_freq)

    # 6️⃣ Entropy of MinHash signatures
    H_sig = signature_entropy(sig_a)  # symmetric, use one side

    H = (H_token + H_sig) / 2.0  # merged entropy term

    # 7️⃣ Reconstruction risk
    ρ = reconstruction_risk_score(unique_quasi_ids, total_records)

    # 8️⃣ Final hybrid score
    ψ = (α * S + (1 - α) * R) * (1 - β * H) * (1 - ρ)
    return max(min(ψ, 1.0), 0.0)  # clamp to [0,1]


def rank_models_by_score(
    models: List[ModelTier],
    scores: List[float],
) -> List[Tuple[ModelTier, float]]:
    """
    Return models sorted descending by their associated scores.
    """
    paired = list(zip(models, scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired


def simulate_hybrid_scenario() -> Dict[str, Any]:
    """
    Generate random data for two endpoints, compute the hybrid score,
    and produce a ranking of three candidate model tiers based on that score.
    """
    # Random morphologies
    morph_a = Morphology(
        length=random.uniform(1.0, 10.0),
        width=random.uniform(1.0, 5.0),
        height=random.uniform(0.5, 3.0),
        mass=random.uniform(10.0, 100.0),
    )
    morph_b = Morphology(
        length=random.uniform(1.0, 10.0),
        width=random.uniform(1.0, 5.0),
        height=random.uniform(0.5, 3.0),
        mass=random.uniform(10.0, 100.0),
    )

    # Synthetic logs
    logs_a = [
        "INFO start processing request",
        "WARN latency high",
        "ERROR failed to acquire lock",
    ]
    logs_b = [
        "INFO request received",
        "INFO processing completed",
        "DEBUG cache miss",
    ]

    # Token sets (could be extracted from logs or elsewhere)
    tokens_a = ["start", "process", "request", "latency", "error"]
    tokens_b = ["receive", "process", "complete", "cache", "miss"]

    # Recovery priorities (simulated)
    recovery_a = random.random()
    recovery_b = random.random()

    # Quasi‑identifier statistics
    unique_quasi_ids = random.randint(10, 100)
    total_records = random.randint(100, 500)

    ψ = hybrid_recovery_score(
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
        α=0.5,
        β=0.3,
        γ=0.6,
    )

    # Three candidate models
    candidates = [
        ModelTier("qwen-0.5b", 512, "T1"),
        ModelTier("reasoning-t2", 3000, "T2"),
        ModelTier("qwen-7b", 7000, "T3"),
    ]

    # Derive a simplistic model score from the hybrid score (higher ψ → higher model utility)
    model_scores = [ψ * (1.0 - i * 0.1) for i in range(len(candidates))]

    ranked = rank_models_by_score(candidates, model_scores)

    return {
        "hybrid_score": ψ,
        "recovery_priorities": (recovery_a, recovery_b),
        "model_ranking": [(m.name, s) for m, s in ranked],
    }


if __name__ == "__main__":
    result = simulate_hybrid_scenario()
    print("Hybrid Recovery Score (Ψ):", result["hybrid_score"])
    print("Recovery Priorities (A, B):", result["recovery_priorities"])
    print("Model ranking (best → worst):")
    for name, score in result["model_ranking"]:
        print(f"  {name}: {score:.4f}")