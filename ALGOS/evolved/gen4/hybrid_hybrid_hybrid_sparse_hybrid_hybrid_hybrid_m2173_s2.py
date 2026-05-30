# DARWIN HAMMER — match 2173, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py (gen3)
# born: 2026-05-29T23:41:05Z

"""
Hybrid Sparse-WTA & Bayesian Decision Module

Parents:
- hybrid_hybrid_sparse_wta_hy_hybrid_capybara_opti_m180_s0.py
- hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py

Mathematical Bridge:
The bridge is the *confidence scalar* C, defined as the product of
1) the Shannon entropy H of textual evidence features (parent B) and
2) a signal‑to‑noise ratio S derived from the numeric input vector (parent A):
    C = H * S

C rescales:
* the random sign coefficient in the hash‑based sparse expansion,
* the step size in the exponential evasion schedule,
* the prior probability in the Bayesian marginal update.

Thus the hybrid algorithm intertwines the sparse projection and top‑k
winner‑take‑all mechanism of parent A with the Bayesian update and
entropy‑based prior of parent B, yielding a unified system that adapts
its sparsity and inference strength to the information content of the
input data."""
import hashlib
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def expand(values: List[float], m: int, salt: str = "", confidence: float = 1.0) -> List[float]:
    """Hash‑based sparse expansion of `values` into a vector of length `m`.

    Each hash contribution is multiplied by `confidence` to modulate the
    magnitude of the random sign, implementing the bridge described above.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v * confidence
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    """Binary mask with 1 at the indices of the top‑k values."""
    k = max(0, min(k, len(values)))
    winners = {
        i
        for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def extract_features(text: str) -> Dict[str, int]:
    """Regex‑based extraction of evidence‑related tokens."""
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked)\b",
        flags=re.IGNORECASE,
    )
    features = defaultdict(int)
    for match in evidence_re.finditer(text):
        token = match.group().lower()
        features[token] += 1
    return dict(features)


def shannon_entropy(features: Dict[str, int]) -> float:
    """Shannon entropy of a discrete feature distribution."""
    total = sum(features.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in features.values()]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def signal_to_noise(values: List[float]) -> float:
    """Simple signal‑to‑noise ratio: (max‑mean)/std, clipped to [0, ∞)."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std == 0.0:
        return 0.0
    return max(0.0, (arr.max() - mean) / std)


def confidence_scalar(values: List[float], text: str) -> float:
    """Combined confidence C = H * S."""
    H = shannon_entropy(extract_features(text))
    S = signal_to_noise(values)
    return H * S


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = likelihood·prior + false_positive·(1‑prior)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def hybrid_wta_bayes(
    values: List[float],
    m: int,
    text: str,
    k: int,
    salt: str = "",
    likelihood: float = 0.9,
    false_positive: float = 0.1,
) -> Tuple[List[float], List[int], float]:
    """Perform a hybrid operation:

    1. Compute confidence C.
    2. Expand `values` sparsely with scaling by C.
    3. Apply top‑k mask to obtain a sparse winner‑take‑all vector.
    4. Use C as the Bayesian prior to compute a marginal probability.

    Returns (expanded_vector, mask, marginal).
    """
    C = confidence_scalar(values, text)
    expanded = expand(values, m, salt=salt, confidence=C)
    mask = top_k_mask(expanded, k)
    marginal = bayes_marginal(prior=min(C, 1.0), likelihood=likelihood, false_positive=false_positive)
    return expanded, mask, marginal


def evasion_delta(t: int, t_max: int, confidence: float, delta_max: float = 1.0, alpha: float = 3.0) -> float:
    """Exponential decay schedule modulated by confidence.

    The base schedule is delta_max * exp(-alpha * t / t_max).
    Confidence stretches the schedule: larger C slows decay.
    """
    if t < 0 or t_max <= 0:
        raise ValueError("Invalid time parameters")
    base = delta_max * math.exp(-alpha * t / t_max)
    return base * (1.0 + confidence)  # confidence ≥ 0 expands the magnitude


def hybrid_process(
    numeric_input: List[float],
    text_input: str,
    m: int = 128,
    k: int = 10,
    salt: str = "hybrid",
    t_max: int = 100,
) -> Dict[str, Any]:
    """End‑to‑end hybrid pipeline.

    - Computes confidence.
    - Runs the hybrid WTA‑Bayes step.
    - Generates an evasion schedule over `t_max` steps.
    Returns a dictionary with all intermediate results.
    """
    C = confidence_scalar(numeric_input, text_input)
    expanded, mask, marginal = hybrid_wta_bayes(
        numeric_input, m, text_input, k, salt=salt
    )
    evasion_series = [evasion_delta(t, t_max, C) for t in range(t_max + 1)]
    return {
        "confidence": C,
        "expanded_vector": expanded,
        "top_k_mask": mask,
        "bayesian_marginal": marginal,
        "evasion_series": evasion_series,
    }


if __name__ == "__main__":
    # Simple smoke test
    numeric = [random.uniform(-1, 1) for _ in range(20)]
    text = (
        "The evidence was verified and the source was confirmed. "
        "A screenshot and log record were attached as proof."
    )
    result = hybrid_process(numeric, text)
    # Print summary statistics to ensure execution
    print("Confidence:", result["confidence"])
    print("Expanded vector norm:", np.linalg.norm(result["expanded_vector"]))
    print("Top‑k mask sum:", sum(result["top_k_mask"]))
    print("Bayesian marginal:", result["bayesian_marginal"])
    print("Evasion series first 5 values:", result["evasion_series"][:5])