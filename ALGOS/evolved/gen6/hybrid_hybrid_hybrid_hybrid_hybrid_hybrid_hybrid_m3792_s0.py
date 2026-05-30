# DARWIN HAMMER — match 3792, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2716_s6.py (gen5)
# born: 2026-05-29T23:51:36Z

"""Hybrid Stylometry‑Geometric‑Certainty (HSGC) algorithm.

Parents:
- hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s2.py (stylometry analysis,
  geometric product calculations, vector‑point operations)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m2716_s6.py (certainty flag,
  2‑D geometric‑algebra rotor, Fisher‑SSIM decision metric)

Mathematical bridge:
Both parents operate on vectors and employ a geometric transformation.
The stylometry vector `s ∈ ℝⁿ` (parent A) is first reduced to a 2‑D subspace
by selecting its first two components `s₂ = (s₀, s₁)`.  A certainty‑scaled
rotor `R(w_c) = exp(-½ w_c π e₁e₂)` (parent B) is a 2‑D rotation matrix that
rotates `s₂`.  The rotated stylometry vectors are then fed into the
certainty‑weighted decision metric

    M(t) = p(t) · [ w_f·SSIM(r₁, r₂) + w_h·H·Σ_i w_i·f_i ] · (1 + w_c),

where `p(t)=1/(1+t)` is the pruning probability, `w_f` and `w_h` are
Fisher‑information‑derived and entropy‑derived scalar weights, `H` is the
Shannon entropy of the combined stylometry multiset, and `f_i` are the raw
category counts.  This single formulation fuses the textual‑style analysis
of parent A with the certainty‑driven geometric‑algebra routing of parent B.
"""

import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Stylometry utilities
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
}

def _tokenize(text: str) -> List[str]:
    """Very light tokenizer – split on whitespace and strip punctuation."""
    return [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]

def stylometry_vector(text: str) -> np.ndarray:
    """
    Compute a normalized frequency vector for the four FUNCTION_CATS.
    Returns a 4‑element float array whose entries sum to 1 (or 0 if no tokens).
    """
    tokens = _tokenize(text)
    total = len(tokens) or 1
    counts = []
    for cat in ("pronoun", "article", "preposition", "auxiliary"):
        cat_set = FUNCTION_CATS[cat]
        cnt = sum(1 for t in tokens if t in cat_set)
        counts.append(cnt / total)
    return np.array(counts, dtype=float)

# ----------------------------------------------------------------------
# Parent B – Certainty & geometric‑algebra helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for epistemic certainty."""
    label: str

    @property
    def weight(self) -> float:
        """Map label to a scalar w_c ∈ [0,1]."""
        mapping = {
            "FACT": 1.0,
            "PROBABLE": 0.8,
            "POSSIBLE": 0.5,
            "SURE_MAYBE": 0.2,
            "BULLSHIT": 0.0,
        }
        return mapping.get(self.label.upper(), 0.0)

def rotor_matrix(w_c: float) -> np.ndarray:
    """
    2‑D rotation matrix derived from the GA rotor
        R(w_c) = exp(-½ w_c π e₁e₂)
    which corresponds to a rotation by angle θ = -½ π w_c.
    """
    theta = -0.5 * math.pi * w_c
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def fisher_weight(vec: np.ndarray) -> float:
    """
    Approximate Fisher information weight as the coefficient of variation
    (std/mean) of the vector components, normalized to [0,1].
    """
    mean = np.mean(vec)
    if mean == 0:
        return 0.0
    std = np.std(vec)
    w = std / mean
    return max(0.0, min(1.0, w))

def shannon_entropy(vec: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    eps = np.finfo(float).eps
    probs = np.clip(vec, eps, 1.0)
    probs /= probs.sum()
    return -np.sum(probs * np.log2(probs))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def rotate_stylometry(s_vec: np.ndarray, w_c: float) -> np.ndarray:
    """
    Reduce the 4‑D stylometry vector to its first two components,
    rotate them with the certainty‑scaled rotor, and embed back into
    a 4‑D space (rotated components replace the original ones).
    """
    if s_vec.shape != (4,):
        raise ValueError("stylometry vector must be length 4")
    sub = s_vec[:2]                     # (pronoun, article)
    R = rotor_matrix(w_c)
    rotated = R @ sub
    result = s_vec.copy()
    result[:2] = rotated
    return result

def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """
    A lightweight Structural Similarity (SSIM) surrogate for 1‑D vectors:
        SSIM = (2μₐμ_b + C₁)(2σ_ab + C₂) / ((μₐ²+μ_b²+C₁)(σₐ²+σ_b²+C₂))
    where C₁,C₂ are small constants to avoid division by zero.
    """
    C1, C2 = 1e-4, 1e-4
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2, sigma_b2 = a.var(), b.var()
    sigma_ab = np.cov(a, b, bias=True)[0, 1]
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return numerator / denominator if denominator != 0 else 0.0

def hybrid_metric(
    text_a: str,
    text_b: str,
    certainty: CertaintyFlag,
    t: float = 0.0
) -> float:
    """
    Unified decision metric M(t) that fuses stylometry, geometric rotation,
    Fisher information, and Shannon entropy.

    Steps
    -----
    1. Compute raw stylometry vectors `s_a`, `s_b`.
    2. Apply certainty‑scaled rotation to the first two components.
    3. Derive Fisher weight `w_f` from the concatenated rotated vectors.
    4. Compute Shannon entropy `H` of the combined category counts.
    5. Compute an SSIM‑like similarity `S` between the rotated vectors.
    6. Assemble the metric:
           p(t) = 1/(1+t)
           M(t) = p(t) * [ w_f·S + w_h·H·Σ_i w_i·f_i ] * (1 + w_c)
    """
    w_c = certainty.weight

    # 1. Stylometry
    s_a = stylometry_vector(text_a)
    s_b = stylometry_vector(text_b)

    # 2. Rotate
    r_a = rotate_stylometry(s_a, w_c)
    r_b = rotate_stylometry(s_b, w_c)

    # 3. Fisher weight from concatenated rotated vectors
    concat = np.concatenate([r_a, r_b])
    w_f = fisher_weight(concat)

    # 4. Entropy and raw counts
    # raw counts are the un‑normalized token category frequencies multiplied by token count
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    total_tokens = len(tokens_a) + len(tokens_b) or 1

    raw_counts = np.zeros(4, dtype=float)
    for idx, cat in enumerate(("pronoun", "article", "preposition", "auxiliary")):
        cat_set = FUNCTION_CATS[cat]
        cnt = sum(1 for t in tokens_a + tokens_b if t in cat_set)
        raw_counts[idx] = cnt

    probs = raw_counts / total_tokens
    H = shannon_entropy(probs)

    # w_h is a normalized entropy weight (entropy divided by log2(num_categories))
    max_entropy = math.log2(len(probs))
    w_h = H / max_entropy if max_entropy != 0 else 0.0

    # Σ_i w_i·f_i  – we use raw_counts as f_i and weight each by its probability
    weighted_sum = float(np.dot(probs, raw_counts))

    # 5. SSIM‑like similarity on rotated vectors
    S = ssim_like(r_a, r_b)

    # 6. Assemble metric
    p_t = 1.0 / (1.0 + t)
    M = p_t * (w_f * S + w_h * H * weighted_sum) * (1.0 + w_c)
    return M

# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    txt1 = "I think we should go to the market because it is near the river."
    txt2 = "You might consider visiting the museum; it has many exhibits."
    flag = CertaintyFlag("PROBABLE")
    score = hybrid_metric(txt1, txt2, flag, t=0.3)
    print(f"Hybrid metric score (certainty={flag.label}): {score:.6f}")
    # Ensure no exception for edge cases
    empty_score = hybrid_metric("", "", CertaintyFlag("BULLSHIT"))
    print(f"Score for empty inputs: {empty_score:.6f}")