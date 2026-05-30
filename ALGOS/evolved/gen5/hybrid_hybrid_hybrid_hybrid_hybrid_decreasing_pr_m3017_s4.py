# DARWIN HAMMER — match 3017, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_ternary_router_m906_s3.py (gen4)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_bandit_m365_s0.py (gen3)
# born: 2026-05-29T23:47:15Z

"""Hybrid Fusion Module: hybrid_hard_truth_math_model + ternary_router + decreasing_pruning

Parents:
* **Parent A** – high‑dimensional stylometric feature extraction, bilinear projection,
  reliability‑curvature scalar γ = ρ·κ and ternary routing by ‖s‖₂.
* **Parent B** – temperature‑dependent developmental rate (exponential) and
  exponential pruning probability.

Mathematical Bridge
------------------
Both parents rely on scalar factors that modulate a linear algebraic core.
We embed the *developmental rate* 𝜏(T) from Parent B as a temperature‑dependent
multiplier of the reliability‑curvature scalar of Parent A, obtaining

    γ′ = ρ · κ · 𝜏(T)

The fused signature becomes

    s = γ′ · (fᵀ·W)

Finally the exponential pruning probability of Parent B is used to sparsify
the signature vector element‑wise, yielding a pruned signature ŝ that is then
routed to one of three channels by its L2‑norm.

The implementation below follows this formulation and provides a self‑contained
pipeline."""
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Simplified linguistic resources (subset of Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": {
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
        "he", "him", "his", "she", "her", "hers", "they", "them", "their",
        "theirs", "we", "us", "our", "ours"
    },
    "article": {"a", "an", "the"},
    "preposition": {
        "about", "above", "after", "against", "around", "as", "at", "before",
        "behind", "below", "between", "by", "during", "for", "from", "in",
        "into", "of", "off", "on", "onto", "over", "through", "to", "under",
        "with", "without"
    },
    "auxiliary": {
        "am", "are", "be", "been", "being", "can", "could", "did", "do",
        "does", "had", "has", "have", "is", "may", "might", "must", "shall",
        "should", "was", "were", "will", "would"
    },
}

# ----------------------------------------------------------------------
# Parent B – Schoolfield developmental rate (temperature dependent)
# ----------------------------------------------------------------------
class SchoolfieldParams:
    """Immutable parameters for the Schoolfield model."""
    def __init__(
        self,
        rho_25: float = 1.0,
        delta_h_activation: float = 12_000.0,
        t_low: float = 283.15,
        t_high: float = 307.15,
        delta_h_low: float = -45_000.0,
        delta_h_high: float = 65_000.0,
        r_cal: float = 1.987,
    ):
        self.rho_25 = rho_25
        self.delta_h_activation = delta_h_activation
        self.t_low = t_low
        self.t_high = t_high
        self.delta_h_low = delta_h_low
        self.delta_h_high = delta_h_high
        self.r_cal = r_cal

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Temperature‑dependent developmental rate (Parent B)."""
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")
    num = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp(
        (params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k))
    )
    high = math.exp(
        (params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k))
    )
    den = 1.0 + low + high
    return num / den

def prune_probability(t: float, lam: float, alpha: float, temp_k: float) -> float:
    """Exponential pruning probability (Parent B)."""
    rate = developmental_rate(temp_k)
    return min(1.0, lam * math.exp(-alpha * t * rate))

# ----------------------------------------------------------------------
# Core hybrid operations (fusion of Parent A & B)
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Very lightweight stylometric extractor.
    Returns a vector f ∈ ℝⁿ where n = total number of distinct tokens across
    all FUNCTION_CATS.
    """
    tokens = [tok.lower().strip(".,!?;:") for tok in text.split()]
    categories = list(FUNCTION_CATS.keys())
    vocab = sorted({w for cat in categories for w in FUNCTION_CATS[cat]})
    idx = {word: i for i, word in enumerate(vocab)}
    f = np.zeros(len(vocab), dtype=float)
    for tok in tokens:
        if tok in idx:
            f[idx[tok]] += 1.0
    return f

def bilinear_project(f: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Bilinear projection p = fᵀ·W (Parent A).
    f : (n,)
    W : (n, m)
    Returns p : (m,)
    """
    if f.ndim != 1 or W.ndim != 2 or f.shape[0] != W.shape[0]:
        raise ValueError("Shape mismatch in bilinear projection")
    return f @ W

def compute_reliability_score(meta: Dict[str, Any]) -> float:
    """
    Derive a reliability scalar ρ from packet meta‑information.
    For demonstration we map a field ``'confidence'`` in [0,1] to ρ,
    defaulting to 0.5 if absent.
    """
    conf = float(meta.get("confidence", 0.5))
    return max(0.0, min(1.0, conf))

def compute_curvature(p: np.ndarray) -> float:
    """Curvature κ defined as the variance of the projected vector."""
    if p.size == 0:
        return 0.0
    return float(np.var(p))

def prune_vector(v: np.ndarray, t: float, lam: float, alpha: float, temp_k: float, rng: random.Random) -> np.ndarray:
    """
    Apply element‑wise exponential pruning (Parent B) to a vector.
    Each component is kept with probability 1‑prune_probability(...).
    """
    prob = prune_probability(t, lam, alpha, temp_k)
    mask = np.array([rng.random() > prob for _ in range(v.size)], dtype=bool)
    pruned = np.where(mask, v, 0.0)
    return pruned

def fuse_and_route(
    text: str,
    meta: Dict[str, Any],
    W: np.ndarray,
    temp_c: float,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    thresholds: Sequence[float] = (5.0, 15.0),
    rng_seed: int | str | None = None,
) -> int:
    """
    Full hybrid pipeline:

    1. Extract high‑dimensional stylometric vector f.
    2. Bilinear projection p = fᵀ·W.
    3. Compute reliability ρ and curvature κ.
    4. Obtain temperature‑dependent developmental rate τ(T).
    5. Form fused scalar γ′ = ρ·κ·τ(T) and signature s = γ′·p.
    6. Sparsify s via exponential pruning → ŝ.
    7. Route to one of three channels based on ‖ŝ‖₂ and supplied thresholds.

    Returns the channel index (0, 1, 2).
    """
    # 1‑2
    f = extract_features(text)
    p = bilinear_project(f, W)

    # 3
    rho = compute_reliability_score(meta)
    kappa = compute_curvature(p)

    # 4‑5
    temp_k = c_to_k(temp_c)
    tau = developmental_rate(temp_k)
    gamma_prime = rho * kappa * tau
    s = gamma_prime * p

    # 6
    rng = random.Random(rng_seed)
    s_pruned = prune_vector(s, t, lam, alpha, temp_k, rng)

    # 7 – ternary routing
    norm = float(np.linalg.norm(s_pruned, ord=2))
    low_thr, high_thr = thresholds
    if norm < low_thr:
        channel = 0
    elif norm < high_thr:
        channel = 1
    else:
        channel = 2
    return channel

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy weight matrix: n = vocab size, m = 8 low‑dimensional channels
    dummy_text = "I have a belief that the quick brown fox jumps over the lazy dog."
    vocab = sorted({w for cat in FUNCTION_CATS.values() for w in cat})
    n = len(vocab)
    m = 8
    rng = np.random.default_rng(42)
    W = rng.normal(loc=0.0, scale=0.1, size=(n, m))

    meta_example = {"confidence": 0.78}
    channel = fuse_and_route(
        text=dummy_text,
        meta=meta_example,
        W=W,
        temp_c=22.0,          # ambient temperature in Celsius
        t=3.5,                # abstract time step for pruning
        lam=0.9,
        alpha=0.15,
        thresholds=(2.0, 8.0),
        rng_seed=12345,
    )
    print(f"Routed to channel {channel}")