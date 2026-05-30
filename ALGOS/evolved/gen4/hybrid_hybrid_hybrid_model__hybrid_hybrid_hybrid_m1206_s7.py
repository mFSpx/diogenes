# DARWIN HAMMER — match 1206, survivor 7
# gen: 4
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_ssim_doomsday_m214_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s1.py (gen3)
# born: 2026-05-29T23:34:36Z

"""Hybrid VRAM‑Signal‑Similarity & Entropic Model‑Pool Algorithm
================================================================
Parent A: hybrid_hybrid_model_vram_sc_hybrid_ssim_doomsday_m214_s0.py  
Parent B: hybrid_hybrid_hybrid_privac_hybrid_infotaxis_min_m106_s1.py  

Mathematical Bridge
-------------------
Both parents manipulate *resource‑related signals* and *similarity measures*:

* Parent A treats the time‑series of free VRAM as a signal and compares it to a
  periodic doomsday signal using the Structural Similarity Index (SSIM).
* Parent B manages a pool of models (each consuming RAM) and evaluates
  similarity between model signatures with MinHash‑based Jaccard estimates,
  weighting the decision by an entropy‑derived information gain and a
  reconstruction‑risk score.

The fusion therefore builds a **single similarity score** that
1. measures structural similarity between a VRAM‑usage signal and a
   synthetic periodic signal (SSIM), and
2. modulates that score by an *information‑theoretic* factor derived from
   the loaded model pool (entropy of MinHash signatures) and a *risk‑aware*
   factor (reconstruction risk of the loaded models).

The final hybrid score 𝛴 is:


𝛴 = λ·SSIM(vram, periodic) ×
    (1 – ρ̄) ×
    (1 – Ĵ) ×
    Ĥ


where  
λ – user‑tunable weight,  
ρ̄ – average reconstruction risk of the pool,  
Ĵ – average MinHash Jaccard estimate between the VRAM signal signature and
     each model’s signature,  
Ĥ – normalized Shannon entropy of the signature distribution.

The implementation below provides three core functions that realise this
fusion, together with a lightweight model‑pool manager compatible with the
original Parent B design.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ----------------------------------------------------------------------
# Model‑pool utilities (derived from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Immutable description of a model tier."""
    name: str
    ram_mb: int
    tier: str

# Example tiers (could be extended by the user)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

def reconstruction_risk_score(model: ModelTier) -> float:
    """
    Simple risk proxy: larger RAM footprints imply higher reconstruction risk.
    Normalised to the interval [0, 1] using a soft‑sigmoid.
    """
    # Parameters chosen to map typical RAM sizes (0‑8000 MB) into (0‑1)
    scale = 0.001
    return 1.0 / (1.0 + math.exp(-scale * (model.ram_mb - 4000)))

class ModelPool:
    """Manages a bounded set of loaded models, respecting tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model, raising if constraints are violated."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 models cannot coexist with any T2 model.")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded.")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting the oldest entries until enough space is free."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)

    def models(self) -> List[ModelTier]:
        return list(self.loaded.values())

# ----------------------------------------------------------------------
# Signal generation (Parent A)
# ----------------------------------------------------------------------
def generate_vram_signal(duration_s: int, interval_s: int,
                         budget_mb: int = 4096,
                         reserve_mb: int = 768) -> np.ndarray:
    """
    Simulate free VRAM over time as a stochastic process.
    The signal fluctuates around (budget - reserve) with a small random walk.
    """
    steps = max(1, duration_s // interval_s)
    base = budget_mb - reserve_mb
    signal = [base]
    for _ in range(steps - 1):
        delta = random.gauss(0, 0.05 * base)  # 5 % Gaussian noise
        new_val = max(0, min(budget_mb, signal[-1] + delta))
        signal.append(new_val)
    return np.array(signal, dtype=np.float64)

def generate_periodic_signal(duration_s: int, interval_s: int,
                             period_s: int = 86400) -> np.ndarray:
    """
    Produce a simple sinusoidal signal with the given period.
    """
    steps = max(1, duration_s // interval_s)
    t = np.arange(steps) * interval_s
    # Normalised to the same scale as VRAM (0‑budget)
    amplitude = 0.5
    offset = 0.5
    signal = amplitude * np.sin(2 * math.pi * t / period_s) + offset
    # Scale to a typical VRAM range (0‑budget)
    return signal * 4096

# ----------------------------------------------------------------------
# Similarity & information‑theoretic primitives
# ----------------------------------------------------------------------
def ssim_1d(x: np.ndarray, y: np.ndarray,
            C1: float = 0.01 * 4096 ** 2,
            C2: float = 0.03 * 4096 ** 2) -> float:
    """
    One‑dimensional Structural Similarity Index (SSIM) for two equal‑length signals.
    The constants C1, C2 are scaled to the VRAM magnitude.
    """
    if x.shape != y.shape:
        raise ValueError("Signals must have identical shape.")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

def minhash_signature(data: List[int], num_perm: int = 64) -> List[int]:
    """
    Very lightweight MinHash: for each permutation we hash the data with a
    random linear function (a·x + b) mod prime and keep the minimum.
    """
    prime = (1 << 31) - 1  # a large Mersenne prime
    max_hash = prime
    signatures = []
    for _ in range(num_perm):
        a = random.randint(1, prime - 1)
        b = random.randint(0, prime - 1)
        min_val = max_hash
        for v in data:
            hv = (a * v + b) % prime
            if hv < min_val:
                min_val = hv
        signatures.append(min_val)
    return signatures

def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signature lengths must match.")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def entropy_of_signature(sig: List[int]) -> float:
    """
    Compute the Shannon entropy of a MinHash signature interpreted as a
    categorical distribution over its integer values.
    """
    if not sig:
        return 0.0
    counts: Dict[int, int] = {}
    for v in sig:
        counts[v] = counts.get(v, 0) + 1
    total = len(sig)
    ent = -sum((c / total) * math.log(c / total, 2) for c in counts.values())
    # Normalise by the maximum possible entropy (log2(num_unique))
    max_ent = math.log(len(counts), 2) if len(counts) > 1 else 0.0
    return ent / max_ent if max_ent > 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_similarity(vram_signal: np.ndarray,
                      periodic_signal: np.ndarray,
                      pool: ModelPool,
                      weight: float = 1.0) -> Dict[str, float]:
    """
    Compute the fused similarity score 𝛴.

    Returns a dictionary with the intermediate components and the final score.
    """
    # 1️⃣ SSIM between the two signals
    ssim_val = ssim_1d(vram_signal, periodic_signal)

    # 2️⃣ Convert the VRAM signal to a discrete integer list for MinHash
    vram_ints = (vram_signal / vram_signal.max() * 1e6).astype(int).tolist()
    vram_sig = minhash_signature(vram_ints)

    # 3️⃣ Aggregate model‑level statistics
    if not pool.models():
        # No models loaded → fall back to pure SSIM
        return {"ssim": ssim_val, "final_score": weight * ssim_val}

    risk_vals = []
    jaccard_vals = []
    entropy_vals = []

    for model in pool.models():
        # Risk
        risk = reconstruction_risk_score(model)
        risk_vals.append(risk)

        # Model signature (hash of model name characters)
        name_ints = [ord(ch) for ch in model.name]
        model_sig = minhash_signature(name_ints)
        jaccard = jaccard_estimate(vram_sig, model_sig)
        jaccard_vals.append(jaccard)

        # Entropy of the model signature
        entropy_vals.append(entropy_of_signature(model_sig))

    avg_risk = sum(risk_vals) / len(risk_vals)          # ρ̄
    avg_jaccard = sum(jaccard_vals) / len(jaccard_vals)  # Ĵ
    avg_entropy = sum(entropy_vals) / len(entropy_vals)  # Ĥ

    # 4️⃣ Fuse according to the bridge equation
    fused = weight * ssim_val * (1.0 - avg_risk) * (1.0 - avg_jaccard) * avg_entropy

    return {
        "ssim": ssim_val,
        "avg_risk": avg_risk,
        "avg_jaccard": avg_jaccard,
        "avg_entropy": avg_entropy,
        "final_score": fused,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a model pool and load a couple of models
    pool = ModelPool(ram_ceiling_mb=8000)
    pool.load_with_eviction(TIER_T1_QWEN_0_5B)
    pool.load_with_eviction(TIER_T2_REASONING)

    # Generate signals
    duration = 3600          # 1 hour
    interval = 60            # 1 minute samples
    vram_sig = generate_vram_signal(duration, interval)
    periodic_sig = generate_periodic_signal(duration, interval, period_s=86400)

    # Compute hybrid similarity
    result = hybrid_similarity(vram_sig, periodic_sig, pool, weight=1.0)

    # Simple sanity printout
    for k, v in result.items():
        print(f"{k}: {v:.6f}")
    sys.exit(0)