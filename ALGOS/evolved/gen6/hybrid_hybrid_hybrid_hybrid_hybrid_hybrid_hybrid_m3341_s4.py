# DARWIN HAMMER — match 3341, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m886_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_korpus_text_h_m537_s0.py (gen5)
# born: 2026-05-29T23:49:22Z

"""hybrid_hybrid_hybrid_fusion_m1_s0.py
Hybrid Fusion of:
- Parent Algorithm A: TTT‑Linear weight matrix, Count‑Min sketch, reconstruction‑risk ratio.
- Parent Algorithm B: Morphology‑based recovery priority, regex‑derived textual feature vector,
  positive/negative weight matrices, and entropy scoring.

Mathematical Bridge
------------------
Algorithm A provides a dense weight matrix **W** ∈ ℝ^{d_out×d_in} (the TTT‑Linear matrix) that
is continuously updated by gradient descent on the *reconstruction‑risk ratio* ρ.
Algorithm B requires two weight matrices **W⁺**, **W⁻** to map a 9‑dimensional text
feature vector **v** to a score vector **s** = W⁺·v − W⁻·v.

The fusion identifies **W** as the source of both **W⁺** and **W⁻**:

W⁺ = max(W, 0)          # element‑wise positive part
W⁻ = -min(W, 0)         # element‑wise magnitude of negative part

Thus the TTT‑Linear matrix supplies the linear scoring core of the text engine.

The *reconstruction‑risk ratio* ρ (from A) quantifies how much the sketch‑based
privacy projection deviates from the original feature distribution.
We use ρ to modulate the morphology‑derived recovery priority **p** (from B) and
the final hybrid decision vector **ŝ**:


p̂ = p * (1 - ρ)                 # lower risk → higher effective priority
ŝ = p̂ * (W⁺·v - W⁻·v)           # morphology‑aware, privacy‑aware score


The hybrid system therefore fuses the linear algebraic core of A with the
morphology‑aware scaling of B, while the sketch updates close the loop by
producing the risk term that feeds back into the priority scaling.

The module implements:
1. TTT‑Linear matrix creation and gradient‑descent update.
2. Simple Count‑Min sketch with hashed quasi‑identifier insertion.
3. Morphology‑based priority computation.
4. Regex‑driven 9‑dimensional textual feature extraction.
5. Hybrid scoring that combines the above via the mathematical bridge.
"""

import sys
import math
import random
import hashlib
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent A – TTT‑Linear & Count‑Min Sketch utilities
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Create a TTT‑Linear weight matrix W ∈ ℝ^{d_out×d_in}."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_forward(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear forward pass using the TTT‑Linear matrix."""
    return W @ x


def reconstruction_risk(sketch: np.ndarray, hashes: Iterable[int], original_sum: float) -> float:
    """
    Compute a simple reconstruction‑risk ratio ρ.
    Sketch entries indexed by `hashes` approximate the count of a feature.
    ρ = |sketch_estimate - original| / (original + ε)
    """
    eps = 1e-12
    estimate = np.mean([sketch[h] for h in hashes])
    return abs(estimate - original_sum) / (original_sum + eps)


def gradient_descent_step(W: np.ndarray, grad: np.ndarray, lr: float = 0.001) -> np.ndarray:
    """One step of gradient descent on the weight matrix."""
    return W - lr * grad


def init_sketch(width: int = 1024, depth: int = 4, seed: int = 0) -> np.ndarray:
    """Initialize a Count‑Min sketch (depth × width) with zeros."""
    rng = np.random.default_rng(seed)
    return np.zeros((depth, width), dtype=np.float64)


def sketch_hashes(item: str, depth: int, width: int) -> List[int]:
    """Generate `depth` hash indices for the given item."""
    base = hashlib.blake2b(item.encode(), digest_size=8).digest()
    # Derive deterministic seeds from the base hash
    seeds = [int.from_bytes(base[i:i+2], "little") for i in range(0, 2 * depth, 2)]
    return [(seed % width) for seed in seeds]


def sketch_insert(sketch: np.ndarray, item: str, increment: float = 1.0) -> None:
    """Insert `item` into the Count‑Min sketch."""
    depth, width = sketch.shape
    indices = sketch_hashes(item, depth, width)
    for d, idx in enumerate(indices):
        sketch[d, idx] += increment


# ----------------------------------------------------------------------
# Parent B – Morphology priority & textual feature extraction
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Map righting‑time index to a priority p ∈ [0,1]."""
    rti = righting_time_index(m)
    return min(1.0, rti / max_index)


def extract_text_features(text: str) -> np.ndarray:
    """
    Produce a 9‑dimensional feature vector from regex counts.
    The patterns are deliberately simple for demonstration.
    """
    patterns = [
        r"\bthe\b", r"\band\b", r"\bof\b", r"\bto\b", r"\b[aA]nd\b",
        r"\d+", r"[A-Z][a-z]+", r"\s+", r"[^\w\s]"
    ]
    counts = [len(re.findall(p, text)) for p in patterns]
    return np.array(counts, dtype=np.float64)


def entropy_of_vector(v: np.ndarray) -> float:
    """Shannon entropy of a non‑negative vector (treated as a distribution)."""
    eps = 1e-12
    v = np.maximum(v, 0.0) + eps
    prob = v / v.sum()
    return -np.sum(prob * np.log2(prob))


# ----------------------------------------------------------------------
# Hybrid operations – the mathematical bridge
# ----------------------------------------------------------------------
def split_weights(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive positive and negative weight matrices from the TTT‑Linear matrix.
    W⁺ = max(W, 0)
    W⁻ = -min(W, 0)
    """
    W_pos = np.maximum(W, 0.0)
    W_neg = -np.minimum(W, 0.0)
    return W_pos, W_neg


def hybrid_score(text: str,
                 morph: Morphology,
                 W: np.ndarray,
                 sketch: np.ndarray,
                 sketch_items: List[str] | None = None) -> Tuple[np.ndarray, float]:
    """
    Compute the hybrid decision vector ŝ and its entropy.

    Steps:
    1. Extract 9‑dimensional textual features v.
    2. Split TTT matrix into W⁺, W⁻.
    3. Linear score s = W⁺·v - W⁻·v.
    4. Compute morphology‑derived priority p.
    5. If sketch_items are supplied, insert them into the sketch and compute
       a reconstruction‑risk ratio ρ; otherwise ρ = 0.
    6. Effective priority p̂ = p * (1 - ρ).
    7. Hybrid decision ŝ = p̂ * s.
    8. Return ŝ and its Shannon entropy.
    """
    v = extract_text_features(text)                     # (9,)
    W_pos, W_neg = split_weights(W)                    # both (d_out, 9)
    # Ensure dimensional compatibility
    if W_pos.shape[1] != v.shape[0]:
        raise ValueError("Weight matrix column dimension must match feature length")
    s = (W_pos @ v) - (W_neg @ v)                       # (d_out,)

    p = recovery_priority(morph)                       # scalar ∈ [0,1]

    # Reconstruction‑risk ratio via sketch (optional)
    if sketch_items:
        for item in sketch_items:
            sketch_insert(sketch, item)
        # Use the first item as representative for risk computation
        hashes = sketch_hashes(sketch_items[0], *sketch.shape)
        # Original sum approximated by the L1 norm of v for demonstration
        rho = reconstruction_risk(sketch, hashes, original_sum=v.sum())
    else:
        rho = 0.0

    p_hat = p * (1.0 - rho)                            # effective priority
    decision = p_hat * s                               # hybrid decision vector

    ent = entropy_of_vector(decision)
    return decision, ent


def train_step(W: np.ndarray,
               text_batch: List[str],
               morph_batch: List[Morphology],
               sketch: np.ndarray,
               lr: float = 0.001) -> np.ndarray:
    """
    Perform a single training iteration over a batch.
    The gradient is approximated as the mean outer product of feature vectors
    with the error between current decision and a dummy target (zeros).
    This keeps the example self‑contained.
    """
    grads = np.zeros_like(W)
    for txt, morph in zip(text_batch, morph_batch):
        v = extract_text_features(txt)               # (9,)
        decision, _ = hybrid_score(txt, morph, W, sketch, sketch_items=[txt])
        # Target is a zero vector of same shape as decision
        error = decision  # since target = 0
        # Approximate gradient: outer(error, v)
        grads += np.outer(error, v) / v.size
    grad_mean = grads / len(text_batch)
    return gradient_descent_step(W, grad_mean, lr)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample morphology
    sample_morph = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Sample text
    sample_text = "The quick brown fox jumps over the lazy dog. And then it ran 2 miles!"

    # Initialise components
    W = init_ttt(d_in=9, d_out=5, scale=0.05, seed=42)   # 5‑dim decision space
    sketch = init_sketch(width=256, depth=4, seed=99)

    # Compute hybrid decision
    decision_vec, decision_entropy = hybrid_score(
        text=sample_text,
        morph=sample_morph,
        W=W,
        sketch=sketch,
        sketch_items=[sample_text]
    )

    print("Hybrid decision vector:", decision_vec)
    print("Decision entropy:", decision_entropy)

    # Perform a dummy training step
    batch_texts = [sample_text, "Another example sentence with numbers 12345."]
    batch_morphs = [sample_morph,
                    Morphology(length=1.5, width=0.8, height=0.4, mass=2.2)]

    W = train_step(W, batch_texts, batch_morphs, sketch, lr=0.005)
    print("Updated weight matrix shape:", W.shape)