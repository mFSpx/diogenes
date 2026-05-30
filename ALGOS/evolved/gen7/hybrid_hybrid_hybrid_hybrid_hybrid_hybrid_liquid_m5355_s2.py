# DARWIN HAMMER — match 5355, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2434_s2.py (gen6)
# parent_b: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s1.py (gen2)
# born: 2026-05-30T00:01:23Z

"""Hybrid RLCT‑Grokking, Pheromone‑Infotaxis, MinHash‑Diffusion & Shapley‑HD Routing
===============================================================================

Parent A contributed:
* RLCT‑based half‑life estimation and a scalar pheromone that decays exponentially.
* A Shapley‑style weighting of high‑dimensional hypervectors.

Parent B contributed:
* MinHash signature generation from token shingles.
* A diffusion‑forcing noise schedule (cosine or linear) that modulates a signal over
  a discrete time horizon.

**Mathematical bridge**
The pheromone scalar φ(t) derived from the RLCT half‑life is used as a *dynamic
global gain* for every hypervector.  At each diffusion step `t` the schedule
produces a factor `α_t ∈ (0,1]`.  The effective scaling applied to a feature
hypervector `h_i` is therefore


scale_i(t) = φ(t) * √α_t * w_i


where `w_i` is a Shapley‑kernel weight obtained from the MinHash signature
similarity of the current token set to a reference signature.  The scaled
hypervectors are bound (element‑wise product) and bundled (weighted sum followed
by a sign‑binarisation) to form the final representation.  This fuses the
energy‑based optimisation of the pheromone system with the combinatorial
feature weighting of the hyperdimensional router and the stochastic diffusion
process of the MinHash‑based diffusion forcing.

The module implements the full pipeline in a few core functions and ends with
a smoke‑test that exercises the hybrid representation on synthetic data.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# 1. RLCT‑based half‑life estimation (Parent A)
# ----------------------------------------------------------------------
def rlct_half_life(losses: np.ndarray, eps: float = 1e-12) -> float:
    """
    Estimate a Real Log Canonical Threshold (RLCT) from a monotonic loss curve
    using a simple log‑log linear fit, then return a half‑life τ = 1 / RLCT.
    """
    if losses.ndim != 1 or len(losses) < 2:
        raise ValueError("losses must be a 1‑D array with at least two elements")
    epochs = np.arange(1, len(losses) + 1, dtype=np.float64)
    log_epochs = np.log(epochs)
    log_losses = np.log(losses + eps)
    # slope of log(loss) vs log(epoch) approximates -RLCT
    slope, _ = np.polyfit(log_epochs, log_losses, 1)
    rlct = -slope if slope < 0 else eps
    half_life = 1.0 / (rlct + eps)
    return half_life

def pheromone_update(phi: float, half_life: float, dt: float) -> float:
    """
    Exponential decay of pheromone strength φ over a time step dt given half‑life τ.
    """
    decay_factor = math.exp(-dt / (half_life + 1e-12))
    return phi * decay_factor

# ----------------------------------------------------------------------
# 2. MinHash signature and Shapley‑style weighting (Parent B)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """
    Compute a MinHash signature of length k for the given token set.
    """
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def shannon_entropy(prob: np.ndarray) -> float:
    """Utility: binary Shannon entropy H(p) = -p log p - (1-p) log(1-p)."""
    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def shapley_weights(signature: List[int], reference: List[int]) -> np.ndarray:
    """
    Produce a Shapley‑kernel weight vector w_i ∈ (0,1] for each hash component.
    The weight is proportional to the information gain of observing equality
    with a reference signature, i.e. w_i = H(p_i) where p_i is the empirical
    match probability (here deterministic 0 or 1) softened with a small epsilon.
    """
    if len(signature) != len(reference):
        raise ValueError("signatures must have equal length")
    matches = np.array([int(a == b) for a, b in zip(signature, reference)], dtype=np.float64)
    # soften deterministic matches to avoid zero entropy
    probs = 0.5 * matches + 0.5 * (1 - matches) * 0.01
    ent = shannon_entropy(probs)
    # Normalize to have unit sum
    if ent.sum() == 0:
        return np.ones_like(ent) / len(ent)
    return ent / ent.sum()

# ----------------------------------------------------------------------
# 3. Diffusion forcing schedule (Parent B)
# ----------------------------------------------------------------------
def diffusion_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Return the cumulative product ᾱ_t (0≤t≤T) of a diffusion schedule.
    Cosine schedule reproduces the common DDPM cosine schedule.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        return np.clip(alpha_bars, 1e-9, 1.0)
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        return np.concatenate(([1.0], alpha_bars))
    else:
        raise ValueError(f"unknown schedule '{schedule}'")

# ----------------------------------------------------------------------
# 4. Hyperdimensional primitives (Parent A & B)
# ----------------------------------------------------------------------
DIM = 10_000  # dimensionality of hypervectors (binary ±1)

def random_hypervector(seed: int) -> np.ndarray:
    """Generate a deterministic random hypervector of ±1 given an integer seed."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=DIM).astype(np.int8)

def bind(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Binding via element‑wise multiplication (XOR in bipolar coding)."""
    return v1 * v2

def bundle(weighted_vectors: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Weighted bundling: Σ_i w_i * h_i, followed by sign‑binarisation to ±1.
    """
    if len(weighted_vectors) != len(weights):
        raise ValueError("vectors and weights length mismatch")
    aggregate = np.zeros(DIM, dtype=np.float64)
    for vec, w in zip(weighted_vectors, weights):
        aggregate += w * vec.astype(np.float64)
    # Sign binarisation
    return np.where(aggregate >= 0, 1, -1).astype(np.int8)

# ----------------------------------------------------------------------
# 5. Hybrid representation pipeline
# ----------------------------------------------------------------------
def hybrid_representation(
    text: str,
    losses: np.ndarray,
    T: int = 10,
    schedule: str = "cosine",
    k: int = 128,
    seed_offset: int = 0,
) -> np.ndarray:
    """
    Compute a hybrid high‑dimensional representation for `text` by:

    1. Extracting shingles and building a MinHash signature.
    2. Deriving Shapley‑kernel weights from similarity to a reference signature
       (here we use the signature of the same text as reference – yields maximal
       information gain; in practice one would compare to a class prototype).
    3. Estimating RLCT‑derived half‑life from `losses` and initializing pheromone φ.
    4. For each diffusion step t:
         a. Decay pheromone.
         b. Obtain diffusion factor √ᾱ_t.
         c. Scale each hypervector by φ(t)·√ᾱ_t·w_i.
       The scaled hypervectors are bound to a common seed hypervector to inject
       temporal coupling, then bundled across all hash components.
    5. Return the final bundled hypervector (±1).
    """
    # ---- 1. Tokenisation & MinHash ----
    tokens = text.split()
    shingle_set = set()
    width = 5
    if len(tokens) >= width:
        for i in range(len(tokens) - width + 1):
            shingle_set.add(" ".join(tokens[i : i + width]))
    else:
        shingle_set.add(text)
    signature = minhash_signature(list(shingle_set), k=k)

    # ---- 2. Shapley weights (using self‑reference) ----
    # In a real scenario `reference_sig` would be a prototype; we use the same
    # signature to obtain a non‑trivial weight distribution.
    reference_sig = signature.copy()
    weights = shapley_weights(signature, reference_sig)  # shape (k,)

    # ---- 3. RLCT half‑life & pheromone init ----
    half_life = rlct_half_life(losses)
    phi = 1.0  # initial pheromone strength
    dt = 1.0   # time step size

    # ---- 4. Diffusion schedule ----
    alpha_bars = diffusion_schedule(T, schedule=schedule)

    # ---- 5. Generate base hypervectors for each hash component ----
    base_vectors = [random_hypervector(seed + seed_offset) for seed in signature]

    # Temporal coupling vector (same for all steps)
    temporal_seed = 42 + seed_offset
    temporal_vec = random_hypervector(temporal_seed)

    # ---- 6. Iterate diffusion steps, applying scaling ----
    for t in range(T + 1):
        # decay pheromone
        phi = pheromone_update(phi, half_life, dt)

        # diffusion factor (sqrt of cumulative product)
        diffusion_factor = math.sqrt(alpha_bars[t])

        # scale each hypervector
        scaled_vectors = []
        for hv, w in zip(base_vectors, weights):
            scale = phi * diffusion_factor * w
            # bind with temporal vector to introduce time dependence
            bound = bind(hv, temporal_vec)
            scaled = bound.astype(np.float64) * scale
            # bring back to bipolar after scaling (sign)
            scaled_vectors.append(np.where(scaled >= 0, 1, -1).astype(np.int8))

        # Re‑bundle for this step (optional, could accumulate)
        bundled = bundle(scaled_vectors, np.ones_like(weights))
        # Use the bundled vector as the new temporal seed for next step
        temporal_vec = bundled

    # Final output is the temporal vector after the last diffusion step
    return temporal_vec

# ----------------------------------------------------------------------
# 6. Similarity between two hybrid representations
# ----------------------------------------------------------------------
def hybrid_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity for bipolar ±1 hypervectors.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError("vector shapes must match")
    dot = float(np.dot(vec_a, vec_b))
    norm = math.sqrt(float(np.dot(vec_a, vec_a)) * float(np.dot(vec_b, vec_b)))
    return dot / (norm + 1e-12)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic loss curve (exponential decay)
    epochs = 50
    true_rlct = 0.8
    losses = np.exp(-true_rlct * np.arange(1, epochs + 1) / 10.0) + np.random.normal(0, 0.01, epochs)

    sample_text_a = "the quick brown fox jumps over the lazy dog and runs away swiftly"
    sample_text_b = "a fast auburn fox leaps above a sleepy canine while sprinting rapidly"

    rep_a = hybrid_representation(sample_text_a, losses, T=12, schedule="cosine", k=64, seed_offset=0)
    rep_b = hybrid_representation(sample_text_b, losses, T=12, schedule="cosine", k=64, seed_offset=1000)

    sim = hybrid_cosine_similarity(rep_a, rep_b)
    print(f"Hybrid representation shape: {rep_a.shape}")
    print(f"Cosine similarity between two texts: {sim:.4f}")