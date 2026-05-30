# DARWIN HAMMER — match 1449, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m961_s0.py (gen5)
# born: 2026-05-29T23:36:37Z

"""Hybrid Fusion of Decision Hygiene Entropy and TTT‑Linear Dynamics

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m650_s1.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_liquid_m961_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a Shannon entropy of decision‑hygiene feature counts.
Algorithm B supplies a TTT‑Linear weight matrix that is updated by gradient
descent and modulated by a Liquid‑Time‑Constant (LTC) recurrent dynamics.
The fusion uses the entropy as a *global scaling factor* for the TTT‑Linear
gradient step and as a weight for Laplace‑style noise injected into the
weight matrix.  The entropy‑scaled update produces a weight matrix that
drives an LTC‑modulated state vector.  The state vector is then passed
through a fractional‑power binding (square‑root non‑linearity) and a
Gaussian kernel to yield a unified “health score”.  This construction
interleaves the information‑theoretic core of A with the linear‑algebraic
core of B, yielding a single coherent system."""

import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core primitives from Parent A
# ---------------------------------------------------------------------------

def shannon_entropy(counts: List[int]) -> float:
    """Shannon entropy of a histogram of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hyper‑vector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice([-1, 1], size=d)
    if kind == "real":
        vec = rng.normal(size=d)
        return vec / np.linalg.norm(vec)
    raise ValueError("Invalid hypervector kind")


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used in the health‑score computation."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("Lists must have equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Core primitives from Parent B
# ---------------------------------------------------------------------------

def init_ttt(d_in: int, d_out: int | None = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize the TTT‑Linear weight matrix."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale


def ttt_forward(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear forward pass of the TTT‑Linear module."""
    return W @ x


def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    """Mean‑squared Euclidean loss."""
    pred = ttt_forward(W, x)
    return np.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Hybrid operations (the fused mathematics)
# ---------------------------------------------------------------------------

def entropy_weighted_ttt_update(
    W: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    feature_counts: List[int],
    lr: float = 0.01,
    laplace_scale: float = 0.001,
) -> np.ndarray:
    """
    Perform a single gradient‑descent update of the TTT‑Linear matrix.
    The learning‑rate is scaled by the Shannon entropy of the decision‑hygiene
    feature histogram, and Laplace noise (scaled by the same entropy) is added
    to the weight matrix after the update.

    Returns the updated weight matrix.
    """
    # 1. Entropy as a global scaling factor (0 ≤ H ≤ log2(N))
    H = shannon_entropy(feature_counts) + 1e-9  # avoid zero

    # 2. Gradient of MSE loss w.r.t. W: (W·x - target) ⊗ x
    pred = ttt_forward(W, x)
    grad = np.outer(pred - target, x) / x.shape[0]

    # 3. Entropy‑scaled learning step
    W_new = W - (lr * H) * grad

    # 4. Inject Laplace noise proportional to entropy
    noise = np.random.laplace(loc=0.0, scale=laplace_scale * H, size=W.shape)
    W_new += noise
    return W_new


def minhash_similarity(v1: np.ndarray, v2: np.ndarray, num_perm: int = 128) -> float:
    """
    Approximate Jaccard similarity via random hyperplane (sign) hashing.
    Returns a value in [0, 1].
    """
    rng = np.random.default_rng(42)
    perms = rng.standard_normal((num_perm, v1.size))
    hash1 = (perms @ v1) > 0
    hash2 = (perms @ v2) > 0
    return np.mean(hash1 == hash2)


def liquid_time_constant(tau_base: float, similarity: float) -> float:
    """
    Compute an effective time constant τ_eff that shrinks as similarity grows.
    τ_eff = τ_base / (1 + similarity)
    """
    return tau_base / (1.0 + similarity)


def hybrid_state_update(
    state: np.ndarray,
    input_vec: np.ndarray,
    W: np.ndarray,
    tau_eff: float,
) -> np.ndarray:
    """
    LTC‑modulated recurrent update:
        s_{t+1} = s_t + τ_eff * (W·input - s_t)
    """
    drive = ttt_forward(W, input_vec)
    return state + tau_eff * (drive - state)


def fractional_power_binding(vec: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Element‑wise fractional power (sign‑preserving):
        f_i = sign(v_i) * |v_i|^α
    """
    return np.sign(vec) * (np.abs(vec) ** alpha)


def reconstruction_risk_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Ratio of reconstruction error to original norm.
    """
    err = np.linalg.norm(original - reconstructed)
    norm = np.linalg.norm(original) + 1e-9
    return err / norm


def hybrid_health_score(
    feature_counts: List[int],
    state: np.ndarray,
    tau_eff: float,
) -> float:
    """
    Unified health metric:
        H = H_entropy * mean(fractional_state) * Gaussian(||state||) / τ_eff
    """
    H_entropy = shannon_entropy(feature_counts) + 1e-9
    frac_state = fractional_power_binding(state)
    mean_frac = np.mean(frac_state)
    gauss = gaussian(np.linalg.norm(state))
    return (H_entropy * mean_frac * gauss) / (tau_eff + 1e-9)


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic decision‑hygiene feature histogram
    feature_counts = [random.randint(0, 20) for _ in range(15)]

    # Dimensionalities
    dim_in = 64
    dim_out = 64

    # Random input / target vectors
    rng = np.random.default_rng(123)
    x = rng.standard_normal(dim_in)
    target = rng.standard_normal(dim_out)

    # Initialise TTT‑Linear matrix
    W = init_ttt(dim_in, dim_out, scale=0.02, seed=7)

    # Initialise state vector (zero)
    state = np.zeros(dim_out)

    # Hyper‑vector projection (complex) – serves as a second view of x
    hv = random_hv(d=dim_in, kind="complex", seed=99)
    proj = np.real(hv * x)  # element‑wise multiplication, then real part

    # One hybrid update step
    W = entropy_weighted_ttt_update(W, proj, target, feature_counts, lr=0.05)

    # Similarity between current state and projected input
    sim = minhash_similarity(state, proj)

    # Effective time constant
    tau_base = 0.1
    tau_eff = liquid_time_constant(tau_base, sim)

    # Update the recurrent state
    state = hybrid_state_update(state, proj, W, tau_eff)

    # Compute health score
    health = hybrid_health_score(feature_counts, state, tau_eff)

    # Reconstruction risk (for diagnostics)
    recon = ttt_forward(W, proj)
    risk = reconstruction_risk_ratio(target, recon)

    print(f"Entropy: {shannon_entropy(feature_counts):.4f}")
    print(f"Similarity (MinHash): {sim:.4f}")
    print(f"Effective τ: {tau_eff:.6f}")
    print(f"Health score: {health:.6f}")
    print(f"Reconstruction risk ratio: {risk:.6f}")