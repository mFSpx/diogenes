# DARWIN HAMMER — match 4733, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s4.py (gen5)
# parent_b: hybrid_hybrid_liquid_time_c_hybrid_hybrid_hybrid_m259_s1.py (gen4)
# born: 2026-05-29T23:57:47Z

"""Hybrid RBF‑MinHash‑LTC‑Ternary Router (HRMLT)

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – provides radial‑basis‑function (RBF) kernels, Euclidean distances,
  perceptual hashing and Hamming metrics.
* **Parent B** – contributes a Liquid‑Time‑Constant (LTC) dynamical update,
  MinHash‑based similarity signatures and a ternary‑router that maps continuous
  activations to {‑1,0,+1}.

**Mathematical bridge**

1. From Parent A we obtain an RBF kernel matrix **K** ∈ ℝⁿˣⁿ over raw numeric
   features. Each row *Kᵢ* is a similarity profile of node *i* with all others.
2. Treating *Kᵢ* as a multiset of real numbers, we generate a MinHash signature
   **hᵢ** (Parent B) that approximates Jaccard similarity between rows.
3. The ternary‑router (Parent B) maps the raw row *Kᵢ* to a ternary vector
   **rᵢ** ∈ {‑1,0,+1}ⁿ, providing a sparse directional cue.
4. The LTC update (Parent B) evolves a hidden state **sᵢ** with an
   input‑dependent time‑constant τᵢ that is a function of both the MinHash
   signature and the ternary output:
   
   τᵢ = τ₀ / (1 + α·σ( mean(hᵢ) + mean(rᵢ) ))  

   sᵢ(t+1) = sᵢ(t) + (Kᵢ – sᵢ(t))·(1 – exp(‑Δt/τᵢ))

The resulting system simultaneously respects kernel‑based similarity,
approximate set‑based Jaccard similarity, sparse ternary routing and
input‑adaptive temporal dynamics.

The public API consists of three high‑level functions:
`init_hybrid`, `hybrid_forward`, and `hybrid_loss`.
"""

import math
import random
import sys
import pathlib
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# 1. Core primitives from Parent A
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return float(np.linalg.norm(a - b))


def compute_phash(values: np.ndarray) -> int:
    """Simple perceptual hash based on mean threshold (64‑bit)."""
    if values.size == 0:
        return 0
    avg = values.mean()
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()


# ----------------------------------------------------------------------
# 2. RBF kernel matrix (Parent A)
# ----------------------------------------------------------------------
def rbf_kernel_matrix(
    features: Dict[int, np.ndarray], epsilon: float = 1.0
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the symmetric RBF kernel matrix K_ij = exp(-epsilon^2 * ||x_i-x_j||^2).

    Parameters
    ----------
    features : dict[int, np.ndarray]
        Mapping from node id to its raw feature vector.
    epsilon : float, optional
        Bandwidth of the Gaussian kernel.

    Returns
    -------
    K : np.ndarray, shape (n, n)
        RBF kernel matrix.
    idx : list[int]
        Node ids in the order used to build K.
    """
    idx = list(features.keys())
    X = np.stack([features[i] for i in idx])  # shape (n, d)
    # Pairwise squared Euclidean distances
    diff = X[:, None, :] - X[None, :, :]  # (n, n, d)
    sqdist = np.sum(diff ** 2, axis=2)   # (n, n)
    K = np.exp(- (epsilon ** 2) * sqdist)
    return K, idx


# ----------------------------------------------------------------------
# 3. MinHash signature (Parent B)
# ----------------------------------------------------------------------
def _hash_fn(data: bytes, seed: int) -> int:
    """Deterministic hash based on SHA‑256 and an integer seed."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(4, byteorder="little"))
    h.update(data)
    # Use first 8 bytes → 64‑bit integer
    return int.from_bytes(h.digest()[:8], byteorder="little")


def minhash_signature(row: np.ndarray, num_perm: int = 64) -> np.ndarray:
    """
    Compute a MinHash signature for a 1‑D numeric array.

    The row is first quantised into a set of string tokens (rounded values).
    For each permutation (seed) we keep the minimal hash value.

    Returns
    -------
    signature : np.ndarray, shape (num_perm,)
        64‑bit integers representing the MinHash sketch.
    """
    # Quantise to strings to form a multiset
    tokens = [f"{v:.5f}" for v in row]
    signature = np.empty(num_perm, dtype=np.uint64)
    for i in range(num_perm):
        min_hash = (1 << 64) - 1
        for t in tokens:
            h = _hash_fn(t.encode("utf-8"), i)
            if h < min_hash:
                min_hash = h
        signature[i] = min_hash
    return signature


def jaccard_estimate(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return float(np.mean(sig_a == sig_b))


# ----------------------------------------------------------------------
# 4. Ternary router (Parent B)
# ----------------------------------------------------------------------
def ternary_router(x: np.ndarray, low: float = -0.33, high: float = 0.33) -> np.ndarray:
    """
    Map continuous values to {-1, 0, +1}.

    -1 if x < low, 0 if low ≤ x ≤ high, +1 if x > high.
    """
    out = np.zeros_like(x, dtype=np.int8)
    out[x < low] = -1
    out[x > high] = 1
    return out


# ----------------------------------------------------------------------
# 5. Liquid‑Time‑Constant (LTC) dynamics (Parent B)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )


def liquid_time_constant_update(
    state: np.ndarray,
    input_vec: np.ndarray,
    tau_base: float = 1.0,
    alpha: float = 1.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Perform one LTC update step.

    τ = τ₀ / (1 + α·σ(mean(input_vec)))
    s_{t+1} = s_t + (input_vec - s_t)·(1 - exp(-dt/τ))
    """
    mean_input = float(input_vec.mean())
    tau = tau_base / (1.0 + alpha * float(sigmoid(np.array([mean_input]))))
    decay = 1.0 - math.exp(-dt / tau)
    return state + (input_vec - state) * decay


# ----------------------------------------------------------------------
# 6. Hybrid system orchestration
# ----------------------------------------------------------------------
def init_hybrid(
    raw_features: Dict[int, np.ndarray],
    epsilon: float = 1.0,
    num_minhash: int = 64,
) -> Dict[str, Any]:
    """
    Initialise the hybrid state.

    Returns a dictionary containing:
    - 'K'   : RBF kernel matrix
    - 'ids' : node ordering
    - 'state' : hidden state vector (zeros)
    - 'mh_sig' : MinHash signatures for each row
    - 'router' : ternary routing vectors for each row
    """
    K, ids = rbf_kernel_matrix(raw_features, epsilon)
    mh_sig = np.stack([minhash_signature(K[i], num_minhash) for i in range(K.shape[0])])
    router = np.stack([ternary_router(K[i]) for i in range(K.shape[0])])
    state = np.zeros(K.shape[0], dtype=np.float64)
    return {
        "K": K,
        "ids": ids,
        "state": state,
        "mh_sig": mh_sig,
        "router": router,
        "epsilon": epsilon,
        "num_minhash": num_minhash,
    }


def hybrid_forward(
    hybrid_state: Dict[str, Any],
    external_input: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform a single forward step.

    The effective input to the LTC is the element‑wise sum of:
    - the kernel row (similarity profile)
    - the average MinHash signature (scaled to [0,1])
    - the ternary routing vector (scaled to [-0.5,0.5])

    Parameters
    ----------
    hybrid_state : dict
        The structure returned by `init_hybrid`.
    external_input : np.ndarray, optional
        Optional additional stimulus of shape (n,). If provided it is added
        to the computed input before the LTC update.

    Returns
    -------
    new_state : np.ndarray
        Updated hidden state.
    """
    K = hybrid_state["K"]
    state = hybrid_state["state"]
    mh_sig = hybrid_state["mh_sig"]
    router = hybrid_state["router"]

    # Normalise MinHash signatures to [0,1] by dividing by max 64‑bit value
    mh_norm = mh_sig.mean(axis=1) / np.float64((1 << 64) - 1)

    # Scale ternary router to small magnitude so it does not dominate
    router_scaled = router.mean(axis=1) * 0.5

    # Composite input per node
    input_vec = K.mean(axis=1) + mh_norm + router_scaled
    if external_input is not None:
        if external_input.shape != input_vec.shape:
            raise ValueError("external_input shape mismatch")
        input_vec = input_vec + external_input

    # LTC dynamics
    new_state = liquid_time_constant_update(state, input_vec)

    # Store back
    hybrid_state["state"] = new_state
    return new_state


def hybrid_loss(
    hybrid_state: Dict[str, Any],
    target: np.ndarray,
) -> float:
    """
    Simple reconstruction loss: Euclidean distance between current state
    and a target vector (e.g., desired similarity profile).
    """
    state = hybrid_state["state"]
    if state.shape != target.shape:
        raise ValueError("target shape mismatch")
    return euclidean(state, target)


# ----------------------------------------------------------------------
# 7. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic raw features for 5 nodes (dim=3)
    rng = np.random.default_rng(42)
    raw = {i: rng.normal(size=3) for i in range(5)}

    # Initialise hybrid system
    hstate = init_hybrid(raw, epsilon=0.8, num_minhash=32)

    # Run a few forward steps with optional external stimulus
    for step in range(4):
        ext = rng.normal(scale=0.05, size=5)  # small noise stimulus
        new_s = hybrid_forward(hstate, external_input=ext)
        loss = hybrid_loss(hstate, target=np.ones_like(new_s) * 0.5)
        print(f"Step {step+1:02d} | state: {new_s.round(3)} | loss: {loss:.4f}")

    sys.exit(0)