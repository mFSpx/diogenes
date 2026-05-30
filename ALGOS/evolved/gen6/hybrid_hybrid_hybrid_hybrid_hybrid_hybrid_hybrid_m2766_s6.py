# DARWIN HAMMER — match 2766, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1862_s2.py (gen5)
# born: 2026-05-29T23:45:43Z

"""Hybrid Caputo‑MinHash Weekday (HCMW) algorithm.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_infota_m1404_s2.py (HCGSI)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1862_s2.py (Hybrid Workshare)

Mathematical bridge:
Parent A supplies a Caputo fractional‑derivative kernel  
 K(t,τ)= (t‑τ)^{‑α}/Γ(1‑α) (α∈(0,1))  
which provides long‑range temporal memory.  
Parent B builds a weekday‑dependent weight vector w(d) whose amplitude is
modulated by a MinHash signature σ(text).  

The fusion embeds the Caputo kernel as a **temporal weighting** of the
weekday‑weight vectors generated at each past timestep.  The resulting
memory‑augmented vector is finally scaled by the average of the MinHash
signature, thus coupling the fractional‑memory dynamics of Parent A with
the similarity‑driven amplitude modulation of Parent B.

The implementation below provides three core functions:
1. `caputo_kernel` – evaluates the fractional kernel.
2. `minhash_signature` – produces a lightweight MinHash of a string.
3. `hybrid_weekday_vector` – combines the kernel‑weighted weekday vectors
   with MinHash amplitude and returns a normalized hybrid weight vector.

A tiny multi‑armed bandit update (`update_bandit`) demonstrates downstream
use of the hybrid score.  The script ends with a smoke test that runs
without error."""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np
from datetime import date

# ----------------------------------------------------------------------
# Helper: Lanczos approximation for the Gamma function (Parent A)
# ----------------------------------------------------------------------
def lanczos_gamma(z: float) -> float:
    """Lanczos approximation of Γ(z) for z > 0."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    # Coefficients for g=7, n=9
    g = 7
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    z_minus_one = z - 1
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z_minus_one + i)
    t = z_minus_one + g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z_minus_one + 0.5) * math.exp(-t) * x

# ----------------------------------------------------------------------
# 1. Caputo fractional kernel (scalar)
# ----------------------------------------------------------------------
def caputo_kernel(t_now: float, t_past: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the Caputo fractional‑derivative kernel values
    K(t_now, τ_i) = (t_now - τ_i)^{-α} / Γ(1-α) for each past time τ_i.

    Parameters
    ----------
    t_now : float
        Current time.
    t_past : np.ndarray
        1‑D array of past timestamps (τ_i) with τ_i < t_now.
    alpha : float
        Fractional order, 0 < α < 1.

    Returns
    -------
    np.ndarray
        Kernel weights of shape (len(t_past),).
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    delta = t_now - t_past
    if np.any(delta <= 0):
        raise ValueError("All past times must be strictly less than t_now")
    gamma_term = lanczos_gamma(1 - alpha)
    return delta ** (-alpha) / gamma_term

# ----------------------------------------------------------------------
# 2. MinHash signature (Parent B)
# ----------------------------------------------------------------------
def minhash_signature(text: str, num_perm: int = 8, seed: int = 42) -> np.ndarray:
    """
    Generate a simple MinHash signature for *text*.

    The algorithm hashes the text with ``num_perm`` different seeds and
    records the integer hash values modulo 2**32.

    Parameters
    ----------
    text : str
        Input string to be hashed.
    num_perm : int
        Number of permutation functions (signature length).
    seed : int
        Base seed for reproducibility.

    Returns
    -------
    np.ndarray
        1‑D array of length ``num_perm`` containing uint32 hash values.
    """
    random.seed(seed)
    sig = np.empty(num_perm, dtype=np.uint32)
    for i in range(num_perm):
        h = hashlib.sha256()
        # Mix the seed into the text to obtain independent permutations
        h.update(f"{seed + i}_{text}".encode("utf-8"))
        # Take the first 4 bytes as uint32
        sig[i] = int.from_bytes(h.digest()[:4], byteorder="big")
    return sig

# ----------------------------------------------------------------------
# 3. Weekday base weight vector (Parent B)
# ----------------------------------------------------------------------
def weekday_base_vector(groups: tuple, dow: int) -> np.ndarray:
    """
    Produce a base (unmodulated) weight vector for *groups* based on the
    weekday index ``dow`` (0=Sunday … 6=Saturday). The vector lies on the
    unit circle in an n‑dimensional embedding.

    Parameters
    ----------
    groups : tuple
        Names of the groups (determines dimensionality).
    dow : int
        Weekday index.

    Returns
    -------
    np.ndarray
        Normalized weight vector of shape (len(groups),).
    """
    n = len(groups)
    angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    vec = np.cos(angles + phase)  # simple sinusoidal embedding
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# ----------------------------------------------------------------------
# 4. Hybrid operation: temporal memory + MinHash amplitude
# ----------------------------------------------------------------------
def hybrid_weekday_vector(
    groups: tuple,
    timestamps: np.ndarray,
    past_vectors: np.ndarray,
    alpha: float,
    current_time: float,
    dow: int,
    text: str,
) -> np.ndarray:
    """
    Compute the hybrid weight vector.

    Steps
    -----
    1. Evaluate the Caputo kernel K(t_now, τ_i) for each past timestamp.
    2. Form a kernel‑weighted sum of the stored weekday vectors
       ``past_vectors`` (shape: (m, n) where m = len(timestamps), n = len(groups)).
    3. Compute the MinHash signature of ``text`` and obtain its mean.
    4. Scale the kernel‑weighted sum by ``amp = 0.2 * (1 + mean(sig)/2**32)``.
    5. Normalize the final vector to unit length.

    Parameters
    ----------
    groups : tuple
        Group identifiers (dimensionality).
    timestamps : np.ndarray
        1‑D array of past timestamps (τ_i), length m.
    past_vectors : np.ndarray
        2‑D array of shape (m, n) containing the weekday base vectors
        that were generated at each τ_i.
    alpha : float
        Fractional order for the Caputo kernel.
    current_time : float
        Current timestamp t_now.
    dow : int
        Current weekday index.
    text : str
        Text whose MinHash signature modulates the amplitude.

    Returns
    -------
    np.ndarray
        Normalized hybrid weight vector of shape (n,).
    """
    if timestamps.shape[0] != past_vectors.shape[0]:
        raise ValueError("timestamps and past_vectors must have the same first dimension")
    # 1. Kernel weights
    k = caputo_kernel(current_time, timestamps, alpha)  # (m,)
    # 2. Weighted sum
    weighted_sum = np.dot(k, past_vectors)  # (n,)
    # 3. MinHash amplitude
    sig = minhash_signature(text)
    amp = 0.2 * (1.0 + np.mean(sig) / (2 ** 32))
    # 4. Scale and normalize
    hybrid_vec = amp * weighted_sum
    norm = np.linalg.norm(hybrid_vec)
    return hybrid_vec / norm if norm != 0 else hybrid_vec

# ----------------------------------------------------------------------
# 5. Simple multi‑armed bandit update (demonstration)
# ----------------------------------------------------------------------
def update_bandit(policy: np.ndarray, reward: float, lr: float = 0.1) -> np.ndarray:
    """
    Perform a single gradient‑ascent update on a probability policy
    vector based on the received *reward*.

    Parameters
    ----------
    policy : np.ndarray
        Current probability distribution over arms (must sum to 1).
    reward : float
        Scalar reward signal.
    lr : float
        Learning rate.

    Returns
    -------
    np.ndarray
        Updated policy (still normalized).
    """
    if not np.isclose(policy.sum(), 1.0):
        raise ValueError("policy must be a probability distribution")
    # Simple REINFORCE‑style update: increase probability proportional to reward
    grad = policy * (reward - np.dot(policy, reward))
    new_policy = policy + lr * grad
    # Re‑normalize to avoid drift
    new_policy = np.clip(new_policy, 0, None)
    if new_policy.sum() == 0:
        # fallback to uniform
        new_policy = np.full_like(policy, 1.0 / len(policy))
    else:
        new_policy /= new_policy.sum()
    return new_policy

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define groups and simulate a short history
    GROUPS = ("codex", "groq", "cohere", "local_models")
    n_groups = len(GROUPS)

    # Simulated timestamps (seconds)
    past_ts = np.array([0.0, 1.0, 2.5, 4.0])
    current_ts = 5.0

    # Generate weekday base vectors for each past timestamp using a dummy weekday
    past_dows = [doomsday(2026, 5, 20 + int(t)) % 7 for t in past_ts]
    past_vecs = np.vstack([weekday_base_vector(GROUPS, dow) for dow in past_dows])

    # Parameters
    alpha = 0.6  # fractional order
    today_dow = doomsday(2026, 5, 25) % 7
    sample_text = "Hybrid algorithm fusing Caputo memory with MinHash similarity."

    # Compute hybrid vector
    hybrid_vec = hybrid_weekday_vector(
        groups=GROUPS,
        timestamps=past_ts,
        past_vectors=past_vecs,
        alpha=alpha,
        current_time=current_ts,
        dow=today_dow,
        text=sample_text,
    )
    print("Hybrid weight vector:", hybrid_vec)

    # Use hybrid vector as a reward proxy for a 4‑armed bandit
    initial_policy = np.full(n_groups, 1.0 / n_groups)
    reward = float(np.dot(hybrid_vec, np.arange(1, n_groups + 1)))  # arbitrary linear reward
    updated_policy = update_bandit(initial_policy, reward)
    print("Updated bandit policy:", updated_policy)