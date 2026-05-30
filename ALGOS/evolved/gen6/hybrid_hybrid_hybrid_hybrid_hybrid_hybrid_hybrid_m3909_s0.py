# DARWIN HAMMER — match 3909, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1335_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m807_s0.py (gen4)
# born: 2026-05-29T23:52:22Z

"""
Hybrid Sheaf‑RBF‑SSIM‑Weekday‑Bandit Algorithm

Parents:
- hybrid_hybrid_hybrid_ternar_m1335_s4.py (Gaussian RBF + SSIM similarity,
  Fisher pruning, sheaf restriction maps, contextual bandit routing)
- hybrid_hybrid_hybrid_worksh_hybrid_hybrid_hybrid_m807_s0.py (weekday‑dependent
  weight vectors, sheaf restriction maps, liquid‑time‑constant modulation,
  MinHash similarity)

Mathematical Bridge:
Both parents treat a *sheaf restriction map* as a linear weighting of pairwise
similarities between items.  Parent A builds a fused similarity
    S = α·S_g + (1‑α)·S_s
while Parent B injects a weekday‑dependent weight vector w(d) into the same
restriction maps.  The hybrid therefore defines the restriction matrix

    R(i,j) = w_i(d) · S(i,j) ,

where w_i(d) is the i‑th component of the normalized weekday vector for the
current day d.  R is then fed to a Fisher‑style score, to a liquid‑time‑constant
( τ ) modulation, and finally to a multiplicative‑weights contextual bandit.
The resulting system simultaneously captures uncertainty (RBF), structural
image‑like similarity (SSIM), temporal context (weekday), and adaptive decision
making (bandit).
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Node = int
FeatureVec = Tuple[float, ...]


# ----------------------------------------------------------------------
# Parent‑A style utilities
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rbf_similarity_matrix(features: List[FeatureVec], epsilon: float = 1.0) -> np.ndarray:
    """Compute Gaussian RBF similarity matrix S_g(i,j)."""
    n = len(features)
    S = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d = euclidean(features[i], features[j])
            s = gaussian_rbf(d, epsilon)
            S[i, j] = s
            S[j, i] = s
    return S


def ssim_similarity_matrix(features: List[FeatureVec]) -> np.ndarray:
    """
    Approximate SSIM similarity between feature vectors.
    Here we use a normalized dot‑product (cosine‑like) similarity as a proxy.
    """
    n = len(features)
    mat = np.array(features, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    normalized = mat / norms
    S = normalized @ normalized.T
    # shift to [0,1]
    S = (S + 1.0) / 2.0
    return S


def fused_similarity(alpha: float,
                     S_g: np.ndarray,
                     S_s: np.ndarray) -> np.ndarray:
    """Blend Gaussian RBF and SSIM similarities."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0,1]")
    return alpha * S_g + (1.0 - alpha) * S_s


# ----------------------------------------------------------------------
# Parent‑B style utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def liquid_time_constant_update(state: np.ndarray,
                                input_signal: np.ndarray,
                                tau_base: float,
                                tau_mod: np.ndarray,
                                dt: float = 0.01) -> np.ndarray:
    """
    Discrete update of a Liquid‑Time‑Constant (LTC) node.

    d x / d t = - x / τ + u
    τ is modulated element‑wise by ``tau_mod`` (derived from similarity).

    Returns the next state vector.
    """
    if state.shape != input_signal.shape:
        raise ValueError("state and input_signal must share shape")
    tau = tau_base * (1.0 + tau_mod)  # ensure positivity
    dx = (-state / tau) + input_signal
    return state + dt * dx


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def restriction_matrix(similarity: np.ndarray,
                       weekday_vec: np.ndarray) -> np.ndarray:
    """
    Build the sheaf restriction matrix R(i,j) = w_i * S(i,j).

    ``weekday_vec`` is assumed to have length equal to the number of rows of
    ``similarity``.  The vector is broadcast over columns.
    """
    if similarity.shape[0] != weekday_vec.shape[0]:
        raise ValueError("weekday vector length must match similarity dimension")
    return similarity * weekday_vec[:, np.newaxis]


def fisher_score(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute a Fisher‑style score for each entry:
        F(i,j) = (μ - S(i,j))^2 / (σ^2 + eps)

    where μ and σ are the global mean and std of the matrix.
    The score is normalised to [0,1].
    """
    mu = matrix.mean()
    sigma = matrix.std()
    raw = ((mu - matrix) ** 2) / (sigma ** 2 + eps)
    # normalise
    f = (raw - raw.min()) / (raw.max() - raw.min() + eps)
    return f


def contextual_bandit_update(probs: np.ndarray,
                             rewards: np.ndarray,
                             gamma: float = 0.1) -> np.ndarray:
    """
    Multiplicative‑weights update for routing probabilities.

        p_i ← p_i * exp(gamma * r_i)
        p ← p / sum(p)

    ``probs`` and ``rewards`` must be 1‑D arrays of the same length.
    """
    if probs.shape != rewards.shape:
        raise ValueError("probs and rewards must have same shape")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non‑negative")
    new = probs * np.exp(gamma * rewards)
    total = new.sum()
    if total == 0:
        # fallback to uniform distribution
        return np.full_like(probs, 1.0 / probs.size)
    return new / total


def hybrid_step(features: List[FeatureVec],
                alpha: float,
                weekday: int,
                tau_base: float = 1.0,
                gamma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute one hybrid iteration:

    1. Build Gaussian RBF and SSIM similarity matrices.
    2. Fuse them with weight ``alpha``.
    3. Modulate by weekday weight vector → restriction matrix R.
    4. Compute Fisher scores → pruning mask.
    5. Use the mask as a reward signal for a contextual bandit that updates
       routing probabilities.
    6. Feed the masked similarity into a liquid‑time‑constant update.

    Returns:
        probs   – updated routing probabilities (size = n)
        state   – updated LTC state vector (size = n)
    """
    # 1‑2 similarity
    S_g = rbf_similarity_matrix(features)
    S_s = ssim_similarity_matrix(features)
    S = fused_similarity(alpha, S_g, S_s)

    # 3 restriction with weekday weights
    w = weekday_weight_vector(GROUPS, weekday)[:S.shape[0]]
    R = restriction_matrix(S, w)

    # 4 Fisher pruning (used as reward)
    F = fisher_score(R)

    # 5 Bandit routing
    init_probs = np.full(R.shape[0], 1.0 / R.shape[0], dtype=np.float64)
    probs = contextual_bandit_update(init_probs, F.mean(axis=1), gamma)

    # 6 LTC dynamics – treat R.mean(axis=1) as input signal, similarity as τ‑modulation
    input_signal = R.mean(axis=1)
    tau_mod = R.mean(axis=1)  # similarity‑driven modulation
    state = liquid_time_constant_update(np.zeros_like(input_signal),
                                        input_signal,
                                        tau_base,
                                        tau_mod)

    return probs, state


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy feature vectors (5 nodes, 3‑dimensional)
    dummy_features = [tuple(np.random.rand(3)) for _ in range(5)]

    # Choose parameters
    alpha = 0.6
    today = date.today()
    dow = doomsday(today.year, today.month, today.day)

    # Run a single hybrid iteration
    probs, state = hybrid_step(dummy_features, alpha, dow)

    print("Routing probabilities:", probs)
    print("LTC state vector   :", state)