# DARWIN HAMMER — match 4163, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s4.py (gen5)
# parent_b: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s2.py (gen4)
# born: 2026-05-29T23:53:52Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s4 (RBF surrogate + Fisher‑score adjusted circuit breaker)
- hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s2 (JEPA energy regularization, SSIM similarity, bandit router)

Mathematical Bridge:
Both parents rely on *similarity* as a core primitive.
Parent A builds a continuous similarity estimate 𝑆̂ via an RBF surrogate and
derives a Fisher score F(i) that rescales a failure threshold τ_i.
Parent B evaluates similarity between two matrices with SSIM and regularizes
predictions with a JEPA energy term ‖enc(x) − ŷ‖².

The fused algorithm therefore:
1. Constructs an RBF‑based similarity matrix 𝑆̂ from node feature vectors.
2. Computes per‑node Fisher scores from 𝑆̂ and uses them to obtain adaptive
   thresholds τ_i = τ₀·(1+α·F(i)).
3. Encodes the original features (encoder) and measures JEPA energy between the
   encoded features and the surrogate predictions, providing a scalar regularizer.
4. Compares 𝑆̂ to a reference similarity matrix (here the identity) via SSIM,
   yielding a global similarity quality q_ssim.
5. Forms a bandit reward that combines the adaptive‑threshold “stability” (higher
   τ_i → lower failure probability) with the JEPA‑energy improvement and the
   SSIM quality.  A simple Upper‑Confidence‑Bound (UCB) selector chooses the
   next routing action.

The implementation below integrates the matrix operations of both parents
into three high‑level functions that expose the hybrid behaviour."""


import math
import random
import sys
from pathlib import Path
from typing import Sequence, Mapping, Hashable, List, Tuple, Dict, Set, Iterable

import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Basic type aliases
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Core mathematical utilities (shared)
# ----------------------------------------------------------------------
def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def encoder(x: np.ndarray) -> np.ndarray:
    """L2‑normalised encoding (JEPA style)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    # avoid division by zero
    norm = np.where(norm == 0, 1.0, norm)
    return x / norm


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural similarity index for two 1‑D signals (or flattened matrices)."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + 1e-8) * (2 * sigma_xy + 1e-8)
    denominator = (mu_x ** 2 + mu_y ** 2 + 1e-8) * (sigma_x ** 2 + sigma_y ** 2 + 1e-8)
    return numerator / denominator


# ----------------------------------------------------------------------
# 1. RBF surrogate similarity matrix
# ----------------------------------------------------------------------
def rbf_similarity_matrix(features: List[FeatureVec], epsilon: float = 1.0) -> np.ndarray:
    """
    Compute an N×N similarity matrix Ŝ where
        Ŝ_ij = exp( - (ε·‖f_i - f_j‖)² )
    """
    n = len(features)
    S = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            r = euclidean(features[i], features[j])
            S[i, j] = gaussian_rbf(r, epsilon)
    return S


# ----------------------------------------------------------------------
# 2. Fisher score based on surrogate similarities
# ----------------------------------------------------------------------
def fisher_scores(similarity: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """
    Compute a per‑node Fisher‑like discriminative score.
    For each node i we treat its similarity vector s_i as a feature and
    evaluate between‑class variance (leader vs follower) over that vector.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) != 2:
        raise ValueError("Fisher score implementation expects exactly two classes.")
    class_means = {}
    class_vars = {}
    for c in uniq:
        mask = labels == c
        class_means[c] = np.mean(similarity[mask, :], axis=0)
        class_vars[c] = np.var(similarity[mask, :], axis=0) + 1e-8  # avoid zero

    # Between‑class variance (difference of means) squared
    between = (class_means[uniq[0]] - class_means[uniq[1]]) ** 2
    # Within‑class variance is average of the two class variances
    within = (class_vars[uniq[0]] + class_vars[uniq[1]]) / 2.0

    # Fisher score for each node: ratio of between to within (scalar per node)
    # We aggregate the vector score by taking the mean across dimensions.
    node_scores = np.empty(similarity.shape[0], dtype=float)
    for i in range(similarity.shape[0]):
        # Use the node's own similarity vector to weight the global ratio
        node_scores[i] = np.mean(between / within * similarity[i, :])
    # Normalise to [0,1]
    min_s, max_s = node_scores.min(), node_scores.max()
    if max_s - min_s > 0:
        node_scores = (node_scores - min_s) / (max_s - min_s)
    return node_scores


def adjusted_thresholds(base_tau: float, alpha: float, fisher: np.ndarray) -> np.ndarray:
    """
    τ_i = τ₀·(1 + α·F(i))
    """
    return base_tau * (1.0 + alpha * fisher)


# ----------------------------------------------------------------------
# 3. JEPA energy regularizer linking features and surrogate predictions
# ----------------------------------------------------------------------
def jepa_energy(features: List[FeatureVec], surrogate: np.ndarray) -> float:
    """
    Encode the raw features, flatten the surrogate matrix, and compute
    L2 energy: ‖enc(features) − vec(surrogate)‖²
    """
    X = np.asarray(features, dtype=float)          # shape (N, D)
    enc = encoder(X)                               # (N, D) L2‑normed
    flat_enc = enc.ravel()
    flat_sur = surrogate.ravel()
    # Pad the shorter vector with zeros so the shapes match
    max_len = max(flat_enc.size, flat_sur.size)
    pad_enc = np.pad(flat_enc, (0, max_len - flat_enc.size))
    pad_sur = np.pad(flat_sur, (0, max_len - flat_sur.size))
    return np.linalg.norm(pad_enc - pad_sur) ** 2


# ----------------------------------------------------------------------
# 4. Bandit structures and UCB selector
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float               # prior probability of being chosen
    expected_reward: float = 0.0    # updated online
    confidence_bound: float = 0.0   # UCB term
    algorithm: str = "hybrid_router"


def update_bandit(actions: List[BanditAction],
                  chosen_idx: int,
                  reward: float,
                  total_counts: int,
                  gamma: float = 0.1) -> List[BanditAction]:
    """
    Simple incremental update: expected_reward ← (1‑γ)·old + γ·reward
    confidence_bound ← sqrt( log(total_counts) / (chosen_counts+1) )
    """
    new_actions = []
    for idx, act in enumerate(actions):
        if idx == chosen_idx:
            new_exp = (1 - gamma) * act.expected_reward + gamma * reward
            # For demonstration we treat propensity as count proxy
            new_prop = act.propensity + 1
        else:
            new_exp = act.expected_reward
            new_prop = act.propensity
        # confidence term (UCB)
        cb = math.sqrt(math.log(max(total_counts, 1) + 1) / (new_prop + 1e-8))
        new_actions.append(
            BanditAction(
                action_id=act.action_id,
                propensity=new_prop,
                expected_reward=new_exp,
                confidence_bound=cb,
                algorithm=act.algorithm,
            )
        )
    return new_actions


def select_action_ucb(actions: List[BanditAction]) -> Tuple[int, BanditAction]:
    """
    Upper‑Confidence‑Bound selection: choose argmax(expected_reward + confidence_bound)
    Returns the index and the chosen action.
    """
    scores = [a.expected_reward + a.confidence_bound for a in actions]
    idx = int(np.argmax(scores))
    return idx, actions[idx]


# ----------------------------------------------------------------------
# 5. High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(features: List[FeatureVec],
                labels: Sequence[int],
                base_tau: float,
                alpha: float,
                actions: List[BanditAction],
                epsilon: float = 1.0) -> Dict:
    """
    Execute one hybrid iteration:
      • Build RBF surrogate similarity matrix.
      • Derive Fisher scores and adaptive thresholds.
      • Compute JEPA energy (regularizer) and SSIM quality against identity.
      • Assemble a scalar reward = w1·mean(τ_i) - w2·energy + w3·ssim
      • Use a bandit (UCB) to pick the next routing action.
    Returns a dictionary with intermediate results for inspection.
    """
    # 1. RBF surrogate
    S_hat = rbf_similarity_matrix(features, epsilon)

    # 2. Fisher and thresholds
    F = fisher_scores(S_hat, labels)
    tau_i = adjusted_thresholds(base_tau, alpha, F)

    # 3. JEPA energy regularizer
    energy = jepa_energy(features, S_hat)

    # 4. SSIM against an ideal similarity (identity matrix)
    identity = np.eye(len(features))
    ssim_score = ssim(S_hat.ravel(), identity.ravel())

    # 5. Compose reward (higher is better)
    w1, w2, w3 = 0.4, 0.4, 0.2
    reward = w1 * np.mean(tau_i) - w2 * energy + w3 * ssim_score

    # 6. Bandit decision
    total_counts = sum(int(a.propensity) for a in actions) + 1
    chosen_idx, chosen_action = select_action_ucb(actions)
    updated_actions = update_bandit(actions, chosen_idx, reward, total_counts)

    return {
        "similarity_matrix": S_hat,
        "fisher_scores": F,
        "thresholds": tau_i,
        "jepa_energy": energy,
        "ssim": ssim_score,
        "reward": reward,
        "chosen_action": chosen_action,
        "updated_actions": updated_actions,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic data for 5 nodes, 3‑dimensional features
    random.seed(42)
    np.random.seed(42)
    N = 5
    D = 3
    feats = [np.random.rand(D).tolist() for _ in range(N)]

    # binary labels: 0 = follower, 1 = leader
    lbls = np.random.randint(0, 2, size=N).tolist()

    # initial bandit actions
    init_actions = [
        BanditAction(action_id="route_A", propensity=1.0),
        BanditAction(action_id="route_B", propensity=1.0),
        BanditAction(action_id="route_C", propensity=1.0),
    ]

    result = hybrid_step(
        features=feats,
        labels=lbls,
        base_tau=0.5,
        alpha=0.3,
        actions=init_actions,
        epsilon=0.8,
    )

    print("=== Hybrid Iteration Summary ===")
    print(f"Labels                : {lbls}")
    print(f"Mean Fisher score     : {np.mean(result['fisher_scores']):.4f}")
    print(f"Mean adaptive τ       : {np.mean(result['thresholds']):.4f}")
    print(f"JEPA energy           : {result['jepa_energy']:.4f}")
    print(f"SSIM vs identity      : {result['ssim']:.4f}")
    print(f"Reward                : {result['reward']:.4f}")
    print(f"Chosen action         : {result['chosen_action'].action_id}")
    print("Updated bandit actions:")
    for a in result["updated_actions"]:
        print(f"  {a.action_id}: exp_reward={a.expected_reward:.4f}, prop={a.propensity}, cb={a.confidence_bound:.4f}")
    
    # sanity check: thresholds should be positive
    assert np.all(result["thresholds"] > 0), "All thresholds must be positive"
    # sanity check: similarity matrix is symmetric
    assert np.allclose(result["similarity_matrix"], result["similarity_matrix"].T), "Similarity matrix must be symmetric"
    print("Smoke test passed.")