# DARWIN HAMMER — match 5159, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (gen6)
# born: 2026-05-30T00:00:28Z

"""Hybrid Algorithm: Sketch‑Bandit‑RLCT‑Signature‑Entropy Fusion

Parents
-------
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (Bandit‑Sketch‑RLCT‑Ternary)
- hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (Path‑Signature‑Entropy‑MinHash‑RBF‑NLMS)

Mathematical Bridge
-------------------
Both parents expose a *log‑scale scalar* that modulates a learning or
exploration term:

* Parent A estimates the Real Log‑Canonical Threshold λ̂ by fitting  
  `log(Loss) = λ·log(n) + c`.  The term λ̂·log n̂ (with n̂ from a
  HyperLogLog sketch) is added to the UCB confidence bound.

* Parent B computes the Shannon entropy H of the level‑2 path‑signature
  eigen‑spectrum and uses it to scale the Gaussian‑RBF width ε and the
  NLMS learning rate η.

The fusion treats the two scalars as a *single multiplicative factor*  


γ = (1 + λ̂) * (1 + H)


which simultaneously inflates the exploration bonus and contracts the
RBF kernel.  The hybrid sketch combines a Count‑Min sketch (for
reward‑frequency estimation) with a MinHash sketch (for the auxiliary
force series).  Context vectors are enriched with ternary‑label counts,
textual token frequencies, signature moments and the peak velocity
derived from the MinHash force series.

The resulting algorithm is a contextual bandit that selects actions
using a γ‑augmented UCB, predicts rewards with an entropy‑scaled RBF
surrogate, and adapts its linear NLMS predictor with an η scaled by the
same γ.  This unifies the governing equations of both parents into a
single, mathematically coherent system.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Sequence, Dict

import numpy as np

# ----------------------------------------------------------------------
# Sketch primitives
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min sketch with pairwise‑independent hash functions."""
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        # generate random seeds for each depth level
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item, seed):
        return (hash((item, seed)) % self.width)

    def add(self, item):
        for d, seed in enumerate(self.seeds):
            idx = self._hash(item, seed)
            self.table[d, idx] += 1

    def estimate(self, item) -> int:
        return min(self.table[d, self._hash(item, seed)] for d, seed in enumerate(self.seeds))


class MinHashSketch:
    """Very light MinHash: keep the minimum hash value per hash function."""
    def __init__(self, num_perm: int = 64):
        self.num_perm = num_perm
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self.minhash = [2**63 - 1] * num_perm

    def _hash(self, item, seed):
        return hash((item, seed))

    def add(self, item):
        for i, seed in enumerate(self.seeds):
            h = self._hash(item, seed)
            if h < self.minhash[i]:
                self.minhash[i] = h

    def signature(self) -> List[int]:
        return self.minhash


# ----------------------------------------------------------------------
# RLCT estimation (Parent A)
# ----------------------------------------------------------------------
def estimate_rlct(losses: Sequence[float], sample_sizes: Sequence[int]) -> float:
    """Fit log(L) = λ·log(n) + c by ordinary least squares and return λ."""
    if len(losses) != len(sample_sizes) or len(losses) < 2:
        raise ValueError("Need at least two (loss, n) pairs")
    log_n = np.log(np.array(sample_sizes, dtype=float))
    log_L = np.log(np.array(losses, dtype=float) + 1e-12)  # avoid log(0)
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    λ, _ = np.linalg.lstsq(A, log_L, rcond=None)[0]
    return λ


# ----------------------------------------------------------------------
# Path signature and entropy (Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Create a lead‑lag version of a 1‑D path."""
    lead = path[:-1]
    lag = path[1:]
    return np.column_stack((lead, lag))


def compute_path_signature(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return level‑1 and level‑2 signatures (outer products)."""
    level1 = np.mean(path, axis=0, keepdims=True)  # shape (1, d)
    level2 = (path.T @ path) / path.shape[0]       # empirical covariance
    return level1, level2


def signature_entropy(level2_sig: np.ndarray) -> float:
    """Shannon entropy of the normalized eigen‑spectrum of level‑2 signature."""
    eigvals = np.linalg.eigvalsh(level2_sig)  # Hermitian → real eigenvalues
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)  # avoid zeros
    probs = eigvals / eigvals.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


# ----------------------------------------------------------------------
# MinHash force series → peak velocity (Parent B)
# ----------------------------------------------------------------------
def minhash_force_series(data: Sequence[float]) -> List[int]:
    """Hash each scalar into an integer; the collection forms a discrete force series."""
    return [int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % (2**31 - 1) for x in data]


def integrate_force_series(force_series: Sequence[int]) -> float:
    """Simple cumulative sum interpreted as displacement; peak velocity is max absolute diff."""
    cumulative = np.cumsum(force_series)
    velocities = np.diff(cumulative, prepend=0)
    peak_vel = float(np.max(np.abs(velocities)))
    return peak_vel


# ----------------------------------------------------------------------
# Hybrid feature construction
# ----------------------------------------------------------------------
def hybrid_feature_vector(
    path: np.ndarray,
    ternary_labels: Sequence[int],
    text_tokens: Sequence[str],
    force_data: Sequence[float],
) -> Tuple[np.ndarray, float, float]:
    """
    Build a unified feature vector Φ and return auxiliary scalars.
    Returns:
        Φ            – concatenated numeric feature vector
        γ_factor     – (1+λ̂)*(1+H) scaling factor
        peak_velocity– scalar derived from MinHash force series
    """
    # 1. Signature features
    lvl1, lvl2 = compute_path_signature(path)
    H = signature_entropy(lvl2)

    # 2. Ternary label counts (values in {−1,0,1})
    label_counts = Counter(ternary_labels)
    label_vec = np.array([label_counts.get(-1, 0),
                          label_counts.get(0, 0),
                          label_counts.get(1, 0)], dtype=float)

    # 3. Textual token frequencies (simple bag‑of‑words)
    token_counts = Counter(text_tokens)
    # keep only top‑k to bound dimensionality
    top_k = 20
    most_common = token_counts.most_common(top_k)
    token_vec = np.array([cnt for _, cnt in most_common], dtype=float)
    # pad if fewer than top_k tokens
    if token_vec.size < top_k:
        token_vec = np.pad(token_vec, (0, top_k - token_vec.size), constant_values=0)

    # 4. Force series → peak velocity
    force_series = minhash_force_series(force_data)
    peak_vel = integrate_force_series(force_series)

    # 5. Assemble Φ
    Φ = np.concatenate([lvl1.ravel(), lvl2.ravel(), label_vec, token_vec, np.array([peak_vel])])

    # 6. Placeholder RLCT estimate (will be updated later)
    λ_hat_placeholder = 0.0

    # γ = (1+λ̂)*(1+H) will be computed after λ̂ is known
    return Φ, λ_hat_placeholder, H, peak_vel


# ----------------------------------------------------------------------
# Hybrid UCB selection (Bandit + Entropy)
# ----------------------------------------------------------------------
def hybrid_ucb(
    action: str,
    sketch: CountMinSketch,
    total_counts: int,
    rlct_lambda: float,
    distinct_n: int,
    entropy_H: float,
    exploration_coef: float = 1.0,
) -> float:
    """
    Compute a γ‑augmented Upper Confidence Bound for a given action.
    """
    # empirical mean reward estimate via Count‑Min sketch
    reward_est = sketch.estimate(action)
    μ_hat = reward_est / max(1, total_counts)

    # classic UCB term
    bonus = exploration_coef * math.sqrt(math.log(total_counts + 1) / (reward_est + 1))

    # γ factor = (1+λ̂)*(1+H)
    gamma = (1.0 + rlct_lambda) * (1.0 + entropy_H)

    # RLCT‑aware term λ̂·log n̂
    rlct_term = rlct_lambda * math.log(distinct_n + 1)

    ucb = μ_hat + gamma * (bonus + rlct_term)
    return ucb


# ----------------------------------------------------------------------
# RBF surrogate with entropy‑scaled width
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, y: np.ndarray, base_eps: float, entropy_H: float) -> float:
    """
    Gaussian RBF kernel where ε = base_eps / (1+H).
    """
    eps = base_eps / (1.0 + entropy_H)
    diff = x - y
    return math.exp(-np.dot(diff, diff) / (2 * eps ** 2))


def rbf_predict(
    query: np.ndarray,
    support_X: np.ndarray,
    support_y: np.ndarray,
    base_eps: float,
    entropy_H: float,
) -> float:
    """
    Simple kernel ridge regression (λ=0) prediction using RBF kernel.
    """
    if support_X.shape[0] == 0:
        return 0.0
    K = np.array([rbf_kernel(query, xi, base_eps, entropy_H) for xi in support_X])
    # weights proportional to kernel values (normalized)
    weights = K / K.sum()
    return float(np.dot(weights, support_y))


# ----------------------------------------------------------------------
# NLMS update with entropy‑scaled learning rate
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    target: float,
    base_lr: float,
    entropy_H: float,
) -> np.ndarray:
    """
    Normalized LMS update: w ← w + η * (e / (||x||² + ε)) * x
    η = base_lr / (1+H)
    """
    eps = 1e-12
    eta = base_lr / (1.0 + entropy_H)
    pred = np.dot(w, x)
    e = target - pred
    norm = np.dot(x, x) + eps
    w_new = w + (eta * e / norm) * x
    return w_new


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    path: np.ndarray,
    ternary_labels: Sequence[int],
    text_tokens: Sequence[str],
    force_data: Sequence[float],
    action: str,
    sketch: CountMinSketch,
    minhash: MinHashSketch,
    total_counts: int,
    distinct_n: int,
    rlct_losses: List[float],
    rlct_samples: List[int],
    nlms_weights: np.ndarray,
    support_X: np.ndarray,
    support_y: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Perform one interaction step:
      1. Update sketches with the selected action.
      2. Re‑estimate λ̂ from accumulated losses.
      3. Build hybrid feature vector Φ.
      4. Compute γ‑augmented UCB for the action.
      5. Predict reward via RBF surrogate.
      6. Update NLMS weights.
    Returns:
        ucb_score, updated_nlms_weights, updated_support_X (with new Φ)
    """
    # 1. Sketch updates
    sketch.add(action)
    for token in text_tokens:
        minhash.add(token)

    # 2. RLCT estimate
    λ_hat = estimate_rlct(rlct_losses, rlct_samples)

    # 3. Feature vector
    Φ, _, H, _ = hybrid_feature_vector(path, ternary_labels, text_tokens, force_data)

    # 4. UCB
    ucb = hybrid_ucb(
        action=action,
        sketch=sketch,
        total_counts=total_counts,
        rlct_lambda=λ_hat,
        distinct_n=distinct_n,
        entropy_H=H,
        exploration_coef=1.0,
    )

    # 5. RBF prediction (use Φ as query)
    pred = rbf_predict(
        query=Φ,
        support_X=support_X,
        support_y=support_y,
        base_eps=1.0,
        entropy_H=H,
    )

    # Simulated target (for demonstration purposes)
    target = random.random()  # placeholder reward

    # 6. NLMS weight update
    nlms_w_new = nlms_update(
        w=nlms_weights,
        x=Φ,
        target=target,
        base_lr=0.5,
        entropy_H=H,
    )

    # Append new support point for future RBF predictions
    support_X = np.vstack([support_X, Φ]) if support_X.size else np.array([Φ])
    support_y = np.append(support_y, target)

    return ucb, nlms_w_new, support_X, support_y


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic path (2‑D)
    path = np.cumsum(np.random.randn(100, 2), axis=0)

    # Synthetic context
    ternary_labels = [random.choice([-1, 0, 1]) for _ in range(30)]
    text_tokens = [f"tok{random.randint(0, 50)}" for _ in range(40)]
    force_data = [random.random() for _ in range(20)]

    # Bandit scaffolding
    actions = ["a", "b", "c"]
    sketch = CountMinSketch(width=500, depth=4)
    minhash = MinHashSketch(num_perm=32)
    total_counts = 0
    distinct_n = 0  # placeholder for distinct context count

    # RLCT loss history (synthetic)
    rlct_losses = []
    rlct_samples = []

    # NLMS weights (dimension will match Φ after first call)
    nlms_weights = np.zeros(0)

    # RBF support set
    support_X = np.empty((0, 0))
    support_y = np.array([])

    # Run a few steps
    for step in range(5):
        action = random.choice(actions)

        # Simulate receiving a loss (negative reward) for RLCT