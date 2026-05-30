# DARWIN HAMMER — match 3113, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s2.py (gen4)
# born: 2026-05-29T23:47:53Z

"""Hybrid Algorithm combining Radial Basis Function similarity (Parent A) and Regret‑Bandit‑Koopman dynamics (Parent B).

Mathematical Bridge:
- Parent A builds a similarity matrix **S** over node feature vectors using a binary hash
  and Hamming distance.  This matrix is a stochastic kernel after row‑normalisation.
- Parent B evolves a probability (or mean‑reward) vector **μₜ** with a Koopman operator **K**
  (μ̂ₜ₊ₕ = Kʰ μₜ) and modulates exploration via a store whose dynamics depend on the Gini
  coefficient of a regret‑weighted strategy **pₜ**.

The fusion treats **S** as the Koopman operator **K**.  The similarity‑driven kernel
propagates the regret‑weighted probability vector, while the Gini‑derived store level
scales the confidence bounds used by a contextual bandit.  Thus the two topologies are
merged into a single linear‑algebraic pipeline.

The module provides:
1. `similarity_matrix` – builds **S** from feature vectors (Parent A).
2. `compute_regret_weights` – builds a regret‑weighted distribution **pₜ** (Parent B).
3. `koopman_forecast` – propagates **pₜ** with **K = S**.
4. `gini_coefficient` – measures inequality of **pₜ** (Parent B).
5. `StoreState.update` – updates store level using inflow/outflow (Parent B).
6. `hybrid_select_action` – end‑to‑end hybrid decision using the above components.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types used throughout the module
Node = int
FeatureVec = Tuple[float, ...]
ActionID = str

# ----------------------------------------------------------------------
# Parent A – similarity utilities
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    """Euclidean distance between two feature vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: binarise values against their mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer hashes."""
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a symmetric similarity matrix S ∈ [0,1]^{n×n} using perceptual hashes.
    Entry S[i,j] = 1 - (Hamming distance / 64).
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[node])) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

# ----------------------------------------------------------------------
# Parent B – regret, Gini, store, and Koopman utilities
def gini_coefficient(prob_vec: np.ndarray) -> float:
    """Gini coefficient of a probability vector (0 = equal, 1 = maximal inequality)."""
    if prob_vec.ndim != 1:
        raise ValueError("prob_vec must be 1‑dimensional")
    sorted_vals = np.sort(prob_vec)  # ascending
    n = prob_vec.size
    cumulative = np.cumsum(sorted_vals)
    gini = 1.0 - (2.0 / (n - 1)) * (np.sum(cumulative) / cumulative[-1] - (n + 1) / 2.0)
    return max(0.0, min(1.0, gini))

def compute_regret_weights(
    expected_values: List[float],
    baseline: float = 0.0,
) -> np.ndarray:
    """
    Compute a regret‑weighted probability distribution over actions.
    Regret for action i: r_i = max(0, expected_values[i] - baseline).
    The distribution p_i = r_i / Σ r_i (uniform if all regrets are zero).
    """
    regrets = np.maximum(0.0, np.array(expected_values) - baseline)
    total = regrets.sum()
    if total == 0.0:
        # fall back to uniform distribution
        return np.full_like(regrets, 1.0 / regrets.size)
    return regrets / total

class StoreState:
    """
    Simple store dynamics: level evolves with inflow/outflow.
    Used to modulate confidence bounds in the hybrid bandit.
    """
    def __init__(
        self,
        level: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        base: float = 1.0,
        gain: float = 1.0,
        limit: float = 10.0,
    ):
        self.level = level
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.base = base
        self.gain = gain
        self.limit = limit

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Update store level and return (new_level, delta)."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = min(self.limit, max(0.0, self.level + delta * self.dt))
        return self.level, delta

def koopman_forecast(K: np.ndarray, mu: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Propagate the state vector mu with the Koopman operator K for `horizon` steps:
        μ̂ = K^h μ
    K is assumed to be row‑stochastic (each row sums to 1).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    K_power = np.linalg.matrix_power(K, horizon)
    return K_power @ mu

# ----------------------------------------------------------------------
# Hybrid core: combine similarity kernel with regret‑bandit dynamics
def hybrid_select_action(
    features: Dict[Node, FeatureVec],
    action_expected: List[float],
    store: StoreState,
    horizon: int = 1,
) -> Tuple[ActionID, float]:
    """
    End‑to‑end hybrid decision:
    1. Build similarity kernel S from node features (Parent A).
    2. Normalise rows of S to obtain a stochastic Koopman operator K.
    3. Compute regret‑weighted distribution p from action expectations (Parent B).
    4. Propagate p with K → μ̂ (exploitation term).
    5. Compute Gini(p) and update the store; the store level scales a confidence
       multiplier that is added to μ̂ for exploration.
    6. Return the index (as string) of the action with maximal adjusted score.
    """
    # 1‑2. Similarity kernel → stochastic Koopman operator
    S, nodes = similarity_matrix(features)
    row_sums = S.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    K = S / row_sums

    # 3. Regret‑weighted probability vector over actions
    p = compute_regret_weights(action_expected)

    # 4. Koopman forecast (exploitation)
    mu_hat = koopman_forecast(K, p, horizon=horizon)

    # 5. Store update based on Gini inequality
    gini = gini_coefficient(p)
    inflow = [gini * store.gain]
    outflow = [store.base]
    level, _ = store.update(inflow, outflow)

    # Confidence multiplier: higher store level → more exploration
    confidence = store.base + level
    adjusted_scores = mu_hat + confidence * np.random.rand(*mu_hat.shape) * 0.01  # small jitter

    # 6. Select action with highest adjusted score
    selected_index = int(np.argmax(adjusted_scores))
    selected_action_id = f"action_{selected_index}"
    return selected_action_id, float(adjusted_scores[selected_index])

# ----------------------------------------------------------------------
# Smoke test
if __name__ == "__main__":
    # Minimal synthetic data
    rng = np.random.default_rng(42)

    # Feature vectors for 5 nodes (2‑dimensional)
    features = {i: tuple(rng.random(2).tolist()) for i in range(5)}

    # Expected values for 5 actions
    action_expected = rng.random(5).tolist()

    # Initialise store
    store = StoreState(level=0.5, alpha=0.8, beta=0.3, dt=0.5, base=1.0, gain=2.0, limit=5.0)

    # Run hybrid selection
    action_id, score = hybrid_select_action(
        features=features,
        action_expected=action_expected,
        store=store,
        horizon=2,
    )
    print(f"Selected {action_id} with adjusted score {score:.4f}")
    print(f"Store level after update: {store.level:.4f}")