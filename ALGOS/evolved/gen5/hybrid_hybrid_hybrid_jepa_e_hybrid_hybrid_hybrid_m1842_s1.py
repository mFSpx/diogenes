# DARWIN HAMMER — match 1842, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_worksh_m117_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s4.py (gen4)
# born: 2026-05-29T23:39:07Z

"""
Hybrid Algorithm combining:
- Parent A: variational free-energy (VFE) model pool management with feature extraction and master vector generation.
- Parent B: multi-armed bandit actions coupled with an RBF surrogate model.

Mathematical bridge:
Features extracted from a textual context (Parent A) are used as the continuous
state vector *x* for the RBF surrogate (Parent B).  The bandit’s expected
reward *r̂* is multiplied by a VFE‑derived penalty *P(x, m)*, where *m* is the
master vector generated from the same text.  The hybrid prediction is

    y_hybrid = P(x, m) · r̂ · ϕ_RBF(x),

with ϕ_RBF(x) = Σ_i w_i·exp(−‖x−c_i‖²·ε²) the standard RBF surrogate.
Thus the stochastic exploration term (bandit) is modulated by the
variational free‑energy term (model‑pool) while sharing the same feature
space.
"""

import sys
import math
import random
import hashlib
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Feature extraction & VFE penalty
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    """Round a float to 6 decimal places."""
    return round(float(value), 6)

def doomsday(year: int, month: int, day: int) -> int:
    """Utility used by the original parent – retained for compatibility."""
    return (date(year, month, day).weekday() + 1) % 7

def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a SHA‑256 hash of *text*."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)

def extract_full_features(text: str) -> Dict[str, float]:
    """
    Produce a dictionary of 10 pseudo‑numeric features from *text*.
    The original parent used a longer list; we keep a representative subset.
    """
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "operator_entropy_index",
        "operator_latency_factor",
        "operator_complexity",
    ]
    return {k: _pct(rnd.random()) for k in keys}

def generate_master_vector(text: str, dim: int = 8) -> np.ndarray:
    """
    Create a deterministic master vector *m* ∈ ℝᵈ from *text*.
    The vector is normalised to unit length.
    """
    rng = _rng_from_text(text)
    vec = np.array([rng.random() for _ in range(dim)], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def compute_vfe_penalty(features: Dict[str, float], master_vec: np.ndarray) -> float:
    """
    Simple VFE‑style penalty:
        P = α·‖f‖₁ + β·(1 – ⟨f̂, m⟩)
    where f̂ is the feature vector normalised to unit length,
    m is the master vector, and α,β are tunable scalars.
    """
    α = 0.6
    β = 0.4
    f_vals = np.array(list(features.values()), dtype=float)
    l1_norm = np.sum(np.abs(f_vals))
    f_norm = np.linalg.norm(f_vals)
    if f_norm == 0:
        cosine = 0.0
    else:
        f_hat = f_vals / f_norm
        # Pad/truncate to match master_vec dimension
        if len(f_hat) < len(master_vec):
            f_hat = np.pad(f_hat, (0, len(master_vec) - len(f_hat)))
        else:
            f_hat = f_hat[: len(master_vec)]
        cosine = float(np.dot(f_hat, master_vec))
    penalty = α * l1_norm + β * (1.0 - cosine)
    return _pct(penalty)

class ModelPool:
    """
    Minimal model‑pool that decides whether to keep a model based on its
    VFE penalty.  Lower penalty → higher chance of retention.
    """
    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.pool: List[Tuple[str, float]] = []   # (model_id, penalty)

    def consider(self, model_id: str, penalty: float) -> bool:
        """Return True if the model is admitted to the pool."""
        # If space, always admit
        if len(self.pool) < self.capacity:
            self.pool.append((model_id, penalty))
            return True
        # Otherwise replace the worst (largest penalty) if new one is better
        worst_idx = max(range(len(self.pool)), key=lambda i: self.pool[i][1])
        if penalty < self.pool[worst_idx][1]:
            self.pool[worst_idx] = (model_id, penalty)
            return True
        return False

    def __repr__(self) -> str:
        return f"ModelPool(capacity={self.capacity}, entries={self.pool})"

# ----------------------------------------------------------------------
# Parent B – Bandit + RBF surrogate
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "UCB"

@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: List[float]) -> float:
        """Standard RBF prediction Σ w_i·exp(−‖x−c_i‖²·ε²)."""
        return sum(
            w * math.exp(-((self.epsilon * euclidean(x, list(c))) ** 2))
            for w, c in zip(self.weights, self.centers)
        )

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def bandit_rbf_hybrid_predict(
    features: Dict[str, float],
    master_vec: np.ndarray,
    bandit_action: BanditAction,
    rbf: RBFSurrogate,
) -> float:
    """
    Hybrid prediction:
        y = P(x,m) * r̂ * φ_RBF(x)
    where:
        - x  = feature vector (list of floats)
        - m  = master vector (numpy array)
        - P  = VFE penalty (scalar)
        - r̂ = bandit_action.expected_reward
        - φ_RBF = RBF surrogate output
    """
    # Convert feature dict to ordered list
    x = list(features.values())
    # Compute VFE penalty
    penalty = compute_vfe_penalty(features, master_vec)
    # RBF output
    rbf_out = rbf.predict(x)
    # Hybrid combination
    hybrid = penalty * bandit_action.expected_reward * rbf_out
    return hybrid

def social_interaction(x: List[float], g_best: List[float], k: int = 1, r1: float | None = None, seed: int | str | None = None) -> np.ndarray:
    """
    Simple particle‑swarm inspired interaction used as an auxiliary
    transformation of the feature vector before feeding it to the RBF.
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must share dimension")
    if k not in (1, 2):
        raise ValueError("k is normally 1 or 2")
    rng = random.Random(seed)
    r = rng.random() if r1 is None else r1
    return np.array([xi + r * (gj - k * xi) for xi, gj in zip(x, g_best)])

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_feature_pipeline(text: str) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Extract features and master vector from *text*.
    Returns (features, master_vector).
    """
    feats = extract_full_features(text)
    mvec = generate_master_vector(text, dim=len(feats))
    return feats, mvec

def demo_model_pool(texts: List[str]) -> ModelPool:
    """
    Build a ModelPool from a list of texts, using the VFE penalty as the
    admission criterion.
    """
    pool = ModelPool(capacity=3)
    for txt in texts:
        feats, mvec = demo_feature_pipeline(txt)
        penalty = compute_vfe_penalty(feats, mvec)
        pool.consider(model_id=hash(txt), penalty=penalty)
    return pool

def demo_hybrid_prediction(text: str, action: BanditAction, rbf: RBFSurrogate) -> float:
    """
    End‑to‑end hybrid prediction for a single *text* context.
    """
    feats, mvec = demo_feature_pipeline(text)
    return bandit_rbf_hybrid_predict(feats, mvec, action, rbf)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum entanglement enables teleportation of information.",
        "Artificial intelligence models require careful regularisation."
    ]

    # Build a model pool
    pool = demo_model_pool(sample_texts)
    print("ModelPool state:", pool)

    # Define a dummy bandit action
    action = BanditAction(
        action_id="a1",
        propensity=0.9,
        expected_reward=1.2,
        confidence_bound=0.05,
    )

    # Create a tiny RBF surrogate (centers from random vectors)
    rng = random.Random(42)
    dim = len(extract_full_features(sample_texts[0]))
    centers = [tuple(rng.random() for _ in range(dim)) for _ in range(4)]
    weights = [rng.uniform(-1, 1) for _ in range(4)]
    rbf = RBFSurrogate(centers=centers, weights=weights, epsilon=0.8)

    # Run hybrid predictions
    for txt in sample_texts:
        pred = demo_hybrid_prediction(txt, action, rbf)
        print(f"Hybrid prediction for \"{txt[:30]}...\": {pred:.6f}")

    # Demonstrate social interaction transformation
    feats, _ = demo_feature_pipeline(sample_texts[0])
    x = list(feats.values())
    g_best = [0.5] * len(x)
    transformed = social_interaction(x, g_best, seed=123)
    print("Transformed feature vector (first 5 elements):", transformed[:5])