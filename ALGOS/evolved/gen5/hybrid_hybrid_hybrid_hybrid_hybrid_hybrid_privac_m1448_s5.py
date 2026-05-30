# DARWIN HAMMER — match 1448, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s5.py (gen4)
# parent_b: hybrid_hybrid_privacy_model_hybrid_serpentina_se_m179_s1.py (gen2)
# born: 2026-05-29T23:36:25Z

"""Hybrid Algorithm Fusion of `hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s5.py`
and `hybrid_hybrid_privacy_model_hybrid_serpentina_se_m179_s1.py`.

Mathematical Bridge
-------------------
Parent A works with probability distributions over actions and evaluates them
via a Kullback‑Leibler (KL) divergence inside `hybrid_operation`.  
Parent B introduces differential‑privacy (DP) mechanisms: a Laplace‑perturbed
aggregate (`dp_aggregate`) and a reconstruction‑risk score.

The fusion treats the probability vector from Parent A as a *private* data
source.  Before computing the KL divergence we add Laplace noise (DP) to the
probabilities, normalise them back to a simplex, and then compute the KL
divergence against a ternary‑encoded signature vector.  The resulting scalar
is further weighted by the reconstruction‑risk score derived from the model
metadata of Parent B, yielding a single privacy‑aware “hybrid score”.

The module therefore intertwines the matrix‑style feature expansion of
Parent A (`lead_lag_transform`, `kan_basis`) with the privacy‑aware risk
assessment of Parent B, producing a unified hybrid operation."""
import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Tuple, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (merged concepts)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Action description used by Parent A."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class ModelTier:
    """Model metadata used by Parent B."""
    name: str
    ram_mb: int
    tier: str

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """Create linear and quadratic aggregated features."""
    linear_features = np.sum(X, axis=1, keepdims=True)
    quadratic_features = np.sum(X ** 2, axis=1, keepdims=True)
    return np.hstack((linear_features, quadratic_features))

def kan_basis(grid_size: int) -> np.ndarray:
    """Generate a simple exponential basis (Kan‐type)."""
    points = np.linspace(0, 1, grid_size)
    return np.exp(-points)

def prune_candidates(signatures: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Element‑wise mask of signature matrix by a schedule mask."""
    return signatures * schedule

# ----------------------------------------------------------------------
# Utilities from Parent B (privacy)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Differential‑privacy reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float],
                 epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """Laplace‑noised sum of values."""
    total = sum(values)
    noise = np.random.laplace(0.0, scale=sensitivity / epsilon)
    return total + noise

# ----------------------------------------------------------------------
# Hybrid mathematical core
# ----------------------------------------------------------------------
def _normalize_to_simplex(vec: np.ndarray) -> np.ndarray:
    """Project a non‑negative vector onto the probability simplex."""
    vec = np.maximum(vec, 0.0)
    if vec.sum() == 0:
        # Uniform distribution if all entries are zero after clipping
        return np.full_like(vec, 1.0 / vec.size)
    return vec / vec.sum()

def privacy_aware_kl(probabilities: np.ndarray,
                     ternary_vector: np.ndarray,
                     epsilon: float = 1.0) -> float:
    """
    Compute a KL divergence where the probability vector is first
    perturbed with Laplace noise (DP) and then normalised.

    Parameters
    ----------
    probabilities : np.ndarray
        Original probability distribution (must sum to 1).
    ternary_vector : np.ndarray
        Target distribution; entries are expected to be 0, 1 or 2.
        It is converted to a proper probability vector internally.
    epsilon : float
        Privacy budget for the Laplace mechanism.

    Returns
    -------
    float
        The privacy‑augmented KL divergence.
    """
    # Add Laplace noise to each probability entry
    noisy = probabilities + np.random.laplace(0.0, scale=1.0 / epsilon, size=probabilities.shape)
    p_noisy = _normalize_to_simplex(noisy)

    # Convert ternary vector to a probability distribution
    # Map {0,1,2} -> {0,0.5,1} then normalise
    ternary_mapped = ternary_vector.astype(float) / 2.0
    q = _normalize_to_simplex(ternary_mapped)

    # KL divergence; guard against zero entries
    eps = np.finfo(float).eps
    kl = np.sum(p_noisy * np.log((p_noisy + eps) / (q + eps)))
    return float(kl)

def hybrid_score(actions: List[MathAction],
                 model: ModelTier,
                 epsilon: float = 1.0) -> float:
    """
    Produce a single scalar that fuses:
      * Expected value, cost and risk of actions (Parent A)
      * Reconstruction risk derived from model metadata (Parent B)
      * Privacy‑aware KL divergence between an action‑derived probability
        vector and a ternary signature built from the model tier.

    The function demonstrates the mathematical bridge by feeding the
    DP‑perturbed KL term into a weighted sum with the action utilities.
    """
    # 1. Build a probability distribution from actions' expected values
    exp_vals = np.array([a.expected_value for a in actions])
    prob_dist = _normalize_to_simplex(exp_vals)

    # 2. Build a ternary vector from the model tier:
    #    T1 -> 0, T2 -> 1, T3 -> 2 (simple encoding)
    tier_map = {"T1": 0, "T2": 1, "T3": 2}
    ternary_val = tier_map.get(model.tier, 0)
    ternary_vec = np.full_like(prob_dist, ternary_val)

    # 3. Compute privacy‑aware KL divergence
    kl = privacy_aware_kl(prob_dist, ternary_vec, epsilon=epsilon)

    # 4. Compute reconstruction risk based on a synthetic quasi‑identifier count
    #    (here we use number of actions as a proxy for uniqueness)
    risk = reconstruction_risk_score(unique_quasi_identifiers=len(actions),
                                    total_records=100)  # fixed denominator for demo

    # 5. Aggregate everything
    action_score = sum(a.expected_value - a.cost - a.risk for a in actions)
    hybrid = action_score - 10.0 * kl - 5.0 * risk  # weighting chosen for illustration
    return float(hybrid)

def hybrid_operation(probabilities: np.ndarray,
                     ternary_vector: np.ndarray,
                     signatures: np.ndarray,
                     schedule: np.ndarray,
                     epsilon: float = 1.0) -> float:
    """
    Full hybrid pipeline merging both parents:

    1. Apply `prune_candidates` to the signature matrix.
    2. Compute a privacy‑aware KL divergence between `probabilities` and
       `ternary_vector`.
    3. Aggregate the pruned signatures with the KL term using a DP‑noised sum.
    4. Return the final scalar.

    This mirrors the original `hybrid_operation` of Parent A while inserting
    the DP mechanisms of Parent B.
    """
    # Step 1 – mask signatures
    pruned = prune_candidates(signatures, schedule)

    # Step 2 – privacy‑aware KL
    kl = privacy_aware_kl(probabilities, ternary_vector, epsilon=epsilon)

    # Step 3 – DP‑noised aggregation of the flattened pruned matrix
    flat = pruned.ravel()
    agg = dp_aggregate(flat, epsilon=epsilon)

    # Combine: we treat KL as a penalty on the aggregated value
    result = agg - 20.0 * kl
    return float(result)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy actions
    actions = [
        MathAction(id="a1", expected_value=10.0, cost=2.0, risk=0.5),
        MathAction(id="a2", expected_value=5.0, cost=1.0, risk=0.2),
        MathAction(id="a3", expected_value=8.0, cost=1.5, risk=0.3),
    ]

    # Dummy model tier
    model = ModelTier(name="model_X", ram_mb=1500, tier="T2")

    # Compute hybrid score
    score = hybrid_score(actions, model, epsilon=0.8)
    print(f"Hybrid score (actions + model): {score:.4f}")

    # Prepare inputs for hybrid_operation
    probs = np.array([0.4, 0.35, 0.25])
    ternary = np.array([1, 1, 1])  # maps to 0.5 after division by 2
    signatures = np.random.rand(3, 3)
    schedule = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [1, 1, 0]])

    result = hybrid_operation(probs, ternary, signatures, schedule, epsilon=0.8)
    print(f"Hybrid operation result: {result:.4f}")

    # Demonstrate feature expansion
    X = np.random.rand(5, 4)
    transformed = lead_lag_transform(X)
    print(f"Lead‑lag transformed shape: {transformed.shape}")

    # Demonstrate Kan basis
    basis = kan_basis(8)
    print(f"Kan basis (len=8): {basis}")