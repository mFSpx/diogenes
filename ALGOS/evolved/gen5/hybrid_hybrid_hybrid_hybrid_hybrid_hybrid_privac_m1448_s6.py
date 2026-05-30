# DARWIN HAMMER — match 1448, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s5.py (gen4)
# parent_b: hybrid_hybrid_privacy_model_hybrid_serpentina_se_m179_s1.py (gen2)
# born: 2026-05-29T23:36:25Z

"""Hybrid Algorithm Fusion of DARWIN HAMMER Parents A and B.

Parent A provides feature transformations, signature auditing, and a hybrid
operation based on probability distributions (KL divergence).  
Parent B contributes a differential‑privacy aggregation routine, a reconstruction
risk score, and a resource‑constrained model pool.

Mathematical Bridge:
- The reconstruction risk score (unique_quasi_identifiers / total_records) is
  used to modulate the pruning schedule of candidate signatures from Parent A.
- The probability vector supplied to the hybrid operation is first passed
  through a differentially‑private aggregation (Laplace noise) from Parent B.
- The resulting noisy probabilities are then combined with the ternary vector
  and the (risk‑weighted) schedule to compute a KL‑divergence‑based score.

The fused system therefore intertwines privacy‑preserving noise, risk‑aware
resource management, and the original hybrid probability‑signature calculus
into a single unified workflow.
"""

import sys
import math
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Iterable, Tuple, Dict

# ----------------------------------------------------------------------
# Shared data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Model pool and privacy utilities (from Parent B)
# ----------------------------------------------------------------------
class ModelLoadError(RuntimeError):
    pass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the oldest loaded model (FIFO)
            oldest_key = next(iter(self.loaded))
            self.loaded.pop(oldest_key)
        self.load(model)

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk score in [0,1] representing reconstruction likelihood."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Differentially‑private sum using Laplace mechanism."""
    total = sum(values)
    noise = np.random.laplace(0.0, scale=sensitivity / epsilon)
    return total + noise

# ----------------------------------------------------------------------
# Core transformations (from Parent A)
# ----------------------------------------------------------------------
def lead_lag_transform(X: np.ndarray) -> np.ndarray:
    """
    Concatenates linear (sum) and quadratic (sum of squares) features across rows.
    """
    linear_features = np.sum(X, axis=1, keepdims=True)
    quadratic_features = np.sum(X ** 2, axis=1, keepdims=True)
    return np.concatenate((linear_features, quadratic_features), axis=1)

def kan_basis(grid_size: int) -> np.ndarray:
    """
    Generates a KAN‑style exponential basis vector of length `grid_size`.
    """
    points = np.linspace(0, 1, grid_size)
    return np.exp(-points)

def prune_candidates(signatures: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """
    Element‑wise multiplication that zeroes out signatures according to schedule.
    """
    return signatures * schedule

def audit_signature(candidates: List[str]) -> np.ndarray:
    """
    One‑hot encodes a list of classification strings into a matrix.
    """
    CLASSIFICATIONS = [
        "usable_now",
        "research_only",
        "needs_conversion",
        "unsafe_for_fastpath",
        "unsupported",
    ]
    idx_map = {c: i for i, c in enumerate(CLASSIFICATIONS)}
    one_hot = np.zeros((len(candidates), len(CLASSIFICATIONS)))
    for row, cand in enumerate(candidates):
        col = idx_map.get(cand, None)
        if col is not None:
            one_hot[row, col] = 1.0
    return one_hot

# ----------------------------------------------------------------------
# Hybrid fused operations
# ----------------------------------------------------------------------
def hybrid_operation_fused(
    probabilities: np.ndarray,
    ternary_vector: np.ndarray,
    signatures: np.ndarray,
    schedule: np.ndarray,
    unique_quasi_identifiers: int,
    total_records: int,
    epsilon: float = 1.0,
) -> float:
    """
    Combined hybrid operation:

    1. Apply differential privacy to the probability vector (DP sum then re‑normalize).
    2. Compute a risk‑weighted schedule using reconstruction_risk_score.
    3. Prune signatures with the risk‑weighted schedule.
    4. Compute KL‑divergence between the noisy probability distribution and
       the ternary vector (treated as a distribution after normalization).
    """
    # Step 1: DP aggregate and renormalize
    noisy_sum = dp_aggregate(probabilities, epsilon=epsilon, sensitivity=1.0)
    # Avoid division by zero
    if noisy_sum == 0:
        noisy_probs = np.full_like(probabilities, 1.0 / len(probabilities))
    else:
        noisy_probs = probabilities / noisy_sum

    # Step 2: Risk‑weighted schedule
    risk = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    weighted_schedule = schedule * (1.0 - risk)  # higher risk -> stricter pruning

    # Step 3: Prune signatures
    pruned = prune_candidates(signatures, weighted_schedule)

    # Step 4: KL divergence (add epsilon to avoid log(0))
    eps = 1e-12
    ternary_norm = ternary_vector / (np.sum(ternary_vector) + eps)
    kl = np.sum(noisy_probs * np.log((noisy_probs + eps) / (ternary_norm + eps)))

    # Return a scalar that blends KL with a simple norm of pruned signatures
    signature_norm = np.linalg.norm(pruned)
    return kl + 0.01 * signature_norm

def load_and_prepare_models(
    model_descriptions: List[Tuple[str, int, str]],
    pool: ModelPool,
    epsilon: float = 1.0,
) -> List[ModelTier]:
    """
    Loads a list of model descriptors into the ModelPool using privacy‑aware
    loading. The function returns the list of successfully loaded ModelTier
    objects.

    Each descriptor is a tuple (name, ram_mb, tier).
    """
    loaded_models = []
    for name, ram_mb, tier in model_descriptions:
        model = ModelTier(name=name, ram_mb=ram_mb, tier=tier)
        try:
            # Simulate privacy cost: we refuse loading if risk > epsilon threshold
            risk = reconstruction_risk_score(ram_mb, pool.ram_ceiling_mb)
            if risk > epsilon:
                # Skip loading, treat as privacy‑blocked
                continue
            pool.load_with_eviction(model)
            loaded_models.append(model)
        except ModelLoadError:
            # If loading fails, attempt eviction and retry once
            try:
                pool.load_with_eviction(model)
                loaded_models.append(model)
            except ModelLoadError:
                # Give up on this model
                continue
    return loaded_models

def compute_lead_lag_kan_features(X: np.ndarray, grid_size: int = 10) -> np.ndarray:
    """
    Generates a composite feature matrix:
    - Apply lead‑lag transform (linear + quadratic).
    - Append a KAN basis vector replicated for each row.
    """
    transformed = lead_lag_transform(X)                     # shape (n,2)
    basis = kan_basis(grid_size)                           # shape (grid_size,)
    # Replicate basis for each sample and concatenate
    basis_rep = np.tile(basis, (X.shape[0], 1))            # shape (n,grid_size)
    return np.concatenate((transformed, basis_rep), axis=1)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple data for testing
    X = np.array([[1.0, 2.0, 3.0],
                  [0.5, 1.5, 2.5]])

    probs = np.array([0.2, 0.5, 0.3])
    ternary = np.array([0.0, 1.0, 0.0])
    signatures = np.array([1.0, 0.8, 0.6])
    schedule = np.array([1.0, 0.5, 0.2])

    # Compute fused hybrid operation
    score = hybrid_operation_fused(
        probabilities=probs,
        ternary_vector=ternary,
        signatures=signatures,
        schedule=schedule,
        unique_quasi_identifiers=42,
        total_records=1000,
        epsilon=0.8,
    )
    print(f"Hybrid fused score: {score}")

    # Model pool test
    pool = ModelPool(ram_ceiling_mb=2000)
    models_desc = [
        ("model_A", 500, "T1"),
        ("model_B", 800, "T2"),
        ("model_C", 900, "T3"),
    ]
    loaded = load_and_prepare_models(models_desc, pool, epsilon=0.3)
    print(f"Loaded models: {[m.name for m in loaded]}")

    # Feature generation test
    feats = compute_lead_lag_kan_features(X, grid_size=5)
    print(f"Feature matrix shape: {feats.shape}")
    print(feats)