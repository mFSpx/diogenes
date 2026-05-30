# DARWIN HAMMER — match 7, survivor 2
# gen: 1
# parent_a: privacy.py (gen0)
# parent_b: model_pool.py (gen0)
# born: 2026-05-29T23:15:49Z

"""Hybrid module combining privacy scoring (privacy.py) and model resource management (model_pool.py).

Mathematical Bridge:
Both parent algorithms can be expressed as linear systems:

1. Privacy side: For a set of records we compute a risk vector **r** ∈ ℝⁿ where each entry
   r_i = reconstruction_risk_score(u_i, N) ∈ [0,1].

2. Model side: Each loaded model contributes a resource vector **c** ∈ ℝᵐ
   (RAM consumption, tier‑exclusivity penalty, etc.).

The fusion treats privacy risk as an additional *soft* resource that must be
allocated together with RAM.  We form a combined resource matrix **A** whose rows
are models and columns are [RAM, privacy‑load].  The privacy‑load for a model
*m* is defined as:

    p(m) = α * tier_factor(m.tier) * mean(r)

where α is a scaling constant and tier_factor maps tiers to numeric
sensitivity (e.g., T1=1, T2=2, T3=3).  The total load for a selection vector
**x** (binary indicator of loaded models) is:

    L = Aᵀ · x

We then enforce the composite constraint L ≤ [ram_ceiling, privacy_budget].
This linear formulation fuses the core topologies of both parents into a single
matrix‑based decision process.

The module provides three high‑level functions that demonstrate this hybrid
operation:
    - privacy_risk_vector(...)
    - model_resource_matrix(...)
    - select_models_hybrid(...)
"""

from __future__ import annotations

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict, Set, Tuple

import numpy as np

# ---------- Parent A: privacy utilities ----------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def anonymize_for_indexing(record: dict[str, Any], redact_keys: Set[str] | None = None) -> dict[str, Any]:
    """Redact sensitive fields for indexing."""
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Deterministic sum; noise can be added externally."""
    return sum(values)

# ---------- Parent B: model pool utilities ----------
class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded due to policy or resource limits."""
    pass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g., "T1", "T2", "T3"

# Example tiers (could be extended)
TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3")

class ModelPool:
    """Manages loading of models under RAM ceiling and tier exclusivity."""
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
            # Evict the first inserted model (FIFO policy)
            evicted_name = next(iter(self.loaded))
            self.loaded.pop(evicted_name)
        self.load(model)

# ---------- Hybrid Functions ----------
def privacy_risk_vector(records: List[Dict[str, Any]],
                        quasi_id_key: str = "quasi_id") -> np.ndarray:
    """
    Compute a risk vector r where each entry is the reconstruction risk for a record.
    The function assumes each record contains the number of unique quasi‑identifiers.
    """
    total = len(records)
    risks = [reconstruction_risk_score(rec.get(quasi_id_key, 0), total) for rec in records]
    return np.array(risks, dtype=float)


def model_resource_matrix(models: List[ModelTier],
                          privacy_risk: np.ndarray,
                          alpha: float = 0.5) -> np.ndarray:
    """
    Build the combined resource matrix A (shape: len(models) × 2).

    Column 0 : RAM consumption (MB).
    Column 1 : Privacy‑load = α * tier_factor * mean(privacy_risk).

    tier_factor maps tier strings to numeric sensitivity:
        T1 → 1, T2 → 2, T3 → 3.
    """
    if privacy_risk.size == 0:
        mean_risk = 0.0
    else:
        mean_risk = float(privacy_risk.mean())

    tier_factor = {"T1": 1.0, "T2": 2.0, "T3": 3.0}
    rows = []
    for m in models:
        ram = float(m.ram_mb)
        pf = tier_factor.get(m.tier, 1.0)
        privacy_load = alpha * pf * mean_risk
        rows.append([ram, privacy_load])
    return np.array(rows, dtype=float)


def select_models_hybrid(models: List[ModelTier],
                         privacy_risk: np.ndarray,
                         ram_ceiling_mb: int = 6000,
                         privacy_budget: float = 1.0,
                         alpha: float = 0.5) -> List[ModelTier]:
    """
    Greedy selection based on the hybrid linear constraint:
        Aᵀ·x ≤ [ram_ceiling, privacy_budget]

    Returns the list of models that can be loaded without violating either bound.
    The algorithm respects the T3‑vs‑T2 exclusivity rule from the original ModelPool.
    """
    A = model_resource_matrix(models, privacy_risk, alpha=alpha)  # shape (k,2)
    selected: List[ModelTier] = []
    used_ram = 0.0
    used_privacy = 0.0
    has_T2 = False
    has_T3 = False

    # Sort models by increasing RAM to favor smaller footprints
    sorted_models = sorted(zip(models, A), key=lambda pair: pair[1][0])

    for model, resources in sorted_models:
        ram, priv = resources
        # Tier exclusivity check
        if model.tier == "T3" and has_T2:
            continue
        if model.tier == "T2" and has_T3:
            continue

        if used_ram + ram <= ram_ceiling_mb and used_privacy + priv <= privacy_budget:
            selected.append(model)
            used_ram += ram
            used_privacy += priv
            if model.tier == "T2":
                has_T2 = True
            if model.tier == "T3":
                has_T3 = True

    return selected


def dp_aggregate_hybrid(values: Iterable[float],
                        selected_models: List[ModelTier],
                        epsilon: float = 1.0,
                        sensitivity: float = 1.0) -> float:
    """
    Perform a differentially private aggregation where the noise scale is
    adapted to the total RAM of the selected models.

    Effective epsilon = epsilon / (1 + total_ram / 1000)
    """
    total_ram = sum(m.ram_mb for m in selected_models)
    effective_epsilon = epsilon / (1.0 + total_ram / 1000.0)
    # Laplace noise (using standard library random)
    scale = sensitivity / max(effective_epsilon, 1e-9)
    noise = random.laplacevariate(0.0, scale) if hasattr(random, "laplacevariate") else random.gauss(0.0, scale)
    return dp_aggregate(values) + noise


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create dummy records
    dummy_records = [
        {"id": 1, "quasi_id": random.randint(0, 5)},
        {"id": 2, "quasi_id": random.randint(0, 5)},
        {"id": 3, "quasi_id": random.randint(0, 5)},
        {"id": 4, "quasi_id": random.randint(0, 5)},
        {"id": 5, "quasi_id": random.randint(0, 5)},
    ]

    # Compute privacy risk vector
    risk_vec = privacy_risk_vector(dummy_records, quasi_id_key="quasi_id")
    print("Privacy risk vector:", risk_vec)

    # Define model catalog
    catalog = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]

    # Hybrid selection
    chosen = select_models_hybrid(catalog, risk_vec, ram_ceiling_mb=8000, privacy_budget=0.8, alpha=0.3)
    print("Selected models:", [m.name for m in chosen])

    # Perform a DP aggregation on some numbers
    numbers = [10.0, 20.5, 30.2]
    agg = dp_aggregate_hybrid(numbers, chosen, epsilon=1.0, sensitivity=1.0)
    print("DP aggregated result (with hybrid noise):", agg)

    # Verify that ModelPool can load the same selection without violating original rules
    pool = ModelPool(ram_ceiling_mb=8000)
    for m in chosen:
        try:
            pool.load(m)
        except ModelLoadError as e:
            print(f"Failed to load {m.name}: {e}", file=sys.stderr)
    print("Models loaded in pool:", list(pool.loaded.keys()))