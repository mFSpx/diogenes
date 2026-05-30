# DARWIN HAMMER — match 181, survivor 3
# gen: 3
# parent_a: jepa_energy.py (gen0)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s1.py (gen2)
# born: 2026-05-29T23:26:04Z

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelTier:
    """Immutable descriptor of a model tier."""
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


# ----------------------------------------------------------------------
# Model pool with variational free‑energy based management
# ----------------------------------------------------------------------
class ModelPool:
    """
    Manages a pool of loaded models under a RAM ceiling.
    Uses a variational free‑energy (VFE) surrogate to decide loading/eviction.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb: int = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._vfe: float = 0.0  # variational free energy (lower is better)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _vfe_penalty(self, delta: float) -> None:
        """Add a penalty (or reward if delta < 0) to the VFE."""
        self._vfe += delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def free_energy(self) -> float:
        """Return the current variational free energy."""
        return self._vfe

    def add_model(self, model: ModelTier) -> None:
        """
        Add a model without eviction. Penalises tier conflicts and RAM overflow.
        """
        # Tier conflict: T3 cannot co‑exist with any T2 model
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._vfe_penalty(1e10)  # heavy penalty
        # RAM overflow penalty (soft)
        if model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            self._vfe_penalty(1e6)
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        """
        Load a model, rewarding the action.
        """
        self._vfe_penalty(-1e4)  # reward for successful load
        self.add_model(model)

    def _evict_one(self) -> None:
        """
        Evict the model with the highest contribution to VFE.
        Simple heuristic: evict the model with largest (ram_mb / (1+tier_score)).
        """
        if not self.loaded:
            return
        def tier_score(t: str) -> int:
            return {"T1": 1, "T2": 2, "T3": 3}.get(t, 0)

        worst = max(
            self.loaded.values(),
            key=lambda m: m.ram_mb / (1 + tier_score(m.tier)),
        )
        del self.loaded[worst.name]
        self._vfe_penalty(-1e3)  # reward for eviction

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Load a model, evicting as few models as needed to respect the RAM ceiling.
        Eviction decisions are guided by the VFE surrogate.
        """
        while model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            self._evict_one()
        self.load(model)


# ----------------------------------------------------------------------
# Differential privacy utilities
# ----------------------------------------------------------------------
def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """
    Laplace mechanism for sum aggregation.
    """
    total = sum(values)
    noise = np.random.laplace(0.0, scale=sensitivity / epsilon)
    return total + noise


# ----------------------------------------------------------------------
# Risk and collapse utilities
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Simple normalized risk: proportion of unique quasi‑identifiers.
    """
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def collapse_check(representations: Sequence[np.ndarray], threshold: float = 1e-5) -> bool:
    """
    Detect representation collapse by measuring variance across a batch.
    Returns True if variance falls below `threshold`.
    """
    if not representations:
        return True
    stacked = np.stack(representations)
    variance = np.var(stacked, axis=0).mean()
    return variance < threshold


# ----------------------------------------------------------------------
# Core JEPA‑Darwin‑Hammer energy and loss
# ----------------------------------------------------------------------
def jepa_darwin_hammer_energy(
    encoded_observation: np.ndarray,
    predicted_representation: np.ndarray,
    model: ModelTier,
    pool: ModelPool,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> float:
    """
    Compute a unified energy term:
      • Representation error (L2 norm)
      • Reconstruction risk (privacy‑aware)
      • Model‑specific free‑energy contribution
      • Differential‑privacy‑noised aggregate term
    """
    # 1. Representation error
    rep_error = np.linalg.norm(encoded_observation - predicted_representation)

    # 2. Privacy‑aware reconstruction risk
    # Here we treat the model name length as a proxy for quasi‑identifier count.
    risk = reconstruction_risk_score(len(model.name), total_records=1000)

    # 3. Model‑specific free‑energy contribution
    # Lower RAM usage and lower pool VFE are desirable.
    model_energy = model.ram_mb * 0.01 + 0.001 * pool.free_energy()

    # 4. DP‑noised aggregate of error and RAM usage
    dp_term = dp_aggregate([rep_error, model.ram_mb], epsilon=epsilon, sensitivity=sensitivity)

    return rep_error + risk + model_energy + dp_term


def jepa_darwin_hammer_loss_batch(
    models: Sequence[ModelTier],
    encoded_observations: Sequence[np.ndarray],
    predicted_representations: Sequence[np.ndarray],
    pool: ModelPool,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> float:
    """
    Batch loss averaging over all supplied models/observations.
    """
    if not (len(models) == len(encoded_observations) == len(predicted_representations)):
        raise ValueError("All input sequences must have the same length.")
    energies = [
        jepa_darwin_hammer_energy(
            enc,
            pred,
            mdl,
            pool,
            epsilon=epsilon,
            sensitivity=sensitivity,
        )
        for mdl, enc, pred in zip(models, encoded_observations, predicted_representations)
    ]
    return float(np.mean(energies))


# ----------------------------------------------------------------------
# Example usage (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a model pool
    pool = ModelPool(ram_ceiling_mb=6000)

    # Create a few example models
    models = [
        ModelTier(name="alpha", ram_mb=1024, tier="T2"),
        ModelTier(name="beta", ram_mb=2048, tier="T1"),
        ModelTier(name="gamma", ram_mb=512, tier="T3"),
    ]

    # Load models with eviction policy
    for m in models:
        pool.load_with_eviction(m)

    print("Current VFE:", pool.free_energy())
    print("RAM used (MB):", pool._used_ram())

    # Simulate a batch of observations
    batch_size = 4
    encoded = [np.random.rand(16) for _ in range(batch_size)]
    predicted = [np.random.rand(16) for _ in range(batch_size)]
    batch_models = random.choices(models, k=batch_size)

    loss = jepa_darwin_hammer_loss_batch(
        models=batch_models,
        encoded_observations=encoded,
        predicted_representations=predicted,
        pool=pool,
        epsilon=0.8,
        sensitivity=1.0,
    )
    print("Batch loss:", loss)

    # Collapse detection on the predicted batch
    collapsed = collapse_check(predicted, threshold=1e-6)
    print("Representation collapse detected:", collapsed)