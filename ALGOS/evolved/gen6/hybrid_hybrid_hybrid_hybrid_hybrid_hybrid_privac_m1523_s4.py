# DARWIN HAMMER — match 1523, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m761_s0.py (gen5)
# parent_b: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (gen2)
# born: 2026-05-29T23:37:11Z

import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple, List, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A: NLMS core
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Return the dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Base learning rate in (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (new_weights, error) : Tuple[np.ndarray, float]
        Updated weight vector and the instantaneous error `e = target – w·x`.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm_sq = float(x @ x) + eps
    mu_eff = mu / norm_sq
    new_weights = weights + mu_eff * error * x
    return new_weights, error


# ----------------------------------------------------------------------
# Parent‑B: privacy‑risk helpers and model pool
# ----------------------------------------------------------------------
class ModelTier:
    """Simple immutable descriptor for a model."""
    __slots__ = ("name", "ram_mb", "tier", "vram_mb")

    def __init__(self, name: str, ram_mb: int, tier: str, vram_mb: int):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier
        self.vram_mb = vram_mb

    def __repr__(self) -> str:
        return f"ModelTier({self.name!r}, RAM={self.ram_mb}MB, VRAM={self.vram_mb}MB, tier={self.tier})"


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk ∈ [0,1] proportional to the fraction of quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def anonymize_for_indexing(record: Dict[str, Any], redact_keys: set[str] | None = None) -> Dict[str, Any]:
    """Redact sensitive keys before indexing."""
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}


class ModelPool:
    """Tracks loaded models and enforces RAM/VRAM ceilings."""
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def can_load(self, model: ModelTier) -> bool:
        """Return True if loading `model` respects both ceilings."""
        return (self._used_ram() + model.ram_mb <= self.ram_ceiling_mb) and \
               (self._used_vram() + model.vram_mb <= self.vram_ceiling_mb)

    def load(self, model: ModelTier) -> None:
        """Load a model, raising if constraints are violated."""
        if not self.can_load(model):
            raise RuntimeError(f"Cannot load {model.name}: resource limits exceeded.")
        # T3/T2 mutual‑exclusion rule (from parent B)
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 models are mutually exclusive with any T2 model.")
        self.loaded[model.name] = model

    def evict(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)

    def status(self) -> str:
        return (f"RAM: {self._used_ram()}/{self.ram_ceiling_mb} MB, "
                f"VRAM: {self._used_vram()}/{self.vram_ceiling_mb} MB, "
                f"Loaded: {list(self.loaded.keys())}")


# ----------------------------------------------------------------------
# Hybrid layer: risk‑aware NLMS and VRAM scheduling
# ----------------------------------------------------------------------
def nlms_update_with_risk(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    base_mu: float,
    risk: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    NLMS update where the effective learning rate is attenuated by the
    reconstruction risk `risk ∈ [0,1]`.

    μ_eff = base_mu * (1 – risk)
    """
    mu_eff = base_mu * (1.0 - risk)
    return nlms_update(weights, x, target, mu=mu_eff, eps=eps)


def predict_vram_usage(
    weights: np.ndarray,
    feature_vec: np.ndarray,
) -> float:
    """
    Use the NLMS weights as a linear model that predicts *future* VRAM usage
    (in MB) given a feature vector describing the system state.
    """
    return max(0.0, nlms_predict(weights, feature_vec))


def sigmoid(z: float) -> float:
    """Standard logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-z))


def schedule_models(
    pool: ModelPool,
    candidates: List[ModelTier],
    weights: np.ndarray,
    feature_builder,
    base_mu: float = 0.5,
    temperature: float = 10.0,
) -> Tuple[ModelPool, np.ndarray]:
    """
    Decide which candidate models to load.

    For each candidate:
        1. Compute a feature vector `x` via ``feature_builder(pool, candidate)``.
        2. Predict the VRAM usage after loading the candidate:
               v̂ = predict_vram_usage(weights, x)
        3. Obtain the reconstruction risk `r` from the candidate's
           quasi‑identifier statistics (simulated here as a random integer).
        4. Compute a posterior loading probability
               p = sigmoid( (V_max·(1‑r) – v̂) / temperature )
        5. Stochastically load the model with probability `p`.
        6. After the decision, update the NLMS weights using the observed
           outcome (target = actual VRAM usage after the decision) and the same
           risk‑scaled learning rate.

    The function returns the updated ``ModelPool`` and the new weight vector.
    """
    V_max = pool.vram_ceiling_mb
    new_pool = ModelPool(ram_ceiling_mb=pool.ram_ceiling_mb, vram_ceiling_mb=pool.vram_ceiling_mb)
    new_weights = weights.copy()

    for cand in candidates:
        # 1. feature vector
        x = feature_builder(pool, cand)

        # 2. NLMS prediction of VRAM usage after loading
        v_pred = predict_vram_usage(new_weights, x)

        # 3. Simulated risk 
        quasi_identifiers = random.randint(0, 100)
        total_records = 100
        risk = reconstruction_risk_score(quasi_identifiers, total_records)

        # 4. Compute a posterior loading probability
        p_load = sigmoid((V_max * (1 - risk) - v_pred) / temperature)

        # 5. Stochastically load the model with probability `p_load`
        if random.random() < p_load:
            try:
                new_pool.load(cand)
                # 6. Update NLMS weights with observed outcome (target = actual VRAM usage)
                target_vram_usage = new_pool._used_vram()
                new_weights, _ = nlms_update_with_risk(new_weights, x, target_vram_usage, base_mu, risk)
            except RuntimeError:
                pass

    return new_pool, new_weights