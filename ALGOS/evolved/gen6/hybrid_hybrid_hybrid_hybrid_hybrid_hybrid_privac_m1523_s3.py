# DARWIN HAMMER — match 1523, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m761_s0.py (gen5)
# parent_b: hybrid_hybrid_privacy_model_model_vram_scheduler_m14_s1.py (gen2)
# born: 2026-05-29T23:37:11Z

"""Hybrid NLMS‑Risk Scheduler
================================

This module fuses two parent algorithms:

* **Parent A** – an NLMS adaptive filter that updates a weight vector `w`
  using a normalized learning rate `μ`.  The update rule is  

  
  e   = d – w·x
  μ'  = μ / (ε + ‖x‖²)
  w'  = w + μ'·e·x
  

* **Parent B** – a privacy‑risk model that computes a *reconstruction risk
  score* `r ∈ [0,1]` from the number of quasi‑identifiers in a dataset and
  uses that score to decide whether a model can be loaded into VRAM.

The **mathematical bridge** is the risk score `r`.  It is used

1. **as a scaling factor for the NLMS learning rate** – a high risk makes the
   filter adapt more conservatively:

   `μ_eff = μ_base * (1 – r)`

2. **as a Bayesian‑like prior in the VRAM‑scheduling decision** – the posterior
   probability of loading a candidate model is obtained from the NLMS prediction
   of future VRAM usage `v̂` and the risk‑adjusted tolerance `V_max·(1‑r)`:

   `p_load = sigmoid( (V_max·(1‑r) – v̂) / τ )`

   where `τ` is a temperature controlling stochasticity.

The three core functions below illustrate this hybrid behaviour:
`nlms_update_with_risk`, `predict_vram_usage`, and `schedule_models`.  A
light‑weight `ModelPool` class from the privacy parent is retained to keep the
resource‑tracking logic.

"""

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

    for cand in candidates:
        # 1. feature vector
        x = feature_builder(pool, cand)

        # 2. NLMS prediction of VRAM usage after loading
        v_pred = predict_vram_usage(weights, x)

        # 3. Simulated risk score (in real code this would come from data)
        uq = random.randint(0, 10)          # placeholder: number of quasi‑ids
        tot = random.randint(10, 100)       # placeholder: total records
        risk = reconstruction_risk_score(uq, tot)

        # 4. Posterior loading probability
        tolerance = V_max * (1.0 - risk)
        p_load = sigmoid((tolerance - v_pred) / temperature)

        # 5. Stochastic decision
        if random.random() < p_load and pool.can_load(cand):
            try:
                pool.load(cand)
                loaded = True
            except RuntimeError:
                loaded = False
        else:
            loaded = False

        # 6. Observe actual VRAM usage (target) and update NLMS
        actual_vram = pool._used_vram()
        weights, _ = nlms_update_with_risk(
            weights,
            x,
            target=actual_vram,
            base_mu=base_mu,
            risk=risk,
        )

    return pool, weights


# ----------------------------------------------------------------------
# Helper to build a feature vector from the current pool state and a candidate
# ----------------------------------------------------------------------
def default_feature_builder(pool: ModelPool, candidate: ModelTier) -> np.ndarray:
    """
    Construct a simple numeric feature vector:

    [ current_used_vram,
      candidate_vram,
      candidate_ram,
      num_loaded_models,
      risk_placeholder ]
    """
    used_vram = pool._used_vram()
    num_loaded = len(pool.loaded)
    # placeholder risk – will be overridden by the scheduler anyway
    placeholder_risk = 0.0
    return np.array([
        used_vram,
        candidate.vram_mb,
        candidate.ram_mb,
        num_loaded,
        placeholder_risk,
    ], dtype=float)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a tiny NLMS filter (5 features → 5 weights)
    np.random.seed(0)
    w = np.random.randn(5)

    # Create a model pool with modest ceilings
    pool = ModelPool(ram_ceiling_mb=12000, vram_ceiling_mb=8192)

    # Define a few candidate models
    candidates = [
        ModelTier("tiny-1b", 800, "T1", 1024),
        ModelTier("mid-2b", 2500, "T2", 2048),
        ModelTier("big-5b", 6000, "T3", 4096),
    ]

    # Run the hybrid scheduler
    updated_pool, updated_w = schedule_models(
        pool,
        candidates,
        w,
        default_feature_builder,
        base_mu=0.6,
        temperature=5.0,
    )

    # Print results – should run without exception
    print("Final pool status:")
    print(updated_pool.status())
    print("Final NLMS weights:", updated_w)