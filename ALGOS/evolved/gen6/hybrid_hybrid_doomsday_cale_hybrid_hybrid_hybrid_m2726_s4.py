# DARWIN HAMMER — match 2726, survivor 4
# gen: 6
# parent_a: hybrid_doomsday_calendar_hybrid_rlct_grokking_m396_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1175_s1.py (gen5)
# born: 2026-05-29T23:43:56Z

import numpy as np
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
NodeId = str
Edge = Tuple[NodeId, NodeId, int]          # (src, dst, impedance)

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def doomsday_rule(year: int, month: int, day: int) -> int:
    """
    Classic Doomsday algorithm: returns the weekday index
    (0 = Sunday, …, 6 = Saturday) for the supplied Gregorian date.
    """
    # Python's weekday(): Monday=0 … Sunday=6 → shift to Sunday=0
    return (date(year, month, day).weekday() + 1) % 7


def days_until_next_doomsday(ref: date = None) -> int:
    """
    Returns the number of days from ``ref`` (or today) to the next
    Doomsday (the day of the week that repeats each year for a given
    month/day).  This metric is used to adapt the learning rate:
    the closer we are to a Doomsday, the smaller the step size,
    encouraging stability around “critical” dates.
    """
    if ref is None:
        ref = date.today()
    today_doomsday = doomsday_rule(ref.year, ref.month, ref.day)
    # The Doomsday for the current year is the weekday of 4/4, 6/6, 8/8, 10/10, 12/12
    # (or 5/9 for odd months).  We compute the weekday for 4/4 as a proxy.
    year_doomsday = doomsday_rule(ref.year, 4, 4)
    delta = (year_doomsday - today_doomsday) % 7
    # If today is already a Doomsday, we look to the next one (7 days later)
    return 7 if delta == 0 else delta


def adaptive_learning_rate(base_mu: float = 0.5) -> float:
    """
    Scales ``base_mu`` by a factor that decays the closer we are to the next
    Doomsday.  The factor lies in (0.5, 1.0] and is smooth.
    """
    days = days_until_next_doomsday()
    # Map days∈[1,7] → factor∈[0.5,1.0] (farther → larger step)
    factor = 0.5 + 0.5 * (days - 1) / 6.0
    return base_mu * factor


def bayesian_information_criterion(log_likelihood: float,
                                   n_params: int,
                                   n_samples: int) -> float:
    """
    Standard BIC:  -2*logL + k*log(n)
    Lower values indicate a more parsimonious model.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return -2.0 * log_likelihood + n_params * math.log(n_samples)


def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Normalised LMS weight update.
    Returns the updated weight vector and the instantaneous error.
    """
    error = target - nlms_predict(weights, x)
    norm = float(x @ x) + eps
    weights = weights + (mu * error / norm) * x
    return weights, error


def anti_slop_ratio(claims_with_evidence: int,
                    total_claims_emitted: int) -> float:
    """
    Ratio of supported claims, clamped to [0, 1].
    """
    if total_claims_emitted <= 0:
        return 0.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


# ----------------------------------------------------------------------
# Model management
# ----------------------------------------------------------------------
class ModelTier:
    """
    Simple container for a model's resource profile and a trust score.
    ``trust`` should be in [0, 1] and reflects the linguistic‑similarity
    (LSM) measure from Parent B.
    """
    def __init__(self, name: str, ram_mb: int, tier: str, trust: float = 0.5):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier
        self.trust = max(0.0, min(1.0, trust))


class ModelPool:
    """
    Manages a collection of models under a RAM ceiling.
    Eviction decisions are guided by a BIC‑based utility score that
    combines model size, trust, and recent activation count.
    """
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.activation_counts: Dict[str, int] = {}   # from rlct_grokking

    # ------------------------------------------------------------------
    # Basic bookkeeping
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    # ------------------------------------------------------------------
    # Loading / eviction
    # ------------------------------------------------------------------
    def load(self, model: ModelTier) -> None:
        """
        Load a model if constraints allow.  Raises if the load would
        violate mutual‑exclusivity (T3 vs any T2) or the RAM ceiling.
        """
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 models cannot coexist with T2 models")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling would be exceeded")
        self.loaded[model.name] = model
        self.activation_counts.setdefault(model.name, 0)

    def load_with_eviction(self, model: ModelTier) -> None:
        """
        Load a model, evicting the least‑useful existing model(s) until
        there is enough space.  Utility is defined via ``_bic_score``.
        """
        while model.ram_mb + self._used() > self.ram_ceiling_mb:
            # Identify the model with the highest (worst) BIC score
            worst = max(self.loaded.values(),
                        key=lambda m: self._bic_score(m))
            self._evict(worst.name)
        self.load(model)

    def _evict(self, name: str) -> None:
        """Remove a model from the pool."""
        self.loaded.pop(name, None)
        self.activation_counts.pop(name, None)

    # ------------------------------------------------------------------
    # Trust‑weighted LSM score
    # ------------------------------------------------------------------
    def trust_weighted_lsm(self) -> float:
        """
        Returns a normalized trust‑weighted LSM score in [0, 1].
        It is the weighted average of each model's trust, using RAM
        usage as the weighting factor.
        """
        total_ram = self._used()
        if total_ram == 0:
            return 0.0
        weighted_sum = sum(m.trust * m.ram_mb for m in self.loaded.values())
        return weighted_sum / total_ram

    # ------------------------------------------------------------------
    # Activation pattern count (from rlct_grokking)
    # ------------------------------------------------------------------
    def record_activation(self, name: str) -> None:
        """Increment activation counter for a model."""
        if name in self.loaded:
            self.activation_counts[name] = self.activation_counts.get(name, 0) + 1

    # ------------------------------------------------------------------
    # BIC‑based utility for eviction
    # ------------------------------------------------------------------
    def _bic_score(self, model: ModelTier) -> float:
        """
        Compute a BIC‑like utility: lower is better.
        - log_likelihood ≈ trust * (1 + activation/total_activations)
        - n_params ≈ model.ram_mb (larger models are penalised)
        - n_samples ≈ total number of activations + 1 (avoid zero)
        """
        total_acts = sum(self.activation_counts.values()) + 1
        model_acts = self.activation_counts.get(model.name, 0)
        pseudo_loglik = model.trust * (1.0 + model_acts / total_acts)
        return bayesian_information_criterion(pseudo_loglik,
                                               n_params=model.ram_mb,
                                               n_samples=total_acts)

# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_predict(model_pool: ModelPool,
                   weights: np.ndarray,
                   x: np.ndarray) -> float:
    """
    Produce a prediction that blends:
    1. NLMS linear output.
    2. A trust‑weighted LSM scaling factor.
    3. An anti‑slop correction based on the proportion of
       “supported” claims inferred from model trust.
    """
    # 1️⃣ Adaptive learning‑rate factor (used later in update)
    #    – not directly needed for prediction but kept for symmetry.
    _ = adaptive_learning_rate()   # placeholder for symmetry

    # 2️⃣ Trust‑weighted LSM score (now meaningful)
    trust_lsm = model_pool.trust_weighted_lsm()          # ∈[0,1]

    # 3️⃣ Base NLMS prediction
    base_pred = nlms_predict(weights, x)

    # 4️⃣ Derive a pseudo “claims with evidence” count.
    #    We interpret trust as the probability that a claim is
    #    supported; scaling to a 0‑100 integer range.
    claims_with_evidence = int(trust_lsm * 100)
    total_claims_emitted = 100

    # 5️⃣ Anti‑slop ratio (smooths extreme trust values)
    anti_slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Final blended output
    return base_pred * (1.0 + anti_slop)


def hybrid_update(model_pool: ModelPool,
                  weights: np.ndarray,
                  x: np.ndarray,
                  target: float) -> Tuple[np.ndarray, float]:
    """
    Perform an NLMS weight update whose step size adapts to the Doomsday
    proximity and whose error is scaled by the anti‑slop factor.
    Additionally, record model activations to feed the BIC‑based eviction
    policy.
    """
    # Adaptive learning rate based on Doomsday proximity
    mu = adaptive_learning_rate(base_mu=0.5)

    # NLMS weight update
    new_weights, raw_error = nlms_update(weights, x, target, mu=mu)

    # Trust‑weighted LSM influences the error scaling
    trust_lsm = model_pool.trust_weighted_lsm()
    claims_with_evidence = int(trust_lsm * 100)
    anti_slop = anti_slop_ratio(claims_with_evidence, 100)

    # Scale error to reflect confidence in the current model set
    scaled_error = raw_error * (1.0 + anti_slop)

    # Record a generic activation for the “most‑trusted” model
    if model_pool.loaded:
        # Choose model with highest trust (break ties arbitrarily)
        best_model = max(model_pool.loaded.values(),
                         key=lambda m: m.trust)
        model_pool.record_activation(best_model.name)

    return new_weights, scaled_error


# ----------------------------------------------------------------------
# Example usage (self‑test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a pool with a few dummy models
    pool = ModelPool(ram_ceiling_mb=2000)
    pool.load(ModelTier("alpha", 300, "T1", trust=0.8))
    pool.load(ModelTier("beta",  400, "T2", trust=0.6))
    pool.load(ModelTier("gamma", 250, "T1", trust=0.9))

    # Random seed for reproducibility
    rng = np.random.default_rng(42)

    # Random initial weights & input vector
    w = rng.normal(scale=0.1, size=5)
    x_vec = rng.normal(size=5)

    # Target value (synthetic)
    y_target = 3.14

    # Perform a prediction‑update cycle
    pred_before = hybrid_predict(pool, w, x_vec)
    w, err = hybrid_update(pool, w, x_vec, y_target)
    pred_after = hybrid_predict(pool, w, x_vec)

    print(f"Prediction before update: {pred_before:.4f}")
    print(f"Updated weights: {w}")
    print(f"Scaled error: {err:.4f}")
    print(f"Prediction after update: {pred_after:.4f}")

    # Demonstrate eviction when RAM is scarce
    heavy = ModelTier("delta", 1800, "T3", trust=0.4)
    try:
        pool.load_with_eviction(heavy)
        print("Loaded heavy model after eviction.")
    except RuntimeError as exc:
        print("Failed to load heavy model:", exc)