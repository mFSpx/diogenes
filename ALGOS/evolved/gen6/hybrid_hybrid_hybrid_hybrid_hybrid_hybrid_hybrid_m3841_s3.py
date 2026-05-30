# DARWIN HAMMER — match 3841, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_decisi_m2236_s2.py (gen5)
# born: 2026-05-29T23:52:04Z

"""Hybrid Model Resource & Decision Engine

This module fuses two parent algorithms:

* **Parent A** – privacy‑aware reconstruction risk scoring and model
  resource (RAM/VRAM) management.
* **Parent B** – Hoeffding‑bound driven split decisions for streaming
  learners together with tropical algebra (t_add, t_mul) that mirrors the
  piecewise‑linear behaviour of ReLU networks.

**Mathematical bridge**

The reconstruction risk score `ρ ∈ [0,1]` (Parent A) is used as a
multiplicative weight on the gain values that feed the Hoeffding‑bound
split test (Parent B).  Tropical multiplication `t_mul(x, y) = x + y`
provides a natural way to combine a gain with the “privacy penalty”
`(1‑ρ)`.  The resulting privacy‑adjusted gains are then compared using
the Hoeffding bound; the decision is finally merged with endpoint health
(`h = 1‑ρ`) via tropical addition `t_add` to produce a unified work‑share
allocation score.

The three exported functions demonstrate this hybrid operation:
`reconstruction_risk_score`, `hybrid_split_decision`, and
`allocate_endpoint_workshare`.  The module can be executed directly to
run a smoke test. """

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Set, Dict
import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Data structures (from Parent A & B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

class ModelPool:
    """Tracks available RAM/VRAM and decides if a model tier fits."""
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb

    def can_load(self, tier: ModelTier) -> bool:
        return (tier.ram_mb <= self.ram_ceiling_mb) and (tier.vram_mb <= self.vram_ceiling_mb)

    def load(self, tier: ModelTier) -> bool:
        """Attempt to load a model; return True on success, False otherwise."""
        if self.can_load(tier):
            # In a real system we would deduct resources; here we simply succeed.
            return True
        return False

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

# ----------------------------------------------------------------------
# Parent A – privacy / risk utilities
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    raw = unique_quasi_identifiers / total_records
    return max(0.0, min(1.0, raw))

def anonymize_for_indexing(record: Dict[str, Any],
                           redact_keys: Set[str] | None = None) -> Dict[str, Any]:
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0,
                 sensitivity: float = 1.0) -> float:
    """Deterministic aggregation; noise is injected at the runtime boundary."""
    return float(sum(values))

# ----------------------------------------------------------------------
# Parent B – Hoeffding bound & tropical algebra
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def t_add(x: np.ndarray | float, y: np.ndarray | float):
    """Tropical addition = max."""
    return np.maximum(x, y)

def t_mul(x: np.ndarray | float, y: np.ndarray | float):
    """Tropical multiplication = standard addition."""
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tropical matrix multiplication (max‑plus algebra)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = np.max(A[i, :] + B[:, j])
    return result

# ----------------------------------------------------------------------
# Hybrid functions (the new fused logic)
# ----------------------------------------------------------------------
def hybrid_split_decision(best_gain: float,
                          second_best_gain: float,
                          r: float,
                          delta: float,
                          n: int,
                          unique_quasi_identifiers: int,
                          total_records: int,
                          tie_threshold: float = 0.05) -> SplitDecision:
    """
    Combine privacy risk with Hoeffding‑bound split logic.

    1. Compute reconstruction risk ρ.
    2. Derive privacy‑adjusted gains using tropical multiplication:
           g₁ = t_mul(best_gain, 1‑ρ)
           g₂ = t_mul(second_best_gain, 1‑ρ)
    3. Apply the Hoeffding bound on the adjusted gains.
    4. Return a SplitDecision enriched with the risk‑aware reason.
    """
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    privacy_factor = 1.0 - rho  # higher risk → smaller factor

    # Tropical multiplication (adds in ordinary arithmetic)
    adj_best = t_mul(best_gain, privacy_factor)
    adj_second = t_mul(second_best_gain, privacy_factor)

    eps = hoeffding_bound(r, delta, n)
    gap = adj_best - adj_second
    split = (gap > eps) or (eps < tie_threshold)

    if gap > eps:
        reason = "privacy_adjusted_gap_exceeds_bound"
    elif eps < tie_threshold:
        reason = "privacy_adjusted_tie_threshold"
    else:
        reason = "await_more_data"

    return SplitDecision(should_split=split,
                         epsilon=eps,
                         gain_gap=gap,
                         reason=reason)

def allocate_endpoint_workshare(tier: ModelTier,
                               unique_quasi_identifiers: int,
                               total_records: int,
                               split_decision: SplitDecision,
                               pool: ModelPool) -> Dict[str, Any]:
    """
    Produce a work‑share allocation dict that respects both resource limits
    and privacy‑adjusted split decisions.

    Health score h = 1‑ρ (higher = healthier endpoint).
    Allocation weight = tropical addition of health and split flag:
        w = t_add(h, float(split_decision.should_split))

    The function returns a dictionary with the final decision and a flag
    indicating whether the model can be loaded.
    """
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    health = 1.0 - rho

    # Convert boolean split flag to float (0.0 or 1.0) for tropical addition
    split_flag = 1.0 if split_decision.should_split else 0.0
    weight = t_add(health, split_flag)

    can_load = pool.load(tier)

    allocation = {
        "model_tier": tier.name,
        "can_load": can_load,
        "privacy_risk": rho,
        "endpoint_health": health,
        "split_flag": split_decision.should_split,
        "allocation_weight": float(weight),  # cast to Python float for JSON friendliness
        "split_reason": split_decision.reason,
    }
    return allocation

def tropical_decision_score(gains: np.ndarray,
                            unique_quasi_identifiers: int,
                            total_records: int) -> float:
    """
    Example helper that builds a tropical polynomial from a vector of gains,
    then scales it by the privacy factor (1‑ρ).  Demonstrates deeper
    integration of the two mathematical worlds.
    """
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    privacy_factor = 1.0 - rho

    # Tropical polynomial evaluation: max_i (gain_i + coeff_i)
    # Here we use the gains themselves as both coefficients and variables.
    # In a real scenario, separate coefficient vectors would be used.
    poly_val = np.max(gains + gains)  # max_i (gain_i + gain_i) = 2 * max(gains)
    scaled = t_mul(poly_val, privacy_factor)  # adds privacy_factor
    return float(scaled)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulated streaming statistics
    best_gain = 0.42
    second_best_gain = 0.35
    r = 0.5          # range of gain (max - min)
    delta = 0.05
    n = 120          # number of observed instances

    # Privacy dataset characteristics
    unique_ids = 78
    total_records = 200

    # Choose a model tier and resource pool
    tier = TIER_T2_REASONING
    pool = ModelPool(ram_ceiling_mb=8000, vram_ceiling_mb=8192)

    # Hybrid split decision
    split_dec = hybrid_split_decision(best_gain,
                                      second_best_gain,
                                      r,
                                      delta,
                                      n,
                                      unique_ids,
                                      total_records)

    # Allocation
    allocation = allocate_endpoint_workshare(tier,
                                             unique_ids,
                                             total_records,
                                             split_dec,
                                             pool)

    # Additional tropical score demonstration
    gains_vec = np.array([best_gain, second_best_gain, 0.28])
    tropical_score = tropical_decision_score(gains_vec,
                                             unique_ids,
                                             total_records)

    print("Hybrid Split Decision:", asdict(split_dec))
    print("Allocation Result:", allocation)
    print("Tropical Decision Score:", tropical_score)