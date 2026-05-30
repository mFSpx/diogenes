# DARWIN HAMMER — match 3841, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_endpoi_m16_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_decisi_m2236_s2.py (gen5)
# born: 2026-05-29T23:52:04Z

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Set, Dict
import numpy as np
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Data structures
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
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb

    def can_load(self, tier: ModelTier) -> bool:
        return (tier.ram_mb <= self.ram_ceiling_mb) and (tier.vram_mb <= self.vram_ceiling_mb)

    def load(self, tier: ModelTier) -> bool:
        if self.can_load(tier):
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
    return float(sum(values))

# ----------------------------------------------------------------------
# Parent B – Hoeffding bound & tropical algebra
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

def t_add(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(x, y)

def t_mul(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.add(x, y)

def t_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    result = np.full((A.shape[0], B.shape[1]), -np.inf)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i, j] = np.max(A[i, :] + B[:, j])
    return result

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_split_decision(best_gain: float,
                          second_best_gain: float,
                          r: float,
                          delta: float,
                          n: int,
                          unique_quasi_identifiers: int,
                          total_records: int,
                          tie_threshold: float = 0.05) -> SplitDecision:
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    privacy_factor = 1.0 - rho  

    adj_best = t_mul(best_gain, privacy_factor)
    adj_second = t_mul(second_best_gain, privacy_factor)

    eps = hoeffding_bound(max(adj_best, adj_second), delta, n)
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
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    health = 1.0 - rho

    split_flag = 1.0 if split_decision.should_split else 0.0
    weight = t_add(health, split_flag)

    can_load = pool.load(tier)

    allocation = {
        "model_tier": tier.name,
        "can_load": can_load,
        "privacy_risk": rho,
        "endpoint_health": health,
        "split_flag": split_decision.should_split,
        "allocation_weight": float(weight),  
        "split_reason": split_decision.reason,
    }
    return allocation

def tropical_decision_score(gains: np.ndarray,
                            unique_quasi_identifiers: int,
                            total_records: int) -> np.ndarray:
    rho = reconstruction_risk_score(unique_quasi_identifiers, total_records)
    privacy_factor = 1.0 - rho
    return t_mul(gains, privacy_factor)

def main():
    # Test hybrid_split_decision
    best_gain = 10.0
    second_best_gain = 8.0
    r = 1.0
    delta = 0.1
    n = 100
    unique_quasi_identifiers = 10
    total_records = 1000
    decision = hybrid_split_decision(best_gain, second_best_gain, r, delta, n, unique_quasi_identifiers, total_records)
    print(decision)

    # Test allocate_endpoint_workshare
    tier = TIER_T1_QWEN_0_5B
    pool = ModelPool()
    allocation = allocate_endpoint_workshare(tier, unique_quasi_identifiers, total_records, decision, pool)
    print(allocation)

if __name__ == "__main__":
    main()