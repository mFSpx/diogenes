# DARWIN HAMMER — match 3536, survivor 1
# gen: 7
# parent_a: hybrid_counterfactual_effec_hybrid_hybrid_hybrid_m948_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1444_s1.py (gen6)
# born: 2026-05-29T23:50:37Z

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict

# Constants
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

def minhash_signature(confounders: List[float], hash_count: int = 10, seed: int = 42) -> List[float]:
    np.random.seed(seed)
    hash_values = []
    for _ in range(hash_count):
        weights = np.random.uniform(0, 1, size=len(confounders))
        hash_value = np.dot(confounders, weights) / np.sum(weights)
        hash_values.append(hash_value)
    return hash_values

def hybrid_ate_estimator(treatment: List[float], outcome: List[float], confounders: List[float]) -> CausalEffect:
    ate_estimate = np.mean([t * o for t, o in zip(treatment, outcome)])
    
    # Calculate MinHash signature
    minhash_sig = minhash_signature(confounders)
    
    # Calculate SSIM
    ssim = compute_ssim(minhash_sig, PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)
    
    # Weight ATE estimate by SSIM
    weighted_ate_estimate = ate_estimate * ssim
    
    return CausalEffect(
        effect_id="hybrid",
        treatment="treatment",
        outcome="outcome",
        confounders=tuple(map(str, confounders)),
        ate_estimate=weighted_ate_estimate,
        ate_confidence_interval=(weighted_ate_estimate - 0.1, weighted_ate_estimate + 0.1),
        refutation_passed=True,
        refutation_methods=("method1",),
        heterogeneous_effects={"effect1": 0.5}
    )

def hybrid_score(packet: Dict[str, List[float]]) -> float:
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size < PROTOTYPE_VECTOR.size:
            payload_vec = np.pad(payload_vec, (0, PROTOTYPE_VECTOR.size - payload_vec.size))
        elif payload_vec.size > PROTOTYPE_VECTOR.size:
            payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
        return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist(), dynamic_range=1.0)
    except Exception:
        return 0.0

if __name__ == "__main__":
    treatment = [1.0, 0.0, 1.0, 0.0]
    outcome = [2.0, 3.0, 4.0, 5.0]
    confounders = [0.5, 0.6, 0.7, 0.8]
    effect = hybrid_ate_estimator(treatment, outcome, confounders)
    print(effect)