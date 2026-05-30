# DARWIN HAMMER — match 5369, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m157_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_korpus_text_h_m537_s1.py (gen5)
# born: 2026-05-30T00:01:25Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s5 (Differential Privacy & Circuit‑Breaker)
- hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s0 (State‑Space, SSIM, Weighted Entropy)

Mathematical Bridge:
Both parents expose a *morphology*‑driven geometric index (sphericity) and a
risk‑oriented scalar (reconstruction risk).  The bridge is built by letting the
DP‑derived risk modulate the structural‑similarity (SSIM‑like) term while the
morphology‑derived priority (righting‑time index) weights a regret‑weighted
sigmoid.  The final hybrid score combines:

    S = p · σ(R) · (1 + J_minhash) · tanh(R)

where
    p      = righting_time_index(morphology)               (priority)
    σ(R)   = sigmoid(regret)                               (regret weighting)
    J_minhash = MinHash Jaccard similarity of signatures   (structural similarity)
    tanh(R) = bounded control signal (dance)

The DP aggregate is used to privately perturb the raw regret components before
they enter the hybrid formula, ensuring differential‑privacy compliance.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Callable, Iterable, Dict
import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

# ----------------------------------------------------------------------
# Parent A – Differential Privacy utilities
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """DP‑style risk: proportion of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: List[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Laplace‑mechanism sum with differential privacy."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    total = sum(values)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise

# ----------------------------------------------------------------------
# Parent B – Morphology‑based indices & MinHash similarity
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity used by both parents."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Morphology‑driven priority (p)."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def minhash_signature(tokens: Iterable[str], num_perm: int = 128,
                      seed: int = 0) -> List[int]:
    """Create a simple MinHash signature for a set of tokens."""
    random.seed(seed)
    # Generate a list of random hash functions simulated by different salts
    salts = [random.randint(0, 2**31 - 1) for _ in range(num_perm)]
    signature = [2**63 - 1] * num_perm
    for token in tokens:
        token_hash = hash(token)
        for i, salt in enumerate(salts):
            combined = hash((token_hash, salt))
            if combined < signature[i]:
                signature[i] = combined
    return signature

def minhash_jaccard(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def sigmoid(x: float) -> float:
    """Standard logistic sigmoid used as g(·) in the hybrid formula."""
    return 1.0 / (1.0 + math.exp(-x))

def hybrid_risk_similarity(model_tier: ModelTier, morphology: Morphology,
                           total_records: int = 1000) -> float:
    """Combine DP reconstruction risk with geometric sphericity."""
    risk = reconstruction_risk_score(model_tier.ram_mb, total_records)
    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    return risk * sph

def compute_hybrid_score(
    morphology: Morphology,
    model_tier: ModelTier,
    expected: float,
    cost: float,
    risk: float,
    counterfactual: float,
    tokens_i: List[str],
    tokens_ref: List[str],
    epsilon: float = 1.0,
    total_records: int = 1000,
) -> Dict[str, float]:
    """
    Full hybrid score S = p * σ(R) * (1 + J) * tanh(R) with DP‑perturbed regret.

    Parameters
    ----------
    morphology : Morphology
        Physical characteristics feeding the priority term p.
    model_tier : ModelTier
        Supplies the RAM‑based DP risk component.
    expected, cost, risk, counterfactual : float
        Raw regret components (will be privately aggregated).
    tokens_i, tokens_ref : List[str]
        Tokenised representations for MinHash similarity.
    epsilon : float
        Privacy budget for the DP aggregation of regret terms.
    total_records : int
        Population size for reconstruction risk normalisation.

    Returns
    -------
    dict
        Dictionary containing intermediate values and the final hybrid score.
    """
    # 1️⃣ Priority from morphology
    p = righting_time_index(morphology)

    # 2️⃣ Private regret aggregation
    raw_regret_vec = [expected, -cost, -risk, counterfactual]
    R_private = dp_aggregate(raw_regret_vec, epsilon=epsilon)

    # 3️⃣ Regret weighting via sigmoid
    g_R = sigmoid(R_private)

    # 4️⃣ MinHash similarity (structural similarity bridge)
    sig_i = minhash_signature(tokens_i)
    sig_ref = minhash_signature(tokens_ref)
    J = minhash_jaccard(sig_i, sig_ref)

    # 5️⃣ Bounded control signal (dance)
    dance = math.tanh(R_private)

    # 6️⃣ Hybrid risk‑similarity (optional diagnostic)
    risk_sim = hybrid_risk_similarity(model_tier, morphology, total_records)

    # 7️⃣ Final hybrid score
    S = p * g_R * (1.0 + J) * dance

    return {
        "priority_p": p,
        "regret_R_private": R_private,
        "sigmoid_gR": g_R,
        "minhash_J": J,
        "dance_signal": dance,
        "risk_similarity": risk_sim,
        "hybrid_score_S": S,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy objects
    morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.0)
    tier = ModelTier(name="standard", ram_mb=256, tier="B")

    # Regret components
    expected_val = 12.0
    cost_val = 4.5
    risk_val = 2.0
    counterfactual_val = 1.0

    # Token sets for similarity
    tokens_i = ["alpha", "beta", "gamma", "delta"]
    tokens_ref = ["alpha", "epsilon", "gamma", "zeta"]

    result = compute_hybrid_score(
        morphology=morph,
        model_tier=tier,
        expected=expected_val,
        cost=cost_val,
        risk=risk_val,
        counterfactual=counterfactual_val,
        tokens_i=tokens_i,
        tokens_ref=tokens_ref,
        epsilon=0.8,
        total_records=1000,
    )

    for k, v in result.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")