# DARWIN HAMMER — match 2215, survivor 0
# gen: 5
# parent_a: poikilotherm_schoolfield.py (gen0)
# parent_b: hybrid_hybrid_regret_engine_hybrid_model_vram_sc_m1177_s2.py (gen4)
# born: 2026-05-29T23:41:16Z

"""
Hybrid Schoolfield-Regret Algorithm
--------------------------------

Combines the nonlinear activity/admission curve of Schoolfield-Rollinson poikilotherm rate primitive
with the MinHash-based decision-making of Hybrid Regret Engine.

The bridge between the two algorithms is found in the normalized_activity function of Schoolfield,
which maps an observed operating temperature to a 0..1 activity gate. This can be seen as a
similarity between two MinHash signatures, where the temperature is used to compute the similarity
between the current state and a reference state. This similarity is then used to determine the
activity gate.

In the Hybrid Regret Engine, the similarity between two MinHash signatures is computed using the
_jaccard_like_similarity function. We can use a similar approach to compute the similarity between
the current temperature and the reference temperature used in Schoolfield's normalized_activity
function.

This hybrid algorithm combines the advantages of both parents: the nonlinear activity/admission curve
of Schoolfield-Rollinson poikilotherm rate primitive and the MinHash-based decision-making of Hybrid
Regret Engine.
"""

import numpy as np
import math
import random
import sys
import pathlib

from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Tuple, Optional

R_CAL = 1.987  # cal mol^-1 K^-1
K25 = 298.15

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    tokens: Tuple[str, ...] = field(default_factory=tuple)  # semantic tokens for MinHash


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of being selected (0‑1)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not toks:
        # all‑ones signature – maximal distance to any non‑empty set
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def _jaccard_like_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    intersection = sum(x & y for x, y in zip(sig_a, sig_b))
    union = sum(x | y for x, y in zip(sig_a, sig_b))
    return intersection / union


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / K25) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / K25) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


def normalized_activity(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map an observed operating temperature to a 0..1 activity gate."""
    params = SchoolfieldParams(t_low=c_to_k(low_c), t_high=c_to_k(high_c))
    rate = developmental_rate(c_to_k(temp_c), params)
    max_rate = max(developmental_rate(c_to_k(low_c + (high_c - low_c) * i / max(1, samples - 1)), params) for i in range(samples))
    return 0.0 if max_rate <= 0 else max(0.0, min(1.0, rate / max_rate))


def hybrid_activity_gate(temp_c: float, low_c: float = 5.0, high_c: float = 40.0, samples: int = 141) -> float:
    """Map an observed operating temperature to a 0..1 activity gate using the Hybrid Regret Engine."""
    reference_temp = c_to_k(low_c + (high_c - low_c) * random.randint(0, samples - 1) / samples)
    reference_rate = developmental_rate(reference_temp)
    current_rate = developmental_rate(c_to_k(temp_c))
    similarity = _jaccard_like_similarity(signature([str(reference_temp)], k=128), signature([str(c_to_k(temp_c))], k=128))
    return max(0.0, min(1.0, current_rate / max(1, reference_rate) * similarity))


def test_hybrid_activity_gate():
    temp_c = 25.0
    print(hybrid_activity_gate(temp_c))
    print(normalized_activity(temp_c))


def test_bandit_action():
    action_id = "test_action"
    propensity = 0.5
    expected_reward = 10.0
    confidence_bound = 2.0
    bandit_action = BanditAction(action_id, propensity, expected_reward, confidence_bound)
    print(bandit_action)


def test_vram_slot_plan():
    artifact_id = "test_artifact"
    artifact_kind = "test_kind"
    action = "test_action"
    estimated_mb = 1024
    reason = "test_reason"
    detail = {"test_key": 1.0}
    vram_slot_plan = VramSlotPlan(artifact_id, artifact_kind, action, estimated_mb, reason, detail)
    print(vram_slot_plan)


if __name__ == "__main__":
    test_hybrid_activity_gate()
    test_bandit_action()
    test_vram_slot_plan()