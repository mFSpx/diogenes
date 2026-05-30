# DARWIN HAMMER — match 5601, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1383_s0.py (gen4)
# parent_b: hybrid_hybrid_hdc_hybrid_hy_hybrid_hybrid_rbf_su_m182_s3.py (gen4)
# born: 2026-05-30T00:03:31Z

from __future__ import annotations

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Vector utilities
# ----------------------------------------------------------------------
Vector = List[int]
FloatVector = Sequence[float]


def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
    return random_vector(dim, seed)


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]


def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("at least one vector is required")
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError("vectors must have equal length")
    sums = np.zeros(dim, dtype=int)
    for v in vectors:
        sums += np.array(v, dtype=int)
    return [1 if x >= 0 else -1 for x in sums]


def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    if not a:
        raise ValueError("vectors must not be empty")
    return sum(x * y for x, y in zip(a, b)) / len(a)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: FloatVector, b: FloatVector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ----------------------------------------------------------------------
# RBF surrogate
# ----------------------------------------------------------------------
@dataclass
class RBFSurrogate:
    centers: List[FloatVector]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: FloatVector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


def modulate_surrogate(surrogate: RBFSurrogate, modulation_vector: Vector) -> RBFSurrogate:
    modulated_centers = [
        [c_i * modulation_vector[i] for i, c_i in enumerate(c)] 
        for c in surrogate.centers
    ]
    ref = [1] * len(modulation_vector)
    intensity = similarity(modulation_vector, ref)
    modulated_weights = [w * intensity for w in surrogate.weights]
    return RBFSurrogate(modulated_centers, modulated_weights, surrogate.epsilon)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
) -> Dict[str, float]:
    if not actions:
        return {}
    cf: Dict[str, float] = {
        c.action_id: c.outcome_value * c.probability for c in counterfactuals
    }
    adjusted: Dict[str, float] = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0) for a in actions
    }
    best = max(adjusted.values())
    raw_weights = {k: math.exp(v - best) for k, v in adjusted.items()}
    total = sum(raw_weights.values())
    return {k: w / total for k, w in raw_weights.items()}


def build_modulation_vector(
    strategy: Dict[str, float],
    dim: int = 2048,
) -> Vector:
    if not strategy:
        raise ValueError("strategy must contain at least one action")
    signed_vectors: List[Vector] = []
    for action_id, prob in strategy.items():
        base = symbol_vector(action_id, dim)
        sign = 1 if prob > 0 else -1
        signed = [sign * base_i for base_i in base]
        signed_vectors.append(signed)
    return bundle(signed_vectors)


def hybrid_bayesian_update(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    surrogate: RBFSurrogate,
    feature_vec: FloatVector,
    dim: int = 2048,
) -> Dict[str, float]:
    strategy = compute_regret_weighted_strategy(actions, counterfactuals)
    mod_vec = build_modulation_vector(strategy, dim=dim)
    mod_surrogate = modulate_surrogate(surrogate, mod_vec)
    likelihood = mod_surrogate.predict(feature_vec)
    unnorm = {aid: prob * likelihood for aid, prob in strategy.items()}
    total = sum(unnorm.values())
    if total == 0:
        return {aid: 1.0 / len(unnorm) for aid in unnorm}
    return {aid: val / total for aid, val in unnorm.items()}


def elect_leaders_via_bayesian_graph(
    posterior: Dict[str, float],
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    return sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)[:top_n]