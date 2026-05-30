# DARWIN HAMMER — match 1407, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_endpoi_m1071_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_path_s_hybrid_hybrid_hybrid_m417_s2.py (gen5)
# born: 2026-05-29T23:36:13Z

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable

import numpy as np

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(
    m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0
) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def compute_dhash(values: List[float]) -> int:
    bits = 0
    for i in range(len(values) - 1):
        bits = (bits << 1) | int(values[i] > values[i + 1])
    return bits


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def compute_morphology_vector(m: Morphology) -> List[float]:
    sph = sphericity_index(m.length, m.width, m.height)
    flat = flatness_index(m.length, m.width, m.height)
    rti = righting_time_index(m)
    rp = recovery_priority(m)
    return [sph, flat, rti, rp]


@dataclass(frozen=True)
class MathAction:
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2-D array (time, dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[-1] = np.concatenate([path[-1], path[-1]])
    return out


def simple_signature(lead_lag_path: np.ndarray) -> np.ndarray:
    mean = lead_lag_path.mean(axis=0)
    second = (lead_lag_path[:, :, None] * lead_lag_path[:, None, :]).sum(axis=0)
    idx = np.triu_indices_from(second)
    second_flat = second[idx]
    return np.concatenate([mean, second_flat])


def regret_weighted_utility(
    action: MathAction,
    signature: np.ndarray,
    scaling: float,
    alpha: float = 0.5,
) -> float:
    if scaling < 0:
        raise ValueError("scaling must be non-negative")
    base = (action.expected_value - action.cost) * (1.0 - action.risk)
    if base <= 0:
        return 0.0
    sig_mean = signature.mean()
    sig_std = signature.std()
    f_sig = alpha * sig_mean + (1.0 - alpha) * sig_std
    return scaling * base * max(0.0, f_sig)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def regret_weighted_bandit(
    actions: List[MathAction],
    signature: np.ndarray,
    scaling: float,
    time_step: int = 1,
) -> BanditAction:
    utilities = np.array(
        [regret_weighted_utility(a, signature, scaling) for a in actions],
        dtype=float,
    )
    if np.all(utilities == 0):
        propensities = np.full_like(utilities, 1.0 / len(utilities))
    else:
        propensities = softmax(utilities)

    confidence = np.sqrt(np.log(max(time_step, 2)) / (2 * np.maximum(propensities, 1e-12)))

    chosen_idx = int(np.random.choice(len(actions), p=propensities))
    chosen = actions[chosen_idx]
    return BanditAction(
        action_id=chosen.id,
        propensity=propensities[chosen_idx],
        expected_reward=utilities[chosen_idx],
        confidence_bound=confidence[chosen_idx],
    )


def encode_morphology_tokens(morph_vec: List[float]) -> Tuple[str, ...]:
    tokens = []
    for val in morph_vec:
        token = str(int(val * 100))
        tokens.append(token)
    return tuple(tokens)


def improved_regret_weighted_bandit(
    actions: List[MathAction],
    morphologies: List[Morphology],
    time_step: int = 1,
) -> BanditAction:
    signatures = []
    scalings = []
    for morphology in morphologies:
        morph_vec = compute_morphology_vector(morphology)
        tokens = encode_morphology_tokens(morph_vec)
        lead_lag_path = lead_lag_transform(np.array([morph_vec]))
        signature = simple_signature(lead_lag_path)
        scaling = recovery_priority(morphology)
        signatures.append(signature)
        scalings.append(scaling)

    avg_signature = np.mean(signatures, axis=0)
    avg_scaling = np.mean(scalings)

    return regret_weighted_bandit(actions, avg_signature, avg_scaling, time_step)


# Example usage
if __name__ == "__main__":
    morphologies = [
        Morphology(length=1.0, width=2.0, height=3.0, mass=10.0),
        Morphology(length=4.0, width=5.0, height=6.0, mass=20.0),
    ]
    actions = [
        MathAction(id="action1", tokens=("token1",), expected_value=10.0, cost=1.0, risk=0.1),
        MathAction(id="action2", tokens=("token2",), expected_value=20.0, cost=2.0, risk=0.2),
    ]

    bandit_action = improved_regret_weighted_bandit(actions, morphologies)
    print(bandit_action)