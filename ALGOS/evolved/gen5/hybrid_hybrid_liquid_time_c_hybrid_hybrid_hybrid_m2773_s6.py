# DARWIN HAMMER — match 2773, survivor 6
# gen: 5
# parent_a: hybrid_liquid_time_constant_minhash_m10_s2.py (gen1)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m900_s0.py (gen4)
# born: 2026-05-29T23:45:50Z

import numpy as np
import hashlib
from typing import Iterable, List, Tuple

# ----------------------------------------------------------------------
# Parent A – Liquid Time-Constant utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using Blake2b (64‑bit output)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

# ----------------------------------------------------------------------
# Parent B – Ternary Router utilities
# ----------------------------------------------------------------------
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

def broadcast_probability(phase: int, step: int) -> float:
    if phase < 1 or step < 1:
        raise ValueError('phase and step must be positive')
    return min(1.0, 1.0 / (2 ** max(0, phase - step)))

def pulse_force(peak_force: float, steps: int) -> List[float]:
    if peak_force < 0 or steps <= 0:
        raise ValueError("peak_force non-negative and steps positive required")
    mid = (steps - 1) / 2.0
    denom = max(1.0, mid)
    return [peak_force / denom * (1 - (i / mid)) for i in range(steps)]

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def burst_admission_model(edge_weights: List[float], burst_action_scores: List[float]) -> List[float]:
    return [x * (1 + y) for x, y in zip(edge_weights, burst_action_scores)]

def ternary_router_style(edge_weights: List[float], burst_action_scores: List[float]) -> List[float]:
    return [x * (y ** 0.5) for x, y in zip(edge_weights, burst_action_scores)]

def hybrid_forward(sequence: Iterable[str], initial_state: List[float], alpha: float) -> List[float]:
    minhash_signature = []
    shingles = []
    for text in sequence:
        shingles.append(_hash(0, text))
        minhash_signature.append(compute_phash([_hash(0, token) for token in str(shingles[-1])]))

    similarity = [hamming_distance(a, b) for a, b in zip(minhash_signature[:-1], minhash_signature[1:])]
    similarity = [1 - x / MAX64 for x in similarity]

    edge_weights = pulse_force(1.0, 10)
    burst_action_scores = similarity
    burst_action_scores_normalized = [(x - min(burst_action_scores)) / (max(burst_action_scores) - min(burst_action_scores)) for x in burst_action_scores]
    edge_weights = ternary_router_style(edge_weights, burst_action_scores_normalized)
    burst_admission_scores = burst_admission_model(edge_weights, burst_action_scores_normalized)

    # Liquid Time-Constant Network
    tau = 1.0
    dt = 0.01
    x = [initial_state]
    for i in range(len(sequence) - 1):
        f = sigmoid(np.sum(x[-1]))
        s_t = similarity[i]
        tau_eff = tau / (1 + tau * (f + alpha * s_t))
        dxdt = -(1/tau_eff) * x[-1] + (f + alpha * s_t) * edge_weights[i]
        x.append(x[-1] + dxdt * dt)

    return x[-1]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sequence = ["hello", "world", "python", "programming"]
    initial_state = [0.0, 0.0, 0.0]
    alpha = 0.5
    print(hybrid_forward(sequence, initial_state, alpha))