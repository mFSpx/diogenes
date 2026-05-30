# DARWIN HAMMER — match 875, survivor 0
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py (gen3)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_fisher_locali_m132_s0.py (gen2)
# born: 2026-05-29T23:31:21Z

"""
Hybrid Physarum Infotaxis algorithm, combining the flux-based conductance update 
of the Physarum Network algorithm with the entropy-driven decision logic of 
Infotaxis. The mathematical bridge between these two algorithms lies in the 
concept of information density, which is used to determine the best action 
to minimize expected entropy in Infotaxis, and can be related to the 
pressure differences in the Physarum Network algorithm.
"""

import math
import random
import sys
import numpy as np
import pathlib
import hashlib
from datetime import datetime, timezone

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [max(0, (1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def calculate_pressure(conductance: float, edge_length: float, q: float) -> float:
    return conductance * q / edge_length

def calculate_information_density(pressure: float) -> float:
    return math.log(pressure + 1)

def calculate_expected_entropy(sig_a: list[int], sig_b: list[int], p_hit: float) -> float:
    similarity_value = similarity(sig_a, sig_b)
    return p_hit * calculate_information_density(similarity_value) + (1 - p_hit) * calculate_information_density(1 - similarity_value)

def hybrid_operation(conductance: float, edge_length: float, q: float, sig_a: list[int], sig_b: list[int], p_hit: float) -> float:
    pressure = calculate_pressure(conductance, edge_length, q)
    expected_entropy = calculate_expected_entropy(sig_a, sig_b, p_hit)
    return update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05) * math.exp(-expected_entropy)

if __name__ == "__main__":
    conductance = 1.0
    edge_length = 1.0
    q = 1.0
    sig_a = signature(["token1", "token2"], k=128)
    sig_b = signature(["token3", "token4"], k=128)
    p_hit = 0.5
    print(hybrid_operation(conductance, edge_length, q, sig_a, sig_b, p_hit))