# DARWIN HAMMER — match 336, survivor 0
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s3.py (gen3)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py (gen2)
# born: 2026-05-29T23:28:16Z

import numpy as np
import random
import math
import sys
import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Dict, List

# Fusion of physarum_network_hybrid_hybrid_bandit_m11_s3.py and hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py
# This module implements a hybrid algorithm that combines the Flux-based conductance update primitive
# from physarum_network_hybrid_hybrid_bandit_m11_s3.py and the Sparse Winner-Take-All (WTA) encoding
# with privacy-aware model-pool management from hybrid_sparse_wta_hybrid_privacy_model_m62_s1.py.
# The mathematical bridge between the two structures lies in the concept of conductance and propensity.
# In the context of Physarum networks, conductance represents the ease with which material can flow between points.
# Similarly, in the context of the Hybrid Bandit TTT model, propensity represents the inflow rate of a bandit action.
# By integrating these concepts, we can create a hybrid system that updates the conductance of a network based on the propensity of bandit actions.
# In this fusion, we also incorporate the Sparse WTA encoding to project the conductance values into a high-dimensional sparse vector,
# which is then used to compute the Hamming distance between the current pool configuration and a fixed reference mask.

def flux(conductance, edge_length, pressure_a, pressure_b, eps=1e-12):
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05):
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_bandit_update(conductance, propensity, reward, dt=1.0, gain=1.0, decay=0.05):
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)

def expand(values: List[float], m: int, salt: str = '') -> List[float]:
    """Deterministically project `values` into an m-dimensional sparse vector."""
    if m <= 0:
        raise ValueError('m must be positive')
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):
            h = hashlib.sha256(f'{salt}:{i}:{r}'.encode()).digest()
            j = int.from_bytes(h[:8], 'big') % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out

def top_k_mask(e: List[float], k: int) -> List[int]:
    indices = np.argsort(e)[-k:]
    mask = [1 if i in indices else 0 for i in range(len(e))]
    return mask

def hamming(b: List[int], b_ref: List[int]) -> float:
    return sum(1 for a, b in zip(b, b_ref) if a != b)

def hybrid_network_update_conductances(conductances: List[float], propensities: List[float], rewards: List[float], dt=1.0, gain=1.0, decay=0.05) -> List[float]:
    updated_conductances = []
    for conductance, propensity, reward in zip(conductances, propensities, rewards):
        updated_conductance = hybrid_bandit_update(conductance, propensity, reward, dt, gain, decay)
        updated_conductances.append(updated_conductance)
    return updated_conductances

def hybrid_network_update_pool(config: List[float], m: int, dt=1.0, gain=1.0, decay=0.05, k=10) -> List[int]:
    conductances = hybrid_network_update_conductances(config, [random.random() for _ in range(len(config))], [random.random() for _ in range(len(config))], dt, gain, decay)
    sparse_config = expand(conductances, m)
    return top_k_mask(sparse_config, k)

def hybrid_network_update_risk(config: List[float], m: int, dt=1.0, gain=1.0, decay=0.05, k=10) -> float:
    conductances = hybrid_network_update_conductances(config, [random.random() for _ in range(len(config))], [random.random() for _ in range(len(config))], dt, gain, decay)
    sparse_config = expand(conductances, m)
    b = top_k_mask(sparse_config, k)
    b_ref = top_k_mask(expand([1.0 for _ in range(len(config))], m), k)
    return hamming(b, b_ref) / len(b)

if __name__ == "__main__":
    num_nodes = 10
    num_edges = 50
    num_steps = 100
    dt = 1.0
    gain = 1.0
    decay = 0.05
    m = 100
    k = 10

    config = [1.0 for _ in range(num_edges)]
    conductances = hybrid_network_update_conductances(config, [random.random() for _ in range(num_edges)], [random.random() for _ in range(num_edges)], dt, gain, decay)
    sparse_config = expand(conductances, m)
    b = top_k_mask(sparse_config, k)
    print("Top k mask:", b)

    risk = hybrid_network_update_risk(config, m, dt, gain, decay, k)
    print("Risk factor:", risk)