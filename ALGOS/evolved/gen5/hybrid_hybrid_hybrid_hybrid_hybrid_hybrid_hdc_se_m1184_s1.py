# DARWIN HAMMER — match 1184, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py (gen4)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s0.py (gen3)
# born: 2026-05-29T23:33:13Z

"""Hybrid Bandit-RBF-HDC-WTA Model
This module fuses four Darwin-Hammer parents:

* **Parent A** – hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py: 
  A contextual multi-armed bandit that tracks cumulative reward and uses a 
  LinUCB-style confidence bound, fused with an RBF surrogate that learns a 
  nonlinear mapping from a vector to the observed reward.
* **Parent B** – hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s0.py: 
  A hybrid module combining hyperdimensional computing (HDC) and sparse 
  winner-take-all (WTA) model-pool management.

Mathematical bridge:
The bandit’s expected reward is replaced by the RBF surrogate’s prediction 
for the vector [context, action_one_hot]. The HDC encodes morphology scalars 
into a bipolar hypervector. The sparse WTA expansion is used to expand 
model scores into a dense high-dimensional space. The dot product of the 
hyperdimensional encoding of morphology and the sparse WTA hypervector 
is used as a unified operation to capture both the hyperdimensional 
similarity and the winner-take-all saliency.

The module provides three core hybrid functions:
1. `morphology_hv` – encodes morphology scalars into a bipolar hypervector.
2. `sparse_wta_hv` – expands a list of real scores into a sparse WTA hypervector.
3. `hybrid_priority` – fuses the two similarity measures into a single priority 
   value and demonstrates model-pool loading based on that priority.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Vector = Sequence[float]

# ----------------------------------------------------------------------
# Bandit core (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}          
_STORE: Dict[str, float] = {}                 
_SURROGATE = None                             

def reset_policy() -> None:
    _POLICY.clear()
    _STORE.clear()
    global _SURROGATE
    _SURROGATE = RBFSurrogate(centers=[], weights=[], epsilon=1.0)

def _empirical_reward(a: str) -> float:
    total, n = _POLICY.get(a, [0, 0])
    return total / max(n, 1)

class RBFSurrogate:
    def __init__(self, centers: List[Vector], weights: List[float], epsilon: float):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: Vector) -> float:
        return sum(self.weights[i] * np.exp(-self.epsilon**2 * np.linalg.norm(np.array(x) - np.array(self.centers[i]))**2) for i in range(len(self.centers)))

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from parent B)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[int]:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> List[int]:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)

def bind(a: List[int], b: List[int]) -> List[int]:
    return [a[i] * b[i] for i in range(len(a))]

def expand(scores: List[float], dim: int = 10000, k: int = 10) -> List[int]:
    hv = random_vector(dim)
    top_k_indices = np.argsort(scores)[-k:]
    for i in top_k_indices:
        hv[i] = 1 if scores[i] > 0 else -1
    return hv

def hybrid_priority(morphology_hv: List[int], sparse_wta_hv: List[int]) -> float:
    return np.dot(morphology_hv, sparse_wta_hv) / len(morphology_hv)

def morphology_hv(scalars: List[float], dim: int = 10000) -> List[int]:
    hv = random_vector(dim)
    for i, scalar in enumerate(scalars):
        hv = bind(hv, [1 if scalar > 0 else -1 for _ in range(dim)])
    return hv

def sparse_wta_hv(scores: List[float], dim: int = 10000, k: int = 10) -> List[int]:
    return expand(scores, dim, k)

def update_surrogate(context: List[float], action: List[int], reward: float) -> None:
    global _SURROGATE
    _SURROGATE.centers.append(context + action)
    _SURROGATE.weights.append(reward)

def get_expected_reward(context: List[float], action: List[int]) -> float:
    global _SURROGATE
    return _SURROGATE.predict(context + action)

if __name__ == "__main__":
    reset_policy()
    context = [1.0, 2.0]
    action = [1, 0]
    reward = 10.0
    update_surrogate(context, action, reward)
    print(get_expected_reward(context, action))
    scalars = [1.0, 2.0]
    morphology_hv_result = morphology_hv(scalars)
    scores = [1.0, 2.0, 3.0]
    sparse_wta_hv_result = sparse_wta_hv(scores)
    print(hybrid_priority(morphology_hv_result, sparse_wta_hv_result))