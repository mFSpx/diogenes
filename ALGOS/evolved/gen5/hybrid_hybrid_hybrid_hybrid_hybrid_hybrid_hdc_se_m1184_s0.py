# DARWIN HAMMER — match 1184, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py (gen4)
# parent_b: hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s0.py (gen3)
# born: 2026-05-29T23:33:13Z

"""Hybrid Bandit-RBF-HDC Model
This module fuses three Darwin-Hammer parents:

* **Parent A** – hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m18_s3.py: 
  A contextual multi-armed bandit with a LinUCB-style confidence bound and 
  an RBF surrogate that learns a nonlinear mapping from a vector to the 
  observed reward.
* **Parent B** – hybrid_hybrid_hdc_serpentin_hybrid_sparse_wta_hy_m117_s0.py: 
  A hybrid module combining hyperdimensional computing (HDC) and sparse 
  winner-take-all (WTA) model-pool management.

The mathematical bridge between the two parents lies in the combination of 
the bandit's expected reward and the HDC's hyperdimensional similarity. 
The bandit's expected reward is replaced by the RBF surrogate's prediction 
for the vector [context, action_one_hot]. The HDC's hyperdimensional 
similarity is used to select the most salient dimensions in the sparse 
WTA expansion. The dot product of the hyperdimensional similarity and 
the sparse WTA hypervector is used to drive the hybrid recovery priority 
and decision-making of the ModelPool.

The module provides three core hybrid functions:
1. `morphology_hv` – encodes morphology scalars into a bipolar hypervector.
2. `sparse_wta_hv` – expands a list of real scores into a sparse WTA hypervector.
3. `hybrid_priority` – fuses the bandit's expected reward and the HDC's 
   hyperdimensional similarity into a single priority value.
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
BipolarVector = List[int]

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


_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}                 # placeholder VRAM store (unused)
_SURROGATE = None                             # will hold an RBFSurrogate instance


def reset_policy() -> None:
    """Clear all learned statistics and the surrogate model."""
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
        return sum(self.weights[i] * math.exp(-self.epsilon**2 * np.linalg.norm(np.array(x) - np.array(self.centers[i]))**2) for i in range(len(self.centers)))


# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent B)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: str | int | None = None) -> BipolarVector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]


def symbol_vector(symbol: str, dim: int = 10000) -> BipolarVector:
    seed = int.from_bytes(
        hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big"
    )
    return random_vector(dim, seed)


def bind(a: BipolarVector, b: BipolarVector) -> BipolarVector:
    return [a_i * b_i for a_i, b_i in zip(a, b)]


def sparse_wta_hv(scores: List[float], k: int = 10) -> BipolarVector:
    dim = len(scores)
    indices = np.argsort(scores)[-k:]
    hv = [0] * dim
    for i in indices:
        hv[i] = 1
    return hv


def morphology_hv(scalars: List[float], dim: int = 10000) -> BipolarVector:
    hv = random_vector(dim)
    for scalar in scalars:
        hv = bind(hv, [1 if scalar > 0 else -1 for _ in range(dim)])
    return hv


def hybrid_priority(context: List[float], action: List[float], 
                    rbfsurrogate: RBFSurrogate, 
                    dim: int = 10000) -> float:
    hv = morphology_hv(context + action, dim)
    sparse_hv = sparse_wta_hv([rbfsurrogate.predict(context + action) for _ in range(dim)])
    return np.dot(hv, sparse_hv)


def test_hybrid_priority():
    reset_policy()
    global _SURROGATE
    _SURROGATE = RBFSurrogate([[1, 2, 3], [4, 5, 6]], [0.5, 0.5], 1.0)
    context = [1, 2, 3]
    action = [4, 5, 6]
    print(hybrid_priority(context, action, _SURROGATE))


if __name__ == "__main__":
    test_hybrid_priority()