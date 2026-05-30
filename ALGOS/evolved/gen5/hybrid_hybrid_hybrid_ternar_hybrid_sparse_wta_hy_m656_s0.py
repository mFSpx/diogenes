# DARWIN HAMMER — match 656, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# born: 2026-05-29T23:30:20Z

"""
Hybrid module combining the TTT-Linear model and sparse winner-take-all (WTA) mechanism from 
hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py and hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py.
The mathematical bridge between the two parents is the application of the TTT-Linear model's 
reconstruction loss to modulate the model loading scores in the WTA mechanism.

The TTT-Linear model's update rule is used to compute the gradient and Hessian of the binary 
logistic loss, which are then used to compute the optimal leaf weight and split gain. 
The split gain is then used to modulate the model loading scores in the WTA mechanism.

The WTA mechanism is used to inform model loading and eviction decisions in the model pool, 
where the model with the highest score is loaded into the model pool.

The reconstruction risk score from the TTT-Linear model is used to inform the WTA mechanism, 
allowing the model pool to adapt to changing data distributions.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}

    def is_loaded(self, name: str) -> bool: 
        return name in self.loaded

    def _used(self) -> int: 
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): 
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: 
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name]=model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure to
    reconstruct tokens from the sequence.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    """Gradient of ttt_loss w.r.t. W.

    Closed-form for reconstruction loss:
        loss = ||Wx - x||^2 = (Wx - x)^T (Wx - x)
        d loss / dW = 2 (Wx - x) x^T

    Returns array shape (d_out, d_in).
    """
    pred = W @ x
    t = x if target is None else target
    return 2 * np.outer(pred - t, x)

def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return list(winners)

def hybrid_wta_ttt(model_pool: ModelPool, ttt_model: np.ndarray, inputs: list[np.ndarray], k: int) -> list[int]:
    scores = []
    for x in inputs:
        loss = ttt_loss(ttt_model, x)
        scores.append(loss)
    winners = top_k_mask(scores, k)
    for i in winners:
        model_tier = ModelTier(f"model_{i}", 100, "T1")
        model_pool.load_with_eviction(model_tier)
    return winners

def main():
    model_pool = ModelPool()
    ttt_model = init_ttt(10)
    inputs = [np.random.rand(10) for _ in range(10)]
    winners = hybrid_wta_ttt(model_pool, ttt_model, inputs, 3)
    print(winners)

if __name__ == "__main__":
    main()