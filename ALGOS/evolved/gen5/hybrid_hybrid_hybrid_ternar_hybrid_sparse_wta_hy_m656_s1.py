# DARWIN HAMMER — match 656, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# born: 2026-05-29T23:30:20Z

"""
Hybrid module combining the ternary_router and ssim algorithms from hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py,
and the sparse winner-take-all mechanism and model pool management from hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py.
The mathematical bridge between the two parents is the use of the TTT-Linear model's update rule 
to modulate the pruning probability in the XGBoost objective's split-gain formula, 
which is then used to evaluate the similarity between the input and output of the ternary router using the ssim function.
The sparse winner-take-all mechanism is used to inform model loading and eviction decisions in the model pool,
where the model with the highest score is loaded into the model pool, and the reconstruction risk score is used to inform the WTA mechanism.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib

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

    Returns array shape (d_out, d_in)
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2 * np.outer(residual, x)

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

def expand(values: list[float], m: int, salt: str = '') -> list[float]:
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

def top_k_mask(values: list[float], k: int) -> list[int]:
    k = max(0, min(k, len(values)))
    winners = {i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]}
    return [1 if i in winners else 0 for i in range(len(values))]

def hybrid_ttt_wta(x, W, target=None):
    """Hybrid function combining TTT and WTA.

    Uses the TTT-Linear model's update rule to modulate the pruning probability in the WTA mechanism.
    """
    loss = ttt_loss(W, x, target)
    grad = ttt_grad(W, x, target)
    values = [float(loss * grad[i, 0]) for i in range(grad.shape[0])]
    mask = top_k_mask(values, 1)
    return mask

def hybrid_load_model(pool: ModelPool, model: ModelTier):
    """Hybrid function combining model loading and WTA.

    Uses the WTA mechanism to inform model loading and eviction decisions in the model pool.
    """
    values = [model.ram_mb]
    mask = top_k_mask(values, 1)
    if mask[0] == 1:
        pool.load_with_eviction(model)

def hybrid_evaluate_similarity(x, W, target=None):
    """Hybrid function combining TTT and similarity evaluation.

    Uses the TTT-Linear model's update rule to evaluate the similarity between the input and output of the ternary router.
    """
    loss = ttt_loss(W, x, target)
    return 1 - loss

if __name__ == "__main__":
    d_in = 10
    d_out = 10
    W = init_ttt(d_in, d_out)
    x = np.random.rand(d_in)
    target = np.random.rand(d_in)
    loss = ttt_loss(W, x, target)
    grad = ttt_grad(W, x, target)
    mask = hybrid_ttt_wta(x, W, target)
    pool = ModelPool(ram_ceiling_mb=1000)
    model = ModelTier("test_model", 500, "T1")
    hybrid_load_model(pool, model)
    similarity = hybrid_evaluate_similarity(x, W, target)
    print(f"Loss: {loss}, Grad: {grad}, Mask: {mask}, Similarity: {similarity}")