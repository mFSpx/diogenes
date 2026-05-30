# DARWIN HAMMER — match 656, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py (gen2)
# born: 2026-05-29T23:30:20Z

"""
Module for hybrid algorithm combining the core topologies of 
'hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s2.py' and 
'hybrid_sparse_wta_hybrid_privacy_model_m29_s0.py'. 
The mathematical bridge is the application of the TTT-Linear model's update rule 
to modulate the pruning probability in the XGBoost objective's split-gain formula, 
which is then used to evaluate the similarity between the input and output of the ternary router using the ssim function. 
Additionally, the winner-take-all (WTA) mechanism is used to inform model loading and eviction decisions in the model pool.
"""

import numpy as np
import math
import random
import sys
import pathlib

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

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

    Returns array shape (d_out, d_in)
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return 2 * np.outer(residual, x)

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

def hybrid_router(model_pool: ModelPool, input_x: np.ndarray) -> np.ndarray:
    """Hybrid router that combines the ternary router and the model pool."""
    # Initialize the TTT model
    W = init_ttt(input_x.shape[0])
    
    # Compute the reconstruction loss and gradient
    loss = ttt_loss(W, input_x)
    grad = ttt_grad(W, input_x)
    
    # Use the gradient to inform the model loading and eviction decisions
    model_scores = [ttt_loss(W, np.array([model.ram_mb])) for model in model_pool.loaded.values()]
    model_scores = expand(model_scores, len(model_scores))
    mask = top_k_mask(model_scores, 1)
    
    # Load the top-scoring model into the model pool
    top_model = [model for model, score in zip(model_pool.loaded.values(), mask) if score == 1]
    if top_model:
        model_pool.load_with_eviction(top_model[0])
    
    # Use the loaded model to generate a response to the input
    response = np.array([model.ram_mb for model in model_pool.loaded.values()])
    
    return response

def hybrid_privacy_model(model_pool: ModelPool, input_x: np.ndarray) -> np.ndarray:
    """Hybrid privacy model that combines the TTT-Linear model and the model pool."""
    # Initialize the TTT model
    W = init_ttt(input_x.shape[0])
    
    # Compute the reconstruction loss and gradient
    loss = ttt_loss(W, input_x)
    grad = ttt_grad(W, input_x)
    
    # Use the gradient to inform the model loading and eviction decisions
    model_scores = [ttt_loss(W, np.array([model.ram_mb])) for model in model_pool.loaded.values()]
    model_scores = expand(model_scores, len(model_scores))
    mask = top_k_mask(model_scores, 1)
    
    # Evict the lowest-scoring model from the model pool
    low_model = [model for model, score in zip(model_pool.loaded.values(), mask) if score == 0]
    if low_model:
        model_pool.loaded.pop(low_model[0].name)
    
    # Use the updated model pool to generate a response to the input
    response = np.array([model.ram_mb for model in model_pool.loaded.values()])
    
    return response

def hybrid_evaluation(model_pool: ModelPool, input_x: np.ndarray) -> float:
    """Hybrid evaluation that combines the TTT-Linear model and the model pool."""
    # Initialize the TTT model
    W = init_ttt(input_x.shape[0])
    
    # Compute the reconstruction loss and gradient
    loss = ttt_loss(W, input_x)
    grad = ttt_grad(W, input_x)
    
    # Use the gradient to inform the model loading and eviction decisions
    model_scores = [ttt_loss(W, np.array([model.ram_mb])) for model in model_pool.loaded.values()]
    model_scores = expand(model_scores, len(model_scores))
    mask = top_k_mask(model_scores, 1)
    
    # Evaluate the performance of the model pool
    performance = sum(model.ram_mb for model, score in zip(model_pool.loaded.values(), mask) if score == 1)
    
    return performance

if __name__ == "__main__":
    model_pool = ModelPool()
    model1 = ModelTier("model1", 100, "T1")
    model2 = ModelTier("model2", 200, "T2")
    model_pool.load(model1)
    model_pool.load_with_eviction(model2)
    
    input_x = np.array([1.0, 2.0, 3.0])
    response1 = hybrid_router(model_pool, input_x)
    response2 = hybrid_privacy_model(model_pool, input_x)
    performance = hybrid_evaluation(model_pool, input_x)
    
    print("Hybrid router response:", response1)
    print("Hybrid privacy model response:", response2)
    print("Hybrid evaluation performance:", performance)