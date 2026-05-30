# DARWIN HAMMER — match 3148, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hdc_hy_hybrid_hybrid_nlms_h_m1740_s3.py (gen5)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_hybrid_nlms_h_m2300_s0.py (gen5)
# born: 2026-05-29T23:48:01Z

"""
This module fuses the Hybrid HDC and Hybrid NLMS algorithms into a single hybrid system, 
dubbed "Hybrid HDC-NLMS". The mathematical bridge between the two structures is the 
use of the TTT-Linear model's update rule to update the weights of the NLMS update rule, 
which is then used to predict the output of the system. The Hybrid HDC's hyperdimensional 
computing primitives are integrated by using the bind and bundle operations to 
compute the input to the TTT-Linear model.

The Hybrid NLMS's decision-making process is integrated by using its update rule 
to compute the reconstruction loss, which is then used to update the weights of 
the NLMS update rule. This fusion enables the Hybrid HDC-NLMS to leverage the 
strengths of both parent algorithms, combining the robustness of hyperdimensional 
computing with the adaptability of NLMS.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Hyperdimensional Computing primitives
# ----------------------------------------------------------------------
Vector = List[int]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    import hashlib
    seed = int.from_bytes(hashlib.sha256(symbol.encode('utf-8')).digest()[:8], 'big')
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError('vectors must have equal length')
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError('at least one vector is required')
    dim = len(vectors[0])
    if any(len(v) != dim for v in vectors):
        raise ValueError('vectors must have equal length')
    sums = [0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    return [1 if x >= 0 else -1 for x in sums]

# ----------------------------------------------------------------------
# Morphology & Routing
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return min(length, width, height) / max(length, width, height)

# ----------------------------------------------------------------------
# TTT-Linear Model
# ----------------------------------------------------------------------
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

    Returns array shape (d_out, d_in), same shape as W.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t                    # (d_out,)
    return 2.0 * np.outer(residual, x)    # (d_out, d_in)

def predict(weights, x):
    """Predict the output of the system using the given weights and input."""
    return sum(w * xi for w, xi in zip(weights, x))

def hybrid_update(weights, x, target, mu=0.5, eps=1e-9, tau=1.0, beta=0.9):
    """Update the weights using the Hybrid NLMS update rule."""
    error = target - predict(weights, x)
    weights.update({i: w + mu * error * xi / (eps + sum(xi**2 for xi in x)) for i, (w, xi) in enumerate(zip(weights, x))})
    return weights

def hybrid_hdc_nlms(x, target, mu=0.5, eps=1e-9, tau=1.0, beta=0.9):
    """Hybrid HDC-NLMS algorithm."""
    dim = len(x)
    W = init_ttt(dim)
    weights = [0.0] * dim
    for _ in range(100):
        residual = ttt_loss(W, x, target)
        grad = ttt_grad(W, x, target)
        W -= 0.01 * grad
        weights = hybrid_update(weights, x, target, mu, eps, tau, beta)
        x_hdc = bind(x, symbol_vector("hdc"))
        x_nlms = predict(weights, x)
        x = (x_hdc + x_nlms) / 2.0
    return x

def hdc_nlms_fusion(x, target):
    """Fusion of HDC and NLMS."""
    dim = len(x)
    W = init_ttt(dim)
    weights = [0.0] * dim
    for _ in range(100):
        residual = ttt_loss(W, x, target)
        grad = ttt_grad(W, x, target)
        W -= 0.01 * grad
        weights = hybrid_update(weights, x, target)
        x_hdc = bind(x, symbol_vector("hdc"))
        x_nlms = predict(weights, x)
        x = (x_hdc + x_nlms) / 2.0
    return x

if __name__ == "__main__":
    x = np.array([1.0, 2.0, 3.0])
    target = np.array([4.0, 5.0, 6.0])
    result = hybrid_hdc_nlms(x, target)
    print(result)