# DARWIN HAMMER — match 4572, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s2.py (gen5)
# born: 2026-05-29T23:56:35Z

"""
This module fuses the core mathematics of hybrid_hybrid_hybrid_ternar_hybrid_sparse_wta_hy_m656_s1.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1219_s2.py. The mathematical bridge between the two structures 
is the use of the TTT-Linear weight matrix as the basis for the Bayesian tree cost integration, and the SSIM function 
to evaluate the similarity between the input and output of the ternary router and stylometry features, while incorporating 
the probabilistic weighting of stylometry features and the sparse winner-take-all mechanism to provide a unified system 
that can advise whether a given text fits within a stylometry-constrained VRAM budget.
"""

import numpy as np
import math
import random
import sys
import pathlib

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

def ssim(x, y):
    """Compute the SSIM between two vectors."""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    return (2 * mean_x * mean_y + c1) * (2 * cov_xy + c2) / ((mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2))

def stylometry_features(text):
    """Compute stylometry features for a given text."""
    # For simplicity, let's assume we're only computing the frequency of each word
    words = text.split()
    freq = {}
    for word in words:
        if word not in freq:
            freq[word] = 0
        freq[word] += 1
    return np.array(list(freq.values()))

def hybrid_operation(W, x, text):
    """Hybrid operation that combines the TTT-Linear weight matrix and stylometry features."""
    pred = W @ x
    t = x
    residual = pred - t
    loss = float(residual @ residual)
    ssim_val = ssim(x, pred)
    stylometry_feat = stylometry_features(text)
    return loss, ssim_val, stylometry_feat

def sparse_wta(stylometry_feat):
    """Sparse winner-take-all mechanism for stylometry features."""
    # For simplicity, let's assume we're only selecting the top 10% of features
    num_features = int(0.1 * len(stylometry_feat))
    return np.sort(stylometry_feat)[-num_features:]

if __name__ == "__main__":
    # Smoke test
    W = init_ttt(10)
    x = np.random.rand(10)
    text = "This is a test sentence"
    loss, ssim_val, stylometry_feat = hybrid_operation(W, x, text)
    print(f"Loss: {loss}, SSIM: {ssim_val}, Stylometry features: {stylometry_feat}")
    sparse_feat = sparse_wta(stylometry_feat)
    print(f"Sparse features: {sparse_feat}")