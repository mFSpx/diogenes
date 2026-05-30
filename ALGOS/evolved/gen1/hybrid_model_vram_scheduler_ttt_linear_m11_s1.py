# DARWIN HAMMER — match 11, survivor 1
# gen: 1
# parent_a: model_vram_scheduler.py (gen0)
# parent_b: ttt_linear.py (gen0)
# born: 2026-05-29T23:17:54Z

"""
Hybrid algorithm combining the VRAM scheduler from model_vram_scheduler.py and the TTT-Linear model from ttt_linear.py.

The mathematical bridge between the two parents is the update rule of the TTT-Linear model, which can be seen as a form of gradient descent. 
The VRAM scheduler's decision-making process can be viewed as a form of optimization problem, where the goal is to minimize the memory usage while maximizing the model's performance. 
By integrating the TTT-Linear model's update rule into the VRAM scheduler's decision-making process, we can create a hybrid algorithm that adapts to the changing memory requirements of the model.
"""

import numpy as np
import math
import random
import sys
import pathlib
import json
import os

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

def plan_dual_engine_residency(payload=None, state=None, include_gpu=True):
    """Plan the always-on CPU FairyFuse + GPU Q4 DeepSeek residency envelope.

    This is intentionally advisory and side-effect-light: it reads hardware state
    and file receipts but does not import or allocate model weights.
    """
    payload = payload or {}
    state = state or {}
    gpu = gpu_memory() if include_gpu else {"status": "skipped"}
    observed_total = int(gpu.get("total_mb") or 4096) if isinstance(gpu, dict) else 4096
    budget = min(4096, observed_total) if observed_total else 4096
    resident_gpu_mb = 1250 + 1200
    requested_adapters = select_adapters(payload, state)
    adapter_headroom_mb = max(0, budget - resident_gpu_mb - 512)
    decision = "allow" if resident_gpu_mb <= budget else "defer"
    return decision, adapter_headroom_mb

def select_adapters(payload, state):
    """Select adapters based on payload and state.

    This function is a placeholder and should be implemented according to the specific requirements.
    """
    return []

def gpu_memory():
    """Get GPU memory information.

    This function is a placeholder and should be implemented according to the specific requirements.
    """
    return {"total_mb": 4096, "used_mb": 1024, "free_mb": 3072}

def hybrid_ttt_vram_scheduler(x_seq, W0=None, eta=0.01, d_model=None):
    """Process a full token sequence through the hybrid TTT-VRAM scheduler model.

    x_seq: array shape (T, d_in).
    W0: initial weight matrix shape (d_out, d_in). If None, initialized via
        init_ttt with d_out = d_model or d_in.
    eta: learning rate for each gradient step.
    d_model: d_out for W if W0 is None. Defaults to d_in.

    Returns (H shape (T, d_out), W_final shape (d_out, d_in)).

    Each row H[t] is the hidden state produced *after* W has been updated on
    x_seq[t]. The sequence is processed causally: W_t depends only on
    x_0 ... x_t.
    """
    x_seq = np.asarray(x_seq, dtype=float)
    T, d_in = x_seq.shape

    if W0 is None:
        d_out = d_model if d_model is not None else d_in
        W = init_ttt(d_in, d_out=d_out)
    else:
        W = np.array(W0, dtype=float)

    d_out = W.shape[0]
    H = np.empty((T, d_out), dtype=float)

    for t in range(T):
        decision, adapter_headroom_mb = plan_dual_engine_residency()
        if decision == "allow":
            h, W = ttt_forward(W, x_seq[t], eta=eta)
            H[t] = h
        else:
            # If the decision is "defer", we can use a smaller learning rate or
            # reduce the model size to adapt to the limited VRAM.
            eta_reduced = eta * 0.5
            h, W = ttt_forward(W, x_seq[t], eta=eta_reduced)
            H[t] = h

    return H, W

def ttt_forward(W, x, eta=0.01):
    """Full TTT forward pass for one token.

    1. Update: W_new = W - eta * grad_W loss(W, x)
    2. Produce: h = W_new @ x

    Note the order matters: the update happens *before* projection.
    The hidden state h reflects the model *after* it has learned from x.

    Returns (h shape (d_out,), W_new shape (d_out, d_in)).
    """
    g = ttt_grad(W, x)
    W_new = W - eta * g
    h = W_new @ x
    return h, W_new

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    T = 20
    d = 8
    eta = 0.05

    # Sequence: slow sinusoidal drift so the model has real structure to learn
    t_idx = np.linspace(0, 2 * np.pi, T)
    x_seq = np.stack([np.sin(t_idx + k * 0.4) for k in range(d)], axis=1)
    # Small noise on top
    x_seq += rng.standard_normal(x_seq.shape) * 0.05

    W = init_ttt(d, d_out=d, scale=0.01, seed=0)

    print("Hybrid TTT-VRAM scheduler smoke test")
    print(f"  sequence: T={T}, d={d}, eta={eta}")
    print(f"  W0 norm: {np.linalg.norm(W):.6f}\n")
    print(f"{'step':>4}  {'loss':>10}  {'W norm':>10}  {'|h|':>10}")
    print("-" * 44)

    H, W = hybrid_ttt_vram_scheduler(x_seq, W0=W, eta=eta)

    for t in range(T):
        x = x_seq[t]
        loss_before = ttt_loss(W, x)
        print(
            f"{t:>4}  {loss_before:>10.5f}  {np.linalg.norm(W):>10.6f}"
            f"  {np.linalg.norm(H[t]):>10.6f}"
        )