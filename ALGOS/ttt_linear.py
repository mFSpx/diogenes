#!/usr/bin/env python3
"""
Test-Time Training with Linear Hidden States (TTT-Linear).

Core recurrence:

    W_t = W_{t-1} - eta * grad_W loss(W_{t-1}, x_t)
    h_t = W_t x_t

The hidden state is not a vector. It is a weight matrix. On every new input
token, the network executes one gradient descent step before producing output.
The model literally rewrites its synaptic weights during inference. Training
never stops.

This is the opposite of a KV cache. A KV cache stores all past tokens
explicitly and looks them up at query time — memory grows linearly with
sequence length. TTT compresses all past tokens into the weight matrix itself
via gradient descent. The weight matrix is a learned compression of history:
it has seen every x_0 ... x_{t-1} and encoded what it learned from each into
its parameters. Memory is fixed-size regardless of sequence length.

Connection to State Space Models (SSMs): both SSMs and TTT compress history
into a fixed-size state that is updated recurrently. SSM state is a vector
evolved by a linear map (A, B matrices). TTT state is a weight matrix evolved
by gradient descent. The critical difference is that TTT's update rule is
derived from a loss, making the compression explicitly learned and adaptive to
the local statistics of the input stream, while SSM compression is fixed by
the learned A matrix. TTT-Linear is to SSMs what online learning is to a
fixed linear filter.

Reference: Yu et al., "Learning to (Learn at Test Time): RNNs with Expressive
Hidden States", 2024. arXiv:2407.04620.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "init_ttt",
    "ttt_loss",
    "ttt_grad",
    "ttt_step",
    "ttt_forward",
    "ttt_sequence",
]


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


def ttt_step(W, x, eta=0.01, target=None):
    """One TTT weight update: W_new = W - eta * grad_W loss(W, x).

    Returns W_new, same shape as W. The old W is not mutated.
    """
    g = ttt_grad(W, x, target=target)
    return W - eta * g


def ttt_forward(W, x, eta=0.01):
    """Full TTT forward pass for one token.

    1. Update: W_new = W - eta * grad_W loss(W, x)
    2. Produce: h = W_new @ x

    Note the order matters: the update happens *before* projection.
    The hidden state h reflects the model *after* it has learned from x.

    Returns (h shape (d_out,), W_new shape (d_out, d_in)).
    """
    W_new = ttt_step(W, x, eta=eta)
    h = W_new @ x
    return h, W_new


def ttt_sequence(x_seq, W0=None, eta=0.01, d_model=None):
    """Process a full token sequence through TTT-Linear.

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
        h, W = ttt_forward(W, x_seq[t], eta=eta)
        H[t] = h

    return H, W


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

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

    print("TTT-Linear smoke test")
    print(f"  sequence: T={T}, d={d}, eta={eta}")
    print(f"  W0 norm: {np.linalg.norm(W):.6f}\n")
    print(f"{'step':>4}  {'loss':>10}  {'W norm':>10}  {'|h|':>10}")
    print("-" * 44)

    for t in range(T):
        x = x_seq[t]
        loss_before = ttt_loss(W, x)
        h, W = ttt_forward(W, x, eta=eta)
        print(
            f"{t:>4}  {loss_before:>10.5f}  {np.linalg.norm(W):>10.6f}"
            f"  {np.linalg.norm(h):>10.6f}"
        )

    print("\nW norm increases as the model self-modifies on each token.")
    print("Loss generally decreases as W adapts to the local input distribution.")

    # Batch API check
    H, W_final = ttt_sequence(x_seq, eta=eta)
    assert H.shape == (T, d), f"unexpected H shape {H.shape}"
    assert W_final.shape == (d, d), f"unexpected W_final shape {W_final.shape}"
    print(f"\nttt_sequence API check passed. H shape: {H.shape}, W_final shape: {W_final.shape}")
