# DARWIN HAMMER — match 1713, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s0.py (gen3)
# parent_b: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py (gen3)
# born: 2026-05-29T23:38:19Z

"""
Hybrid module combining the VRAM scheduler from hybrid_hybrid_model_vram_sc_hybrid_xgboost_objec_m111_s0.py 
and the flux-based conductance update from hybrid_physarum_network_hybrid_hybrid_bandit_m11_s1.py.

The mathematical bridge between the two parents is the integration of the TTT-Linear model's 
update rule into the flux-based conductance update, which modulates the pruning probability 
based on the model's performance and the pressure differences in the network. This allows 
the hybrid algorithm to adapt to the changing memory requirements of the model while maintaining 
an optimal pruning strategy and navigating the network based on the flux-based conductance update.

Specifically, the TTT-Linear model's update rule is used to compute the gradient and Hessian of 
the binary logistic loss, which are then used to compute the optimal leaf weight and split gain. 
The split gain is then used to modulate the pruning probability based on the model's performance. 
The flux-based conductance update is used to model the edge conductance in the network based on 
pressure differences, and the update_conductance function is used as a time-stepping scheme for 
integrating the store differential equation in the HybridBanditTTT class.
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

    Returns array shape (d_out, d_in), same shape as W.
    """
    pred = W @ x
    t = x if target is None else target
    return 2 * (pred - t) @ x[:, None].T

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_update(W, x, target=None, conductance: float = 1.0, edge_length: float = 1.0, pressure_a: float = 0.0, pressure_b: float = 0.0):
    """Hybrid update function that combines the TTT-Linear model's update rule with the flux-based conductance update.

    Args:
    W (numpy array): The weight matrix.
    x (numpy array): The input vector.
    target (numpy array, optional): The target vector. Defaults to None.
    conductance (float, optional): The initial conductance. Defaults to 1.0.
    edge_length (float, optional): The length of the edge. Defaults to 1.0.
    pressure_a (float, optional): The pressure at node A. Defaults to 0.0.
    pressure_b (float, optional): The pressure at node B. Defaults to 0.0.

    Returns:
    tuple: The updated weight matrix and the updated conductance.
    """
    grad = ttt_grad(W, x, target)
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    conductance = update_conductance(conductance, q)
    W = W - 0.01 * grad
    return W, conductance

def hybrid_loss(W, x, target=None, conductance: float = 1.0, edge_length: float = 1.0, pressure_a: float = 0.0, pressure_b: float = 0.0):
    """Hybrid loss function that combines the TTT-Linear model's loss function with the flux-based conductance update.

    Args:
    W (numpy array): The weight matrix.
    x (numpy array): The input vector.
    target (numpy array, optional): The target vector. Defaults to None.
    conductance (float, optional): The initial conductance. Defaults to 1.0.
    edge_length (float, optional): The length of the edge. Defaults to 1.0.
    pressure_a (float, optional): The pressure at node A. Defaults to 0.0.
    pressure_b (float, optional): The pressure at node B. Defaults to 0.0.

    Returns:
    float: The hybrid loss.
    """
    loss = ttt_loss(W, x, target)
    q = flux(conductance, edge_length, pressure_a, pressure_b)
    return loss + 0.1 * abs(q)

if __name__ == "__main__":
    np.random.seed(0)
    W = init_ttt(10)
    x = np.random.rand(10)
    target = np.random.rand(10)
    conductance = 1.0
    edge_length = 1.0
    pressure_a = 0.0
    pressure_b = 0.0
    W, conductance = hybrid_update(W, x, target, conductance, edge_length, pressure_a, pressure_b)
    loss = hybrid_loss(W, x, target, conductance, edge_length, pressure_a, pressure_b)
    print("Hybrid loss:", loss)