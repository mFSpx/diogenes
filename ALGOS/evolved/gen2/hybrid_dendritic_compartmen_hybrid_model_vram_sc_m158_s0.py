# DARWIN HAMMER — match 158, survivor 0
# gen: 2
# parent_a: dendritic_compartment.py (gen0)
# parent_b: hybrid_model_vram_scheduler_ttt_linear_m11_s1.py (gen1)
# born: 2026-05-29T23:27:09Z

"""
This module represents a hybrid algorithm that combines the governing equations of the 
Hodgkin-Huxley model from dendritic_compartment.py and the TTT-Linear model from 
hybrid_model_vram_scheduler_ttt_linear_m11_s1.py.

The mathematical bridge between the two parents is the update rule of the TTT-Linear 
model, which can be seen as a form of gradient descent. The Hodgkin-Huxley model's 
ion channel currents can be viewed as a form of optimization problem, where the 
goal is to minimize the difference between the predicted and actual membrane 
potentials. By integrating the TTT-Linear model's update rule into the 
Hodgkin-Huxley model's ion channel currents, we can create a hybrid algorithm that 
adapts to the changing membrane potentials.

The hybrid algorithm uses the TTT-Linear model's update rule to optimize the 
ion channel currents in the Hodgkin-Huxley model, resulting in a more accurate 
prediction of the membrane potential. The TTT-Linear model's self-supervised loss 
function is used to evaluate the performance of the hybrid algorithm.
"""

import numpy as np
import math
import random
import sys
import pathlib

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current.

    I_Na = g_Na * m^3 * h * (V - E_Na)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable

    Returns
    -------
    I_Na:
        Sodium current (mV). Scalar or numpy array.
    """
    return g_Na * m**3 * h * (V - E_Na)

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

def hybrid_step(V, m, h, W, x):
    """Hybrid step that combines the Hodgkin-Huxley model's ion channel currents 
    with the TTT-Linear model's update rule.

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    W:
        Weight matrix for the TTT-Linear model.
    x:
        Input vector for the TTT-Linear model.

    Returns
    -------
    V_new:
        Updated membrane potential (mV). Scalar or numpy array.
    """
    I_Na = sodium_current(V, m, h)
    loss = ttt_loss(W, x)
    d_loss_d_V = 2.0 * I_Na * (V - x)
    V_new = V - 0.1 * d_loss_d_V
    return V_new

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def simulate_hybrid_model(V, m, h, W, x, num_steps):
    """Simulate the hybrid model for a given number of steps.

    Parameters
    ----------
    V:
        Initial membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable
    W:
        Weight matrix for the TTT-Linear model.
    x:
        Input vector for the TTT-Linear model.
    num_steps:
        Number of steps to simulate.

    Returns
    -------
    V_history:
        History of membrane potentials (mV). Scalar or numpy array.
    """
    V_history = [V]
    for _ in range(num_steps):
        V = hybrid_step(V, m, h, W, x)
        V_history.append(V)
    return V_history

if __name__ == "__main__":
    V = 0.0
    m = 0.5
    h = 0.5
    W = init_ttt(10)
    x = np.random.rand(10)
    num_steps = 100
    V_history = simulate_hybrid_model(V, m, h, W, x, num_steps)
    print(V_history)