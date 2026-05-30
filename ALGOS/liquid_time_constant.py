#!/usr/bin/env python3
"""Liquid Time-Constant Networks (LTCs) — reference implementation.

Theory
------
LTCs are a class of continuous-time recurrent neural networks derived from
biophysical models of neurons.  The hidden state x(t) evolves according to
an ODE whose time constant is *itself a function of the current input*, making
the network's temporal dynamics input-dependent.

Biological origin
-----------------
The architecture was inspired by the *C. elegans* connectome — a nematode worm
with exactly 302 neurons, 22 of which form the locomotion central-pattern
generator studied by Hasani et al. (2021).  Each neuron is modelled as an RC
circuit with a leaky current and a synaptic drive term.

ODE formulation
---------------
    dx(t)/dt = -[1/τ + f(x(t), I(t), t, θ)] · x(t)
               + f(x(t), I(t), t, θ) · A

where
    x(t)  ∈ ℝ^n   — hidden (neural) state
    I(t)  ∈ ℝ^m   — external input
    τ     ∈ ℝ^+   — base membrane time constant (scalar, shared here)
    A     ∈ ℝ^n   — asymptotic target state (bias attractor)
    θ             — network parameters (W, b)
    f(·)  ∈ ℝ^n   — learned gating / synaptic drive (sigmoid MLP)

Liquid time constant
--------------------
The *effective* (input-dependent) time constant collapses to:

    τ_sys(t) = τ / (1 + τ · f(x(t), I(t), t, θ))

When f is large the system responds fast; when f is small the system is slow.
This is the key property that distinguishes LTCs from vanilla RNNs — the
network can simultaneously be fast at detecting events and slow at integrating
context.

Reference
---------
Hasani R., Lechner M., Amini A., Rus D., Grosu R.
"Liquid Time-Constant Networks", AAAI 2021.
https://arxiv.org/abs/2006.04439
"""

from __future__ import annotations

import numpy as np

__all__ = ["sigmoid", "ltc_f", "ltc_step", "ltc_forward", "init_ltc"]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise sigmoid σ(x) = 1 / (1 + exp(-x))."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Network function f(x, I, t, θ) = σ(W @ [x; I] + b).

    Parameters
    ----------
    x : shape (hidden_dim,)   — current hidden state
    I : shape (input_dim,)    — current external input
    W : shape (hidden_dim, hidden_dim + input_dim)
    b : shape (hidden_dim,)

    Returns
    -------
    f_val : shape (hidden_dim,)   values in (0, 1)
    """
    concat = np.concatenate([x, I], axis=0)  # (hidden_dim + input_dim,)
    return sigmoid(W @ concat + b)


# ---------------------------------------------------------------------------
# Single step
# ---------------------------------------------------------------------------

def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: dict,
    dt: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Advance the LTC state by one Euler step.

    ODE:  dx/dt = -[1/τ + f] · x + f · A

    Parameters
    ----------
    x      : shape (hidden_dim,)  — current state
    I      : shape (input_dim,)   — current input
    params : dict with keys
        "W"   : shape (hidden_dim, hidden_dim + input_dim)
        "b"   : shape (hidden_dim,)
        "tau" : float  — base time constant
        "A"   : shape (hidden_dim,)  — attractor target
    dt     : Euler integration step size

    Returns
    -------
    x_new   : shape (hidden_dim,)  — updated state
    tau_sys : float  — mean effective liquid time constant over neurons
    """
    W = params["W"]
    b = params["b"]
    tau = float(params["tau"])
    A = params["A"]

    f_val = ltc_f(x, I, W, b)                          # (hidden_dim,)

    dx_dt = -(1.0 / tau + f_val) * x + f_val * A       # (hidden_dim,)

    x_new = x + dt * dx_dt                              # Euler step

    tau_sys_vec = tau / (1.0 + tau * f_val)             # (hidden_dim,)
    tau_sys = float(np.mean(tau_sys_vec))

    return x_new, tau_sys


# ---------------------------------------------------------------------------
# Sequence forward pass
# ---------------------------------------------------------------------------

def ltc_forward(
    I_seq: np.ndarray,
    params: dict,
    x0: np.ndarray | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the LTC over a full input sequence.

    Parameters
    ----------
    I_seq  : shape (T, input_dim)  — input time series
    params : dict — same structure as for ltc_step
    x0     : shape (hidden_dim,) optional initial state (zeros if None)
    dt     : Euler step size

    Returns
    -------
    X          : shape (T, hidden_dim)  — hidden state trajectory
    tau_sys_seq: shape (T,)             — effective time constant at each step
    """
    T, input_dim = I_seq.shape
    hidden_dim = params["A"].shape[0]

    x = np.zeros(hidden_dim) if x0 is None else np.array(x0, dtype=float)

    X = np.empty((T, hidden_dim))
    tau_sys_seq = np.empty(T)

    for t in range(T):
        x, tau_sys = ltc_step(x, I_seq[t], params, dt=dt)
        X[t] = x
        tau_sys_seq[t] = tau_sys

    return X, tau_sys_seq


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def init_ltc(
    hidden_dim: int,
    input_dim: int,
    tau: float = 1.0,
    seed: int = 0,
) -> dict:
    """Initialise LTC parameters.

    Parameters
    ----------
    hidden_dim : number of hidden (neural) units
    input_dim  : dimensionality of the external input
    tau        : base membrane time constant (positive scalar)
    seed       : numpy RNG seed for reproducibility

    Returns
    -------
    params : dict with keys "W", "b", "tau", "A"
        W : shape (hidden_dim, hidden_dim + input_dim)  small normal
        b : shape (hidden_dim,)                         zeros
        A : shape (hidden_dim,)                         uniform [0, 1]
        tau: float
    """
    rng = np.random.default_rng(seed)
    fan_in = hidden_dim + input_dim
    W = rng.normal(0.0, 0.1, size=(hidden_dim, fan_in))
    b = np.zeros(hidden_dim)
    A = rng.uniform(0.0, 1.0, size=(hidden_dim,))
    return {"W": W, "b": b, "tau": float(tau), "A": A}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hidden_dim = 8
    input_dim = 4
    T = 20
    dt = 0.1

    rng = np.random.default_rng(42)
    I_seq = rng.normal(0.0, 1.0, size=(T, input_dim))

    params = init_ltc(hidden_dim, input_dim, tau=1.0, seed=0)

    X, tau_sys_seq = ltc_forward(I_seq, params, dt=dt)

    print(f"LTC demo  —  hidden={hidden_dim}  input={input_dim}  T={T}  dt={dt}")
    print(f"{'step':>4}  {'tau_sys':>10}  {'|x| (L2)':>12}")
    print("-" * 32)
    for t in range(T):
        norm_x = float(np.linalg.norm(X[t]))
        print(f"{t:>4}  {tau_sys_seq[t]:>10.4f}  {norm_x:>12.4f}")

    print()
    print(f"tau_sys range: [{tau_sys_seq.min():.4f}, {tau_sys_seq.max():.4f}]")
    print(f"final state norm: {np.linalg.norm(X[-1]):.4f}")
