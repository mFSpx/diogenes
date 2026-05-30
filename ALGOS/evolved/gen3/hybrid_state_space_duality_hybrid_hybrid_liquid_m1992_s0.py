# DARWIN HAMMER — match 1992, survivor 0
# gen: 3
# parent_a: state_space_duality.py (gen0)
# parent_b: hybrid_hybrid_liquid_time_c_diffusion_forcing_m16_s2.py (gen2)
# born: 2026-05-29T23:40:21Z

# hybrid_state_space_duality.py

"""Hybrid State Space Duality and Liquid Time Constant Diffusion Forcing (LTC-DF)  
This module fuses the core mathematics of two parent algorithms:  
- **Parent A – State Space Duality (SSD)**  
  Provides a semiseparable parallel form of state space models, enabling hardware-efficient parallelism.
- **Parent B – Liquid Time Constant Diffusion Forcing (LTC-DF)**  
  Implements per-token diffusion forcing where each token of a sequence is corrupted with an independent noise level `t_i`.

**Mathematical bridge**  
At every LTC step we compute a MinHash similarity `s ∈ [0,1]` between the current input-shingle signature and the accumulated signature so far. This similarity is translated into a diffusion timestep `t_i = round((1 - s) * T)` for each token (dimension) of the current input vector. Thus the LTC state governs the amount of noise injected by the diffusion process, while the noisy input returned to the LTC influences the next signature, closing a feedback loop.

The hybrid system therefore evolves according to a combined set of equations:
- State update: `h_new = A_t @ h_{t-1} + B_t @ x_t`
- Output projection: `y_t = C_t @ h_t`
- Diffusion timestep: `t_i = round((1 - s) * T)`
- Noisy input: `x_noisy_i = √ᾱ[t_i]·I_i + √(1-ᾱ[t_i])·ε_i`

The implementation below contains three public functions that demonstrate the hybrid operation and a smoke test.
"""

import numpy as np
import math
import random

# ----------------------------------------------------------------------
# MinHash utilities (from Parent A)
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: list[str], k: int = 128) -> list[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(seed, t) for t in toks) for _ in range(k)]


# ----------------------------------------------------------------------
# Hybrid SSM and LTC-DF
# ----------------------------------------------------------------------
def hybrid_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I: np.ndarray,
    τ: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Single hybrid step.

    Parameters
    ----------
    h : (state_dim,)       current hidden state
    x : (input_dim,)       current input token
    A : (state_dim, state_dim)   state-transition matrix (diagonal ok)
    B : (state_dim, input_dim)   input projection
    C : (output_dim, state_dim)  output projection
    I : (input_dim,)       input vector
    τ : float              time constant
    T : float              diffusion time limit

    Returns
    -------
    h_new : (state_dim,)
    y     : (output_dim,)
    t_i   : float
    x_noisy_i : (input_dim,)
    """
    h_new = A @ h + B @ x
    y = C @ h_new
    s = sum(_hash(0, t) for t in [x]) / len([x])
    t_i = round((1 - s) * T)
    x_noisy_i = np.sqrt((1 - np.exp(-t_i / τ)) * np.exp(-t_i / τ)) * I + np.sqrt(
        np.exp(-t_i / τ)
    ) * np.random.normal(size=x.shape)
    return h_new, y, t_i, x_noisy_i


def hybrid_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I_seq: np.ndarray,
    τ: float,
    T: float,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run hybrid SSM and LTC-DF sequentially over a sequence.

    Parameters
    ----------
    x_seq : (T, input_dim)
    A     : (state_dim, state_dim)   shared across time steps
    B     : (state_dim, input_dim)   shared across time steps
    C     : (output_dim, state_dim)  shared across time steps
    I_seq : (T, input_dim)           input sequence
    τ     : float              time constant
    T     : float              diffusion time limit
    h0    : (state_dim,) or None     initial state; zeros if None

    Returns
    -------
    Y : (T, output_dim)
    """
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros(state_dim) if h0 is None else h0
    Y = np.zeros((T, C.shape[0]))
    for t in range(T):
        h, y, t_i, x_noisy_i = hybrid_step(
            h, x_seq[t], A, B, C, I_seq[t], τ, T
        )
        Y[t] = y
    return Y


def semiseparable_matrix(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I_seq: np.ndarray,
    τ: float,
    T: float,
) -> np.ndarray:
    """Compute semiseparable matrix M.

    Parameters
    ----------
    x_seq : (T, input_dim)
    A     : (state_dim, state_dim)   shared across time steps
    B     : (state_dim, input_dim)   shared across time steps
    C     : (output_dim, state_dim)  shared across time steps
    I_seq : (T, input_dim)           input sequence
    τ     : float              time constant
    T     : float              diffusion time limit

    Returns
    -------
    M : (T, T)
    """
    M = np.zeros((x_seq.shape[0], x_seq.shape[0]))
    for i in range(x_seq.shape[0]):
        for j in range(i + 1):
            _, _, t_i, _ = hybrid_step(
                np.zeros(A.shape[0]),
                x_seq[i],
                A,
                B,
                C,
                I_seq[i],
                τ,
                T,
            )
            M[i, j] = C @ (np.linalg.matrix_power(A, i - j) @ B) * (1 - np.exp(-t_i / τ))
    return M


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def smoke_test():
    np.random.seed(0)
    random.seed(0)
    state_dim = 10
    input_dim = 5
    output_dim = 3
    T = 10
    x_seq = np.random.normal(size=(T, input_dim))
    I_seq = np.random.normal(size=(T, input_dim))
    A = np.random.normal(size=(state_dim, state_dim))
    B = np.random.normal(size=(state_dim, input_dim))
    C = np.random.normal(size=(output_dim, state_dim))
    τ = 0.1
    T = 1.0
    h0 = np.zeros(state_dim)
    Y = hybrid_sequential(x_seq, A, B, C, I_seq, τ, T, h0)
    M = semiseparable_matrix(x_seq, A, B, C, I_seq, τ, T)
    assert Y.shape == (T, output_dim)
    assert M.shape == (T, T)


if __name__ == "__main__":
    smoke_test()