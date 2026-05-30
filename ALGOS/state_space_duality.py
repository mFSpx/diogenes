#!/usr/bin/env python3
"""Mamba-2 / SSD State Space Duality — sequential and parallel forms.

Sequential SSM (time-varying parameters):
    h_t = A_t h_{t-1} + B_t x_t       # state update
    y_t = C_t h_t                       # output projection

Duality (semiseparable parallel form):
    Y = M X
    M[i, j] = C_i * prod(A_{j+1} .. A_i) * B_j    for i >= j
    M[i, j] = 0                                     for i < j

M is 1-semiseparable and lower-triangular (causal).
The parallel form computes the same output as the sequential scan
but as a dense matrix multiply, enabling hardware-efficient parallelism.

Proof sketch:
    y_i = C_i h_i
         = C_i (A_i h_{i-1} + B_i x_i)
         = C_i B_i x_i  +  C_i A_i (A_{i-1} h_{i-2} + B_{i-1} x_{i-1})
         = sum_{j<=i} [ C_i * prod_{k=j+1}^{i} A_k * B_j ] x_j
    => Y = M X   where M[i,j] = C_i * (prod A_{j+1..i}) * B_j  (causal)

References:
    Dao & Gu (2024) "Transformers are SSMs: Generalized Models and
    Efficient Algorithms Through Structured State Space Duality"
    https://arxiv.org/abs/2405.21060
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ssm_step",
    "ssm_sequential",
    "semiseparable_matrix",
    "ssm_parallel",
    "verify_duality",
]


def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step.

    Parameters
    ----------
    h : (state_dim,)       current hidden state
    x : (input_dim,)       current input token
    A : (state_dim, state_dim)   state-transition matrix (diagonal ok)
    B : (state_dim, input_dim)   input projection
    C : (output_dim, state_dim)  output projection

    Returns
    -------
    h_new : (state_dim,)
    y     : (output_dim,)
    """
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y


def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run SSM sequentially over a sequence.

    Parameters
    ----------
    x_seq : (T, input_dim)
    A     : (state_dim, state_dim)   shared across time steps
    B     : (state_dim, input_dim)   shared across time steps
    C     : (output_dim, state_dim)  shared across time steps
    h0    : (state_dim,) or None     initial state; zeros if None

    Returns
    -------
    Y : (T, output_dim)
    """
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    h = np.zeros(state_dim) if h0 is None else h0.copy()
    outputs = []
    for t in range(T):
        h, y = ssm_step(h, x_seq[t], A, B, C)
        outputs.append(y)
    return np.stack(outputs, axis=0)


def semiseparable_matrix(
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
) -> np.ndarray:
    """Build the (T, T) semiseparable causal matrix.

    M[i, j] = C_i @ prod(A_{j+1} .. A_i) @ B_j   for i >= j
    M[i, j] = 0                                    for i < j

    Parameters
    ----------
    A_seq : (T, state_dim, state_dim)   per-step state matrices
    B_seq : (T, state_dim, input_dim)   per-step input projections
    C_seq : (T, output_dim, state_dim)  per-step output projections

    Returns
    -------
    M : (T, T)  — scalars when output_dim == input_dim == 1,
                  or (T*output_dim, T*input_dim) in general;
                  here we return the scalar-output version
                  (output_dim=1, input_dim=1) for clarity.
                  For the general duality proof see ssm_parallel.
    """
    T = A_seq.shape[0]
    state_dim = A_seq.shape[1]
    M = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1):
            # prod A_{j+1} .. A_i  (identity when j == i)
            P = np.eye(state_dim)
            for k in range(j + 1, i + 1):
                P = A_seq[k] @ P
            # C_seq[i]: (1, state_dim), P: (state_dim, state_dim), B_seq[j]: (state_dim, 1)
            # result: (1, 1) -> scalar
            M[i, j] = float((C_seq[i] @ P @ B_seq[j]).squeeze())
    return M


def ssm_parallel(
    x_seq: np.ndarray,
    A_seq: np.ndarray,
    B_seq: np.ndarray,
    C_seq: np.ndarray,
) -> np.ndarray:
    """Parallel semiseparable form: Y = M X.

    Builds the full (T, T) causal semiseparable matrix M then computes
    Y = M x_seq.  Equivalent to ssm_sequential but fully parallelisable.

    Parameters
    ----------
    x_seq : (T, 1)          scalar-per-step inputs (1-D duality form)
    A_seq : (T, state_dim, state_dim)
    B_seq : (T, state_dim, 1)
    C_seq : (T, 1, state_dim)

    Returns
    -------
    Y : (T, 1)
    """
    M = semiseparable_matrix(A_seq, B_seq, C_seq)   # (T, T)
    Y = M @ x_seq                                    # (T, 1)
    return Y


def verify_duality(
    T: int = 8,
    d_state: int = 4,
    d_model: int = 6,
    tol: float = 1e-9,
) -> bool:
    """Generate random SSM params, compare sequential vs parallel outputs.

    Uses the 1-D (scalar input/output) projection to test the matrix-form
    duality directly.  For the general case the same identity holds
    block-wise.

    Parameters
    ----------
    T       : sequence length
    d_state : hidden state dimension
    d_model : model dimension (used to build random shared A, B, C)
    tol     : max absolute difference allowed

    Returns
    -------
    True if outputs match within tol.
    """
    rng = np.random.default_rng(42)

    # Stable A: small eigenvalues so hidden state doesn't explode
    A_shared = rng.standard_normal((d_state, d_state)) * 0.1
    B_shared = rng.standard_normal((d_state, 1))
    C_shared = rng.standard_normal((1, d_state))

    x_seq = rng.standard_normal((T, 1))

    # Sequential (shared params, scalar i/o)
    Y_seq = ssm_sequential(x_seq, A_shared, B_shared, C_shared)  # (T, 1)

    # Per-step param tensors (same matrix broadcast to each step)
    A_seq = np.stack([A_shared] * T, axis=0)  # (T, d_state, d_state)
    B_seq = np.stack([B_shared] * T, axis=0)  # (T, d_state, 1)
    C_seq = np.stack([C_shared] * T, axis=0)  # (T, 1, d_state)

    Y_par = ssm_parallel(x_seq, A_seq, B_seq, C_seq)  # (T, 1)

    max_err = float(np.max(np.abs(Y_seq - Y_par)))
    passed = max_err < tol

    print(f"sequential Y shape : {Y_seq.shape}")
    print(f"parallel   Y shape : {Y_par.shape}")
    print(f"max |Y_seq - Y_par|: {max_err:.2e}")
    print(f"duality verified   : {passed}")
    return passed


if __name__ == "__main__":
    ok = verify_duality()
    if not ok:
        raise AssertionError("Duality check failed — sequential and parallel outputs diverge.")
