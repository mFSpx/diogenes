# DARWIN HAMMER — match 60, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_bandit_router_state_space_duality_m143_s0.py (gen2)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:26:38Z

"""
This module integrates the Schoolfield-Rollinson poikilotherm rate primitive from hybrid_hybrid_bandit_router_state_space_duality_m143_s0.py
and the State Space Duality from state_space_duality.py.
The mathematical bridge between these two structures is the incorporation of the temperature-dependent 
developmental rate from the poikilotherm model into the state space model's state update and output projection.
This allows the state space model to adapt its state transition and output projection based on the current temperature or state of the system.
"""

import math
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return rate * A

def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step with temperature-dependent state transition.

    Parameters
    ----------
    h : (state_dim,)       current hidden state
    x : (input_dim,)       current input token
    A : (state_dim, state_dim)   state-transition matrix (diagonal ok)
    B : (state_dim, input_dim)   input projection
    C : (output_dim, state_dim)  output projection
    temp_k: float          temperature in Kelvin

    Returns
    -------
    h_new : (state_dim,)
    y     : (output_dim,)
    """
    A_temp = temperature_dependent_state_transition(A, temp_k)
    h_new = A_temp @ h + B @ x
    y = C @ h_new
    return h_new, y

def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run SSM sequentially over a sequence with temperature-dependent state transition.

    Parameters
    ----------
    x_seq : (T, input_dim)
    A     : (state_dim, state_dim)   shared across time steps
    B     : (state_dim, input_dim)   shared across time steps
    C     : (output_dim, state_dim)  shared across time steps
    temp_seq: (T,)        temperature sequence in Kelvin
    h0    : (state_dim,) or None     initial state; zeros if None

    Returns
    -------
    Y : (T, output_dim)
    """
    T, _ = x_seq.shape
    state_dim = A.shape[0]

    if h0 is None:
        h0 = np.zeros(state_dim)

    Y = np.zeros((T, C.shape[0]))
    h = h0

    for t in range(T):
        h, y = ssm_step(h, x_seq[t], A, B, C, temp_seq[t])
        Y[t] = y

    return Y

def semiseparable_matrix(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """Compute the semiseparable matrix M.

    Parameters
    ----------
    A : (state_dim, state_dim)   state-transition matrix (diagonal ok)
    B : (state_dim, input_dim)   input projection
    C : (output_dim, state_dim)  output projection

    Returns
    -------
    M : (output_dim, state_dim * input_dim)
    """
    state_dim, input_dim = B.shape
    output_dim, _ = C.shape

    M = np.zeros((output_dim, state_dim * input_dim))

    for i in range(output_dim):
        for j in range(state_dim * input_dim):
            prod = np.eye(state_dim)
            for k in range(i, j // input_dim, -1):
                prod = prod @ A
            M[i, j] = C[i] * prod @ B[:, j % input_dim]

    return M

if __name__ == "__main__":
    np.random.seed(0)

    state_dim = 2
    input_dim = 1
    output_dim = 1
    T = 10

    A = np.random.rand(state_dim, state_dim)
    B = np.random.rand(state_dim, input_dim)
    C = np.random.rand(output_dim, state_dim)

    x_seq = np.random.rand(T, input_dim)
    temp_seq = np.random.uniform(283.15, 307.15, size=T)

    h0 = np.zeros(state_dim)

    Y_seq = ssm_sequential(x_seq, A, B, C, temp_seq, h0)
    M = semiseparable_matrix(A, B, C)

    print(Y_seq.shape)
    print(M.shape)