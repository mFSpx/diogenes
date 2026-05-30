# DARWIN HAMMER — match 143, survivor 1
# gen: 2
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s1.py (gen1)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:25:55Z

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None: 
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        s = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        s[0] += float(u.reward)
        s[1] += 1.0

def c_to_k(celsius: float) -> float:
    return celsius + 273.15

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp((params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k)))
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    denominator = 1.0 + low + high
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    return numerator / denominator

def temperature_dependent_state_transition(A: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return A * rate

def temperature_dependent_output_projection(C: np.ndarray, temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> np.ndarray:
    rate = developmental_rate(temp_k, params)
    return C * rate

def hybrid_ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> tuple[np.ndarray, np.ndarray]:
    A_temp = temperature_dependent_state_transition(A, temp_k, params)
    C_temp = temperature_dependent_output_projection(C, temp_k, params)
    h_new = A_temp @ h + B @ x
    y = C_temp @ h_new
    return h_new, y

def hybrid_ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    h0: np.ndarray | None = None,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    if h0 is None:
        h0 = np.zeros(state_dim)
    h = h0
    Y = np.zeros((T, C.shape[0]))
    for t in range(T):
        h, Y[t] = hybrid_ssm_step(h, x_seq[t], A, B, C, temp_seq[t], params)
    return Y

def semiseparable_matrix(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_seq: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    T = len(temp_seq)
    state_dim = A.shape[0]
    input_dim = B.shape[1]
    output_dim = C.shape[0]
    M = np.zeros((output_dim * T, input_dim * T))
    A_sequence = np.zeros((T, state_dim, state_dim))
    for i in range(T):
        A_sequence[i] = temperature_dependent_state_transition(A, temp_seq[i], params)
    for i in range(T):
        for j in range(i + 1):
            A_temp = np.eye(state_dim)
            for k in range(j, i):
                A_temp = A_temp @ A_sequence[k]
            C_temp = temperature_dependent_output_projection(C, temp_seq[i], params)
            M[i * output_dim:(i + 1) * output_dim, j * input_dim:(j + 1) * input_dim] = C_temp @ A_temp @ B
    return M

if __name__ == "__main__":
    A = np.array([[0.9, 0.1], [0.1, 0.9]])
    B = np.array([[0.1, 0.2], [0.3, 0.4]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    x_seq = np.array([[1.0, 0.0], [0.0, 1.0]])
    temp_seq = np.array([298.15, 308.15])
    Y = hybrid_ssm_sequential(x_seq, A, B, C, temp_seq)
    print(Y)