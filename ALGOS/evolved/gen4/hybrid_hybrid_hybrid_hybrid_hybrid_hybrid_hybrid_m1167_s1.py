# DARWIN HAMMER — match 1167, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_state_space_duality_m60_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_nlms_omni_cha_m1_s1.py (gen3)
# born: 2026-05-29T23:33:08Z

"""
Hybrid Temperature‑Epistemic State‑Space / NLMS Model

Parent A: state_space_duality with Schoolfield poikilotherm temperature‑dependent transition.
Parent B: NLMS adaptive filter with epistemic certainty flags influencing the learning rate.

Mathematical bridge:
- The state‑transition matrix A is scaled by the developmental rate ρ(T) (Parent A).
- The NLMS step size μ is modulated by the same ρ(T) and by an epistemic certainty factor
  derived from a flag (Parent B).

Thus the hidden state evolves with temperature‑scaled dynamics while the output
projection (NLMS) adapts with a learning rate that reflects both temperature and
epistemic certainty. The three core functions below realise this fusion.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
import numpy as np

# ---------- Parent A components ----------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‑Rollinson poikilotherm developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def temperature_scaled_transition(A: np.ndarray, temp_k: float) -> np.ndarray:
    """Scale the state‑transition matrix by the developmental rate."""
    rate = developmental_rate(temp_k)
    return rate * A

# ---------- Parent B components ----------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
# Mapping flags → confidence factor in (0, 1]
_EPISTEMIC_CONFIDENCE = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "SURE_MAYBE": 0.3,
    "BULLSHIT": 0.1,
}

def epistemic_factor(flag: str) -> float:
    """Return a scalar learning‑rate modifier derived from an epistemic flag."""
    if flag not in EPISTEMIC_FLAGS:
        raise ValueError(f"unknown epistemic flag: {flag}")
    return _EPISTEMIC_CONFIDENCE[flag]

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """NLMS prediction: w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    error: float,
    mu: float,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalized LMS weight update."""
    norm_sq = float(np.dot(x, x)) + epsilon
    return weights + (mu / norm_sq) * error * x

# ---------- Hybrid functions ----------
def hybrid_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    nlms_w: np.ndarray,
    temp_k: float,
    epistemic_flag: str,
    mu_base: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one hybrid iteration.

    1. Temperature‑scaled state transition.
    2. Linear input injection.
    3. Observation projection.
    4. NLMS prediction of the observation.
    5. NLMS weight update using a learning rate
       μ = μ_base * ρ(T) * epistemic_factor(flag).

    Returns
    -------
    h_next : updated hidden state
    y_pred : NLMS prediction of the output
    nlms_w_next : updated NLMS weight vector
    """
    # 1‑2. State update with temperature‑scaled dynamics
    A_t = temperature_scaled_transition(A, temp_k)
    h_next = A_t @ h + B @ x

    # 3. True output (for demonstration we use linear projection)
    y_true = C @ h_next

    # 4. NLMS prediction
    y_pred = nlms_predict(nlms_w, x)

    # 5. Adaptive learning rate
    rho = developmental_rate(temp_k)
    eta = epistemic_factor(epistemic_flag)
    mu = mu_base * rho * eta

    error = float(y_true - y_pred)
    nlms_w_next = nlms_update(nlms_w, x, error, mu)

    return h_next, y_pred, nlms_w_next

def hybrid_predict(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    temp_k: float,
) -> np.ndarray:
    """Predict the next hidden state and observation without NLMS adaptation."""
    A_t = temperature_scaled_transition(A, temp_k)
    h_next = A_t @ h + B @ x
    y_pred = C @ h_next
    return h_next, y_pred

def hybrid_initialize(
    state_dim: int,
    input_dim: int,
    output_dim: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create random matrices/vectors for a reproducible hybrid system."""
    rng = np.random.default_rng(seed)
    h0 = rng.standard_normal(state_dim)
    A = rng.standard_normal((state_dim, state_dim))
    B = rng.standard_normal((state_dim, input_dim))
    C = rng.standard_normal((output_dim, state_dim))
    nlms_w = rng.standard_normal(input_dim)  # NLMS weight vector matches input dimension
    return h0, A, B, C, nlms_w

# ---------- Smoke test ----------
if __name__ == "__main__":
    # dimensions
    STATE_DIM = 4
    INPUT_DIM = 4
    OUTPUT_DIM = 1

    # initialise system
    h, A, B, C, nlms_w = hybrid_initialize(STATE_DIM, INPUT_DIM, OUTPUT_DIM, seed=42)

    # synthetic input and temperature
    x = np.array([0.5, -1.2, 0.3, 0.0])
    temp_k = 295.0  # ~22 °C
    flag = random.choice(EPISTEMIC_FLAGS)

    # run a few hybrid steps
    for step in range(5):
        h, y_pred, nlms_w = hybrid_step(
            h,
            x,
            A,
            B,
            C,
            nlms_w,
            temp_k,
            flag,
            mu_base=0.05,
        )
        print(f"Step {step+1:02d} | Temp {temp_k:.1f}K | Flag {flag} | y_pred {y_pred:.4f}")

    # demonstrate deterministic predict (no NLMS update)
    h_next, y_next = hybrid_predict(h, x, A, B, C, temp_k)
    print(f"\nDeterministic prediction -> hidden norm {np.linalg.norm(h_next):.4f}, output {y_next.item():.4f}")