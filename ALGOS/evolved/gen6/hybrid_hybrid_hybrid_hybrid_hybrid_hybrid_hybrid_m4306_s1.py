# DARWIN HAMMER — match 4306, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_bandit_m48_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1334_s4.py (gen5)
# born: 2026-05-29T23:54:51Z

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

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

def gini_coefficient(values: np.ndarray) -> float:
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non-negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)

def gamma_lanczos(z: float) -> float:
    _LANCZOS_G = 7
    _LANCZOS_C = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma_lanczos(1 - z))
    else:
        z_minus_one = z - 1
        a = _LANCZOS_C[0]
        for i in range(1, len(_LANCZOS_C)):
            a += _LANCZOS_C[i] / (z_minus_one + i)
        t = z_minus_one + _LANCZOS_G + 0.5
        return np.sqrt(2 * np.pi) * t ** (z_minus_one + 0.5) * np.exp(-t) * a

def caputo_derivative(alpha: float, t: int, series: List[float]) -> float:
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be in (0,1) for the Caputo derivative.")
    if t == 0:
        return 0.0
    integral = 0.0
    for tau in range(t):
        kernel = (t - tau) ** (1 - alpha)
        integral += series[tau] * kernel / gamma_lanczos(2 - alpha)
    return integral / gamma_lanczos(1 - alpha)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    y = nlms_predict(weights, x)
    error = target - y
    weights_update = weights + mu * error * x / (x @ x + eps)
    return weights_update, y

def hybrid_update(
    bandit_action: BanditAction,
    reward: float,
    alpha: float,
    t: int,
    series: List[float],
    weights: np.ndarray,
    x: np.ndarray,
) -> Tuple[BanditAction, np.ndarray]:
    gini = gini_coefficient(np.array([bandit_action.propensity, 1 - bandit_action.propensity]))
    caputo = caputo_derivative(alpha, t, series)
    expected_reward = bandit_action.expected_reward + caputo
    propensity = bandit_action.propensity * (1 + gini * (1 - bandit_action.propensity))
    weights_update, _ = nlms_update(weights, x, reward)
    return BanditAction(
        action_id=bandit_action.action_id,
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=bandit_action.confidence_bound,
        algorithm=bandit_action.algorithm,
    ), weights_update

def hybrid_predict(
    bandit_action: BanditAction,
    weights: np.ndarray,
    x: np.ndarray,
) -> Tuple[BanditAction, float]:
    predicted_reward = nlms_predict(weights, x)
    return bandit_action, predicted_reward

if __name__ == "__main__":
    np.random.seed(42)

    bandit_action = BanditAction(
        action_id="action_1",
        propensity=0.5,
        expected_reward=1.0,
        confidence_bound=0.1,
        algorithm="hybrid",
    )

    weights = np.random.rand(10)
    x = np.random.rand(10)

    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    alpha = 0.5
    t = 5

    updated_bandit_action, updated_weights = hybrid_update(
        bandit_action,
        1.5,
        alpha,
        t,
        series,
        weights,
        x,
    )

    predicted_bandit_action, predicted_reward = hybrid_predict(
        updated_bandit_action,
        updated_weights,
        x,
    )

    print(updated_bandit_action)
    print(predicted_reward)