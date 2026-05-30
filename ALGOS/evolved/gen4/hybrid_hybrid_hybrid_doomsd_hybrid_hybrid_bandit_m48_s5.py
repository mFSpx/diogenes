# DARWIN HAMMER — match 48, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_nlms_omni_cha_m115_s0.py (gen2)
# parent_b: hybrid_hybrid_bandit_router_hybrid_hybrid_endpoi_m134_s3.py (gen3)
# born: 2026-05-29T23:26:41Z

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as dt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def weekday_sakamoto(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)

    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    x = np.sort(values.astype(np.float64))
    if x.size == 0 or np.isclose(x.sum(), 0.0):
        return 0.0
    if np.any(x < 0):
        raise ValueError("values must be non‑negative")
    n = x.size
    i = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2 * i - n - 1) * x)
    denominator = n * x.sum()
    return float(numerator / denominator)


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


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


def schoolfield_rate(params: SchoolfieldParams, temperature: np.ndarray) -> np.ndarray:
    T = temperature.astype(np.float64)
    R = params.r_cal * 4.184

    num = np.exp(-params.delta_h_activation / R * (1.0 / T - 1.0 / 298.15))
    low = np.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))
    high = np.exp(params.delta_h_high / R * (1.0 / params.t_high - 1.0 / T))

    denominator = 1.0 + low + high
    return params.rho_25 * num / denominator


_POLICY: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])


def reset_policy() -> None:
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY[u.action_id]
        stats[0] += float(u.reward)
        stats[1] += 1.0


def average_reward(action_id: str) -> float:
    cum, cnt = _POLICY.get(action_id, [0.0, 0.0])
    return cum / cnt if cnt > 0 else 0.0


def compute_feature_matrix(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> np.ndarray:
    wd = weekday_sakamoto(years, months, days).astype(np.float64) / 6.0

    dates = np.empty(years.shape, dtype=object)
    for idx, (y, m, d) in enumerate(zip(years, months, days)):
        dates[idx] = dt(int(y), int(m), int(d))
    doy = np.array([date.timetuple().tm_yday for date in dates], dtype=np.float64) / 365.0

    temperature = 298.0 + 10.0 * np.sin(2.0 * np.pi * doy)

    rate = schoolfield_rate(params, temperature)

    return np.column_stack((wd, doy, rate))


def nlms_predict(
    weight_dict: Dict[str, np.ndarray],
    feature_vec: np.ndarray,
) -> Dict[str, float]:
    predictions = {}
    for action_id, weight in weight_dict.items():
        predictions[action_id] = np.dot(weight, feature_vec)
    return predictions


def nlms_update(
    weight_dict: Dict[str, np.ndarray],
    feature_vec: np.ndarray,
    target: float,
    gini: float,
    learning_rate: float = 0.1,
) -> Dict[str, np.ndarray]:
    updated_weights = {}
    for action_id, weight in weight_dict.items():
        prediction = np.dot(weight, feature_vec)
        error = target - prediction
        update = learning_rate * gini * error * feature_vec
        updated_weights[action_id] = weight + update
    return updated_weights


def hybrid_batch_update(
    updates: List[BanditUpdate],
    feature_matrix: np.ndarray,
    weight_dict: Dict[str, np.ndarray],
    learning_rate: float = 0.1,
) -> Dict[str, np.ndarray]:
    update_policy(updates)
    gini = gini_coefficient(np.array([u.reward for u in updates]))
    targets = np.array([average_reward(u.action_id) for u in updates])
    updated_weights = {}
    for i, update in enumerate(updates):
        feature_vec = feature_matrix[i]
        target = targets[i]
        updated_weights[update.action_id] = nlms_update(
            {update.action_id: weight_dict.get(update.action_id, np.zeros(feature_vec.shape))},
            feature_vec,
            target,
            gini,
            learning_rate,
        )[update.action_id]
    return updated_weights


# Example usage
years = np.array([2022, 2022, 2022])
months = np.array([1, 2, 3])
days = np.array([1, 15, 28])

feature_matrix = compute_feature_matrix(years, months, days)

weight_dict = {}
updates = [
    BanditUpdate("context1", "action1", 10.0, 0.5),
    BanditUpdate("context2", "action2", 20.0, 0.7),
    BanditUpdate("context3", "action1", 15.0, 0.3),
]

updated_weights = hybrid_batch_update(updates, feature_matrix, weight_dict)

print(updated_weights)