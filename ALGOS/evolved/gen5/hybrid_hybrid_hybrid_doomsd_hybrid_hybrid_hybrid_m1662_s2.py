# DARWIN HAMMER — match 1662, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_hybrid_bandit_m121_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_voronoi_parti_m53_s1.py (gen4)
# born: 2026-05-29T23:38:13Z

import numpy as np
import math
from typing import Iterable, Tuple, Union, List, Dict
from datetime import date, datetime

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7


def weekday_counts(
    dates: Iterable[Union[date, Tuple[int, int, int]]],
) -> np.ndarray:
    counts = np.zeros(7, dtype=int)
    for d in dates:
        if isinstance(d, date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        w = (date(y, m, day).weekday() + 1) % 7
        counts[w] += 1
    return counts


def gini_coefficient(vec: np.ndarray) -> float:
    if vec.size == 0:
        return 0.0
    if np.any(vec < 0):
        raise ValueError("Gini undefined for negative values")
    sorted_vec = np.sort(vec.astype(float))
    n = vec.size
    cumulative = np.cumsum(sorted_vec)
    sum_y = cumulative[-1]
    if sum_y == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / sum_y) / n
    return float(gini)


def weighted_difference_matrix(counts: np.ndarray) -> np.ndarray:
    c = counts.astype(float)
    outer = np.outer(c, c)
    idx = np.arange(7)
    diff = np.abs(idx[:, None] - idx[None, :])
    return outer * diff


def flatten_normalise_context(W: np.ndarray) -> np.ndarray:
    vec = W.ravel().astype(float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


class BetaPosterior:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = float(alpha)
        self.beta = float(beta)

    def update(self, reward: float) -> None:
        self.alpha += reward
        self.beta += 1.0 - reward

    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total != 0 else 0.0


def hybrid_reward(counts: np.ndarray) -> float:
    gini = gini_coefficient(counts)
    r1 = 1.0 - gini
    W = weighted_difference_matrix(counts)
    ctx = flatten_normalise_context(W)
    ssim_val = compute_ssim(ctx[: PROTOTYPE_VECTOR.size].tolist(),
                             PROTOTYPE_VECTOR.tolist(),
                             dynamic_range=1.0)
    r2 = (1.0 + ssim_val) / 2.0
    return float(r1 * r2)


def linucb_estimate(context: np.ndarray, posterior: BetaPosterior, alpha: float = 1.0) -> float:
    base = posterior.mean()
    exploration = alpha * math.sqrt(np.linalg.norm(context))
    return base + exploration


def hybrid_step(dates: Iterable[Union[date, Tuple[int, int, int]]],
                posterior: BetaPosterior) -> Tuple[float, float]:
    counts = weekday_counts(dates)
    R = hybrid_reward(counts)
    posterior.update(R)
    W = weighted_difference_matrix(counts)
    ctx = flatten_normalise_context(W)
    estimate = linucb_estimate(ctx, posterior, alpha=0.5)
    return R, estimate


class HybridModel:
    def __init__(self):
        self.posterior = BetaPosterior()

    def update(self, dates: Iterable[Union[date, Tuple[int, int, int]]]) -> Tuple[float, float]:
        return hybrid_step(dates, self.posterior)

    def get_estimate(self, context: np.ndarray) -> float:
        return linucb_estimate(context, self.posterior, alpha=0.5)


def main():
    model = HybridModel()
    dates = [(2022, 1, 1), (2022, 1, 2), (2022, 1, 3), (2022, 1, 4), (2022, 1, 5), (2022, 1, 6), (2022, 1, 7)]
    reward, estimate = model.update(dates)
    print(f"Reward: {reward}, Estimate: {estimate}")


if __name__ == "__main__":
    main()