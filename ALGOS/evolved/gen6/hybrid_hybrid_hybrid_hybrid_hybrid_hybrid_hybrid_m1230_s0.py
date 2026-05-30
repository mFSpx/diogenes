# DARWIN HAMMER — match 1230, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m626_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m612_s0.py (gen5)
# born: 2026-05-29T23:35:58Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

def doomsday_numpy(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            dt.datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7

def weekday_counts(dates):
    counts = np.zeros(7, dtype=int)
    for d in dates:
        counts[d.weekday() % 7] += 1
    return np.roll(counts, 1)

def gini_coefficient(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values.astype(float))
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    return ((np.arange(1, n+1) / n) - (cumulative / cumulative[-1])).sum()

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
    ssim = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return ssim

def expand(v, dim=100):
    e = np.zeros(dim)
    for i, val in enumerate(v):
        hash_val = hash((i, val)) % dim
        e[hash_val] += 1
    return e

def hybrid_operation(dates, prototype):
    weekday_counts_arr = weekday_counts(dates)
    gini = gini_coefficient(weekday_counts_arr)
    e_p = expand(prototype)
    ssim_scores = []
    for date in dates:
        e = expand([date.weekday()])
        ssim = compute_ssim(e, e_p)
        ssim_scores.append(ssim)
    ssim_arr = np.array(ssim_scores)
    reward = gini * (1 - ssim_arr.mean()) * (1 + np.std(ssim_arr))
    return reward

def add_laplace_noise(value, scale):
    return value + np.random.laplace(0, scale)

def regret_match_step(reward, regret, learning_rate):
    regret += learning_rate * (reward - regret)
    return regret

if __name__ == "__main__":
    dates = [dt.date(2022, 1, i) for i in range(1, 8)]
    prototype = [0.2, 0.5, 0.3]
    reward = hybrid_operation(dates, prototype)
    print(reward)
    noise_scale = 0.1
    noisy_reward = add_laplace_noise(reward, noise_scale)
    print(noisy_reward)
    regret = 0.5
    learning_rate = 0.1
    updated_regret = regret_match_step(noisy_reward, regret, learning_rate)
    print(updated_regret)