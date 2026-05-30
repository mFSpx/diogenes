# DARWIN HAMMER — match 4134, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_rbf_su_m1969_s2.py (gen4)
# parent_b: hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py (gen4)
# born: 2026-05-29T23:53:41Z

"""
Module hybrid_hybrid_rbf_doomsday_cale_hybrid_fold_change_d: A hybrid algorithm 
combining the radial-basis surrogate model from hybrid_hybrid_rbf_su_m1969_s2.py 
with the hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s5.py. The 
mathematical bridge between the two structures lies in the use of radial basis 
functions to model the variational free energy of the doomsday algorithm, and 
applying the NLMS adaptive filter to update the weight matrix of the ternary 
router. The radial basis function is used to evaluate the similarity between 
the input and output of the doomsday algorithm, providing a measure of the 
system's performance.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import math
import random

Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target):
    prediction = np.dot(W, x)
    return np.mean((prediction - target) ** 2)

def doomsday(year: int, month: int, day: int) -> int:
    """Return weekday index where 0=Sunday … 6=Saturday."""
    return (datetime(year, month, day).weekday() + 1) % 7

def date_features(year: int, month: int, day: int) -> np.ndarray:
    """
    Normalized feature vector for a calendar date.

    Features (order):
        0 : year scaled to [0,1] over [1900,2100]
        1 : sin(2π*month/12)
        2 : cos(2π*month/12)
        3 : sin(2π*day/31)
        4 : cos(2π*day/31)
        5 : constant bias term (1.0)
    """
    yr_min, yr_max = 1900.0, 2100.0
    year_scaled = (year - yr_min) / (yr_max - yr_min)
    month_scaled = np.sin(2 * np.pi * month / 12)
    month_cos_scaled = np.cos(2 * np.pi * month / 12)
    day_scaled = np.sin(2 * np.pi * day / 31)
    day_cos_scaled = np.cos(2 * np.pi * day / 31)
    return np.array([year_scaled, month_scaled, month_cos_scaled, day_scaled, day_cos_scaled, 1.0])

def hybrid_rbf_doomsday(year: int, month: int, day: int, W: np.ndarray) -> float:
    date_vec = date_features(year, month, day)
    prediction = np.dot(W, date_vec)
    target = doomsday(year, month, day)
    loss = ttt_loss(W, date_vec, np.array([target]))
    return loss

def update_W(W: np.ndarray, date_vec: np.ndarray, target: int, learning_rate: float) -> np.ndarray:
    prediction = np.dot(W, date_vec)
    error = prediction - target
    W -= learning_rate * np.outer(error, date_vec)
    return W

def main():
    year, month, day = 2022, 1, 1
    W = init_ttt(6, 6)
    date_vec = date_features(year, month, day)
    target = doomsday(year, month, day)
    loss = hybrid_rbf_doomsday(year, month, day, W)
    print(f"Initial loss: {loss}")
    W = update_W(W, date_vec, target, 0.01)
    loss = hybrid_rbf_doomsday(year, month, day, W)
    print(f"Loss after update: {loss}")

if __name__ == "__main__":
    main()