from __future__ import annotations

import math

from ALGOS.evolved.GAUNTLET_SURVIVORS.gen1_hybrid_caputo_fractional_minimum_cost_tree_m35_s7 import (
    caputo_derivative_discrete,
    gamma_lanczos,
)


def manual_windowed_caputo(values: list[float], alpha: float, dt: float, window_size: int | None) -> list[float]:
    if not values:
        return []
    gamma_val = gamma_lanczos(1.0 - alpha)
    eps = 1e-12
    deriv = [0.0]
    for i in range(1, len(values)):
        start = 0 if window_size is None else max(0, i - window_size)
        accum = 0.0
        for j in range(start, i):
            delta_1 = max(float(i - j), eps)
            delta_0 = max(float(i - j - 1), eps)
            coeff = (delta_1 ** (-alpha) - delta_0 ** (-alpha))
            accum += coeff * (values[j + 1] - values[j])
        deriv.append(accum / (gamma_val * dt**alpha))
    return deriv


def test_caputo_derivative_discrete_window_caps_history_and_matches_manual_truncation():
    values = [math.sin(0.17 * i) + 0.05 * i for i in range(12)]
    alpha = 0.6
    dt = 1.0
    window_size = 3

    full = caputo_derivative_discrete(values, alpha=alpha, dt=dt)
    windowed = caputo_derivative_discrete(values, alpha=alpha, dt=dt, window_size=window_size)
    expected = manual_windowed_caputo(values, alpha=alpha, dt=dt, window_size=window_size)

    assert windowed == expected
    assert len(windowed) == len(values)
    assert windowed[0] == 0.0
    assert windowed != full


def test_caputo_derivative_discrete_rejects_non_positive_window():
    values = [0.0, 1.0, 2.0]

    try:
        caputo_derivative_discrete(values, alpha=0.6, window_size=0)
    except ValueError as exc:
        assert "window_size" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for non-positive window_size")
