# DARWIN HAMMER — match 2697, survivor 2
# gen: 5
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s0.py (gen4)
# born: 2026-05-29T23:43:32Z

"""
This module introduces a novel hybrid algorithm that mathematically fuses the core topologies of 
two parent algorithms: hybrid_doomsday_calendar_gini_coefficient_m49_s3.py and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_distri_m145_s0.py. The mathematical bridge between their 
structures lies in the integration of the doomsday calendar and Gini coefficient from the first parent 
with the morphology analysis and sphericity index from the second parent. The resulting hybrid algorithm 
provides a comprehensive fusion of date analysis, inequality measurement, and physical property analysis.

The mathematical interface between the two parents is established through the use of a representative date 
to calculate the morphology of a physical object, where the Gini coefficient of the weekday counts is used 
as a weight to calculate the sphericity index of the physical object.

"""

import numpy as np
import datetime as dt
import math
import random
import sys
import pathlib

def doomsday_numpy(
    years: np.ndarray,
    months: np.ndarray,
    days: np.ndarray,
) -> np.ndarray:
    dates = np.stack([years, months, days], axis=-1).astype('datetime64[D]')
    np_weekday = dates.astype('datetime64[D]').astype('datetime64[ns]').astype('int64')
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (dt.datetime.utcfromtimestamp(int(d.astype('datetime64[s]'))).weekday() for d in flat),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7


def gini_coefficient_numpy(values: np.ndarray) -> float:
    if values.ndim != 1:
        raise ValueError("values must be a 1‑D array")
    xs = np.sort(values.astype(float))
    if xs.size == 0 or xs.sum() == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = xs.size
    i = np.arange(1, n + 1)  # 1‑based index
    numerator = np.sum((2 * i - n - 1) * xs)
    denominator = n * xs.sum()
    return numerator / denominator


def weekday_counts(dates: list) -> np.ndarray:
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt.date):
            y, m, day = d.year, d.month, d.day
        else:
            y, m, day = d
        years.append(y)
        months.append(m)
        days.append(day)
    years_np = np.array(years, dtype=np.int32)
    months_np = np.array(months, dtype=np.int32)
    days_np = np.array(days, dtype=np.int32)

    weekdays = doomsday_numpy(years_np, months_np, days_np)
    counts = np.bincount(weekdays, minlength=7)
    return counts.astype(int)


def morphology_sphericity(date: dt.date, length: float, width: float, height: float) -> float:
    counts = weekday_counts([date])
    gini = gini_coefficient_numpy(counts)
    sphericity = (length * width * height) ** (1.0 / 3.0) / length
    return gini * sphericity


def representative_date_morphology(dates: list, length: float, width: float, height: float) -> float:
    counts = weekday_counts(dates)
    gini = gini_coefficient_numpy(counts)
    sphericity = (length * width * height) ** (1.0 / 3.0) / length
    return gini * sphericity


def hybrid_operation(dates: list, length: float, width: float, height: float) -> tuple:
    gini = gini_coefficient_numpy(weekday_counts(dates))
    sphericity = representative_date_morphology(dates, length, width, height)
    return gini, sphericity


if __name__ == "__main__":
    dates = [dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)]
    length, width, height = 10.0, 5.0, 2.0
    gini, sphericity = hybrid_operation(dates, length, width, height)
    print(f"Gini coefficient: {gini}, Sphericity index: {sphericity}")