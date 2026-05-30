# DARWIN HAMMER — match 49, survivor 1
# gen: 1
# parent_a: doomsday_calendar.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations
import numpy as np
import datetime as dt
from collections.abc import Iterable
import math
import random
import sys
import pathlib

"""This module introduces a novel HYBRID algorithm that mathematically fuses the governing equations of 
doomsday_calendar.py and gini_coefficient.py. The connection is established by considering the 
Gini coefficient as a measure of inequality in the distribution of weekdays over a given period, 
and the Doomsday algorithm as a means to determine the weekday of a specific date. The hybrid 
algorithm enables the investigation of temporal patterns and inequality in weekday distributions."""

def hybrid_doomsday_gini(year: int, month: int, day: int, values: Iterable[float]) -> float:
    """This function calculates the Gini coefficient of the provided values and then applies 
    the Doomsday algorithm to determine the weekday of the given date. The result is a weighted 
    sum of the Gini coefficient and the weekday, where the weights are determined by the weekday 
    and the day of the month."""
    doomsday = (dt.date(year, month, day).weekday() + 1) % 7
    gini = gini_coefficient(values)
    weight = doomsday / 7
    return weight * gini + (1 - weight) * doomsday

def gini_coefficient(values: Iterable[float]) -> float:
    """Calculates the Gini coefficient of a given set of non-negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: 
        return 0.0
    if xs[0] < 0: 
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def simulate_weekday_distribution(year: int, month: int, day: int, num_days: int) -> np.ndarray:
    """Simulates a weekday distribution over a given period and calculates the corresponding 
    Gini coefficient."""
    weekdays = []
    for i in range(num_days):
        date = dt.date(year, month, day) + dt.timedelta(days=i)
        weekdays.append((date.weekday() + 1) % 7)
    return np.array(weekdays)

def calculate_temporal_inequality(year: int, month: int, day: int, num_days: int) -> float:
    """Calculates the temporal inequality in a weekday distribution over a given period."""
    weekdays = simulate_weekday_distribution(year, month, day, num_days)
    values = [weekdays.count(i) for i in range(7)]
    return gini_coefficient(values)

if __name__ == "__main__":
    values = [10, 20, 30, 40, 50]
    print(gini_coefficient(values))
    print(hybrid_doomsday_gini(2024, 1, 1, values))
    print(calculate_temporal_inequality(2024, 1, 1, 365))