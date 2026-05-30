# DARWIN HAMMER — match 49, survivor 0
# gen: 1
# parent_a: doomsday_calendar.py (gen0)
# parent_b: gini_coefficient.py (gen0)
# born: 2026-05-29T23:23:58Z

from __future__ import annotations
import numpy as np
from collections.abc import Iterable
import datetime as dt
import math
import random
import sys
import pathlib

"""
This module fuses the doomsday calendar algorithm and the Gini coefficient calculation.
The mathematical bridge between the two structures lies in the application of the Gini coefficient to a set of time-series data, 
such as the sequence of weekdays over a given period. By treating the weekdays as values in a distribution, 
we can use the Gini coefficient to quantify the unevenness of the weekday distribution.

The governing equation of the doomsday calendar is integrated with the Gini coefficient calculation by using the doomsday function 
to generate a sequence of weekdays for a given period, and then applying the Gini coefficient calculation to this sequence.
"""

def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7

def gini_coefficient(values: Iterable[float]) -> float:
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))

def weekday_distribution(year: int, month: int, num_days: int) -> np.ndarray:
    weekdays = [doomsday(year, month, day) for day in range(1, num_days+1)]
    weekday_counts = np.zeros(7)
    for weekday in weekdays:
        weekday_counts[weekday] += 1
    return weekday_counts

def gini_weekday(year: int, month: int, num_days: int) -> float:
    weekday_counts = weekday_distribution(year, month, num_days)
    return gini_coefficient(weekday_counts)

def simulate_random_weekdays(num_days: int) -> np.ndarray:
    random_weekdays = np.random.randint(0, 7, num_days)
    weekday_counts = np.zeros(7)
    for weekday in random_weekdays:
        weekday_counts[weekday] += 1
    return weekday_counts

def gini_random_weekdays(num_days: int) -> float:
    random_weekday_counts = simulate_random_weekdays(num_days)
    return gini_coefficient(random_weekday_counts)

if __name__ == "__main__":
    year = 2022
    month = 6
    num_days = 30
    print(gini_weekday(year, month, num_days))
    print(gini_random_weekdays(num_days))