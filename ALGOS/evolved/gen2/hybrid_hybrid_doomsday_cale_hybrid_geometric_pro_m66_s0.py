# DARWIN HAMMER — match 66, survivor 0
# gen: 2
# parent_a: hybrid_doomsday_calendar_gini_coefficient_m49_s1.py (gen1)
# parent_b: hybrid_geometric_product_voronoi_partition_m4_s1.py (gen1)
# born: 2026-05-29T23:25:28Z

"""This module integrates the DARWIN HAMMER parents hybrid_doomsday_calendar_gini_coefficient_m49_s1.py and 
hybrid_geometric_product_voronoi_partition_m4_s1.py. The mathematical bridge is formed by considering the 
weekday distribution over a given period as a metric space, where the geometric product is used to compute 
the distances and orientations between days. The Gini coefficient from the Doomsday algorithm is used as a 
measure of inequality in the distribution, and the Voronoi partitioning is used to assign days to their 
nearest seeds or weekday groups based on their distances.

The governing equations of the Clifford algebra are used to compute the geometric product of multivectors, 
which are then used to represent days in the metric space. The Voronoi partitioning is used to assign days 
to their nearest seeds or weekday groups, and the geometric product is used to compute the distances and 
orientations between these days and seeds.

This module provides functions to compute the geometric product of multivectors, assign days to their nearest 
seeds using the Voronoi partitioning, and calculate the temporal inequality and weighted sum of the Gini 
coefficient and the weekday."""
import math
import numpy as np
import random
import sys
import pathlib

from collections.abc import Iterable
from datetime import date
import bisect

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, components):
        self.components = components


def gini_coefficient(values: Iterable[float]) -> float:
    """Calculates the Gini coefficient of a given set of non-negative values."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: 
        return 0.0
    if xs[0] < 0: 
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))


def geometric_product(day1: Multivector, day2: Multivector) -> Multivector:
    """Compute the geometric product of two days in the metric space."""
    components1 = day1.components
    components2 = day2.components
    result_components = {}
    for blade_a in components1:
        for blade_b in components2:
            result_blade, sign = _multiply_blades(blade_a, blade_b)
            if result_blade not in result_components:
                result_components[result_blade] = 0
            result_components[result_blade] += components1[blade_a] * components2[blade_b] * sign
    return Multivector(result_components)


def voronoi_partition(days: list, num_weekday_groups: int) -> list:
    """Assign days to their nearest weekday groups using the Voronoi partitioning."""
    days.sort(key=lambda x: x.weekday())
    centers = days[:num_weekday_groups]
    assignments = []
    for day in days:
        closest_center = min(centers, key=lambda x: geometric_product(day, x).components.get(frozenset(), 0))
        assignments.append(closest_center)
    return assignments


def calculate_temporal_inequality(year: int, month: int, day: int, num_days: int) -> float:
    """Calculate the temporal inequality in a weekday distribution over a given period."""
    days = []
    for i in range(num_days):
        date = date(year, month, day) + date.timedelta(days=i)
        days.append(date)
    gini = gini_coefficient([day.weekday() for day in days])
    return gini


def hybrid_operation(year: int, month: int, day: int, num_days: int) -> float:
    """Calculate the weighted sum of the Gini coefficient and the weekday."""
    days = []
    for i in range(num_days):
        date = date(year, month, day) + date.timedelta(days=i)
        days.append(date)
    gini = gini_coefficient([day.weekday() for day in days])
    weighted_sum = 0
    for i, day in enumerate(days):
        weight = bisect.bisect_left([d.weekday() for d in days], day.weekday()) / num_days
        weighted_sum += weight * gini + (1 - weight) * day.weekday()
    return weighted_sum


if __name__ == "__main__":
    print(calculate_temporal_inequality(2024, 1, 1, 100))
    print(hybrid_operation(2024, 1, 1, 100))