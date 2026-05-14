#!/usr/bin/env python3
"""Gini inequality coefficient."""
from __future__ import annotations
from collections.abc import Iterable

def gini_coefficient(values: Iterable[float]) -> float:
    xs=sorted(float(x) for x in values)
    if not xs or sum(xs) == 0: return 0.0
    if xs[0] < 0: raise ValueError("values must be non-negative")
    n=len(xs)
    return sum((2*i-n-1)*x for i,x in enumerate(xs,1))/(n*sum(xs))
