# DARWIN HAMMER — match 14, survivor 0
# gen: 1
# parent_a: workshare_allocator.py (gen0)
# parent_b: doomsday_calendar.py (gen0)
# born: 2026-05-29T23:19:36Z

"""
This module is a fusion of workshare_allocator.py and doomsday_calendar.py.
The mathematical bridge between the two structures is the concept of allocation and distribution, 
where the workshare allocator distributes work units among different groups, 
and the doomsday calendar allocates days of the week to specific dates.
Here, we combine the two by allocating work units based on the day of the week, 
which is determined by the doomsday calendar algorithm.
"""

import numpy as np
import math
import random
import sys
from datetime import date
from pathlib import Path

GROUPS = ("codex", "groq", "cohere", "local_models")

def _pct(value: float) -> float:
    return round(float(value), 6)

def allocate_workshare(*, total_units: float, deterministic_target_pct: float = 90.0, groups: tuple[str, ...] = GROUPS) -> dict[str, float]:
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")
    if not groups:
        raise ValueError("groups required")
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    per_group = llm_units / len(groups)
    lanes = [
        {
            "group": group,
            "llm_units": _pct(per_group),
            "llm_share_pct": _pct(100.0 / len(groups)),
            "proof_required": True,
        }
        for group in groups
    ]
    return {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units),
        "lanes": lanes,
    }

def doomsday(year: int, month: int, day: int) -> int:
    return (date(year, month, day).weekday() + 1) % 7

def allocate_workshare_by_day(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    day_of_week = doomsday(year, month, day)
    allocation = allocate_workshare(total_units=total_units, deterministic_target_pct=deterministic_target_pct)
    allocation["day_of_week"] = day_of_week
    allocation["day_of_week_llm_units"] = allocation["llm_units"] * (day_of_week / 7)
    return allocation

def summarize_savings_by_day(*, total_units: float, year: int, month: int, day: int, deterministic_target_pct: float = 90.0) -> dict[str, float]:
    allocation = allocate_workshare_by_day(total_units=total_units, year=year, month=month, day=day, deterministic_target_pct=deterministic_target_pct)
    return {
        "day_of_week": allocation["day_of_week"],
        "day_of_week_llm_units": allocation["day_of_week_llm_units"],
        "baseline_llm_units": allocation["llm_units"],
        "token_savings_pct": _pct((allocation["total_units"] - allocation["llm_units"]) / allocation["total_units"] * 100.0),
    }

if __name__ == "__main__":
    allocation = allocate_workshare(total_units=100.0)
    print(allocation)
    day_allocation = allocate_workshare_by_day(total_units=100.0, year=2024, month=1, day=1)
    print(day_allocation)
    savings = summarize_savings_by_day(total_units=100.0, year=2024, month=1, day=1)
    print(savings)