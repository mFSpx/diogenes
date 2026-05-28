#!/usr/bin/env python3
"""Doomsday/calendar weekday helper, 0=Sunday..6=Saturday."""
from __future__ import annotations
import datetime as dt

def doomsday(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day).weekday() + 1) % 7
