#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

def redact_path(path: str) -> str:
    p=Path(path)
    if p.is_absolute(): return '<LOCAL_ROOT>/'+p.name
    return path
