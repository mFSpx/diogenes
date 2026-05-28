#!/usr/bin/env python3
from __future__ import annotations
import os

def validate_vram_budget(*, requested_vram_mb:int, available_vram_mb:int|None=None, reserve_mb:int|None=None) -> dict:
    if available_vram_mb is None:
        available_vram_mb = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
    if reserve_mb is None:
        reserve_mb = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))
    usable=max(0,available_vram_mb-reserve_mb)
    if requested_vram_mb > usable:
        return {'ok':False,'error':'VRAM_BUDGET_EXCEEDED','requested_vram_mb':requested_vram_mb,'usable_vram_mb':usable}
    return {'ok':True,'requested_vram_mb':requested_vram_mb,'usable_vram_mb':usable}
