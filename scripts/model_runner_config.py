#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Any
from gpu_runtime_budget import validate_vram_budget

def validate_model_config(config: dict[str,Any]) -> dict:
    for k in ['model_id','backend','requested_vram_mb']:
        if k not in config: return {'ok':False,'error':f'MISSING_{k.upper()}'}
    if config['backend'] not in {'STUB','llama.cpp','cohere','groq'}: return {'ok':False,'error':'UNSUPPORTED_BACKEND'}
    if config['backend'] in {'cohere','groq'}:
        return {'ok':True,'budget':{'ok':True,'backend':'remote_api','requested_vram_mb':0,'available_vram_mb':int(config.get('available_vram_mb') or os.environ.get('LUCIDOTA_VRAM_BUDGET_MB','4096'))}}
    budget=validate_vram_budget(requested_vram_mb=int(config['requested_vram_mb']), available_vram_mb=int(config.get('available_vram_mb') or os.environ.get('LUCIDOTA_VRAM_BUDGET_MB','4096')))
    if not budget['ok']: return budget
    return {'ok':True,'budget':budget}
