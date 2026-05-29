#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from gpu_runtime_budget import validate_vram_budget
from model_runner_config import validate_model_config
from model_runner_stub import run_stub_model

def test_model_lane_writes_stub_receipt_and_rejects_oversized_gtx1650_config():
    assert validate_vram_budget(requested_vram_mb=5000, available_vram_mb=4096)['ok'] is False
    bad=validate_model_config({'model_id':'x','backend':'STUB','requested_vram_mb':5000,'available_vram_mb':4096})
    assert bad['error']=='VRAM_BUDGET_EXCEEDED'
    res=run_stub_model({'model_id':'fixture.gguf','backend':'STUB','requested_vram_mb':512,'available_vram_mb':4096}, 'hello')
    assert res['status']=='PASSED'
    assert res['real_inference_performed'] is False
    assert res['backend']=='STUB'
    assert res['receipt_path']
