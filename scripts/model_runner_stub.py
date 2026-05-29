#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json, receipt, rel
from model_runner_config import validate_model_config

def run_stub_model(config: dict[str,Any], prompt: str) -> dict[str,Any]:
    v=validate_model_config(config)
    if not v['ok']: return {'status':'REJECTED','validation':v,'real_inference_performed':False}
    if config.get('backend')!='STUB': return {'status':'REJECTED','validation':{'ok':False,'error':'STUB_RUNNER_REQUIRES_STUB_BACKEND'},'real_inference_performed':False}
    output=f'STUB_MODEL_OUTPUT:{sha256_json({"prompt":prompt,"model":config["model_id"]})[:16]}'
    rec={'status':'PASSED','model_id':config['model_id'],'backend':'STUB','input_sha256':sha256_json({'prompt':prompt}),'output_sha256':sha256_json({'output':output}),'output':output,'real_inference_performed':False,'vram_budget_mb':config['requested_vram_mb']}
    rp=receipt('model_invocation',rec,root='05_OUTPUTS/model_runtime'); rec['receipt_path']=rel(rp); return rec
