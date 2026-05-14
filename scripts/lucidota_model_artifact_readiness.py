#!/usr/bin/env python3
"""Truthful local model artifact readiness check: registry intent vs actual weight files."""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Any
import psycopg

ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
MODEL_DIRS=[ROOT/'03_VAULT', ROOT/'04_RUNTIME', ROOT/'01_REPOS'/'llama.cpp'/'models', Path.home()/'.cache'/'huggingface']
WEIGHT_PATTERNS=['*.gguf','*.safetensors','pytorch_model*.bin','model*.bin']
VOCAB_RE=re.compile(r'ggml-vocab', re.I)


def weights():
    out=[]
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for pat in WEIGHT_PATTERNS:
            for p in d.rglob(pat):
                try: size=p.stat().st_size
                except OSError: continue
                out.append({'path':str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p),'bytes':size,'vocab_only':bool(VOCAB_RE.search(p.name))})
    return sorted(out, key=lambda x:-x['bytes'])


def registry():
    try:
        with psycopg.connect(DB, connect_timeout=3) as conn:
            rows=conn.execute("""
              SELECT model_id, role, parameter_count, quantization, source_url, local_path, benchmark_status, expected_vram_mb
              FROM lucidota_runtime.model_candidate ORDER BY model_id
            """).fetchall()
        return [dict(zip(['model_id','role','parameter_count','quantization','source_url','local_path','benchmark_status','expected_vram_mb'], r)) for r in rows]
    except Exception as exc:
        return [{'error':str(exc)[:200]}]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--json', action='store_true'); args=ap.parse_args()
    w=weights(); usable=[x for x in w if not x['vocab_only'] and x['bytes']>50_000_000]
    reg=registry()
    checks={
      'needle_six_registry': any(r.get('model_id')=='needle-26m' and r.get('role')=='router' for r in reg),
      'mamba_registry': any('mamba' in str(r.get('model_id','')).lower() for r in reg),
      'deepseek_registry': any('deepseek' in str(r.get('model_id','')).lower() for r in reg),
      'usable_weight_files_present': len(usable)>0,
      'deepseek_weight_present': any('deepseek' in x['path'].lower() for x in usable),
      'mamba_weight_present': any('mamba' in x['path'].lower() for x in usable),
      'needle_weight_present': any('needle' in x['path'].lower() for x in usable),
      'quality_400_600m_weight_candidates': [x for x in usable if 200_000_000 <= x['bytes'] <= 2_000_000_000],
    }
    report={
      'ok': True,
      'routing_registry_ready': checks['needle_six_registry'] and checks['mamba_registry'] and checks['deepseek_registry'],
      'artifact_ready': checks['usable_weight_files_present'] and checks['deepseek_weight_present'] and checks['mamba_weight_present'] and checks['needle_weight_present'],
      'zero_refusal_guarantee': False,
      'zero_refusal_note': 'Cannot guarantee model behavior without actual local artifacts and eval suite; no such guarantee is truthful.',
      'checks': checks,
      'registry': reg,
      'weight_files_top': w[:40],
      'usable_weight_files': usable[:40],
    }
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
    return 0
if __name__=='__main__': raise SystemExit(main())
