#!/usr/bin/env python3
"""LUCIDOTA model VRAM/loadout governor v0.

Advisory only: reads active loadout + GPU memory, estimates missing slot VRAM,
records a decision row, and refuses fake residency when budget is exceeded.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
DB=os.environ.get('DBOS_SYSTEM_DATABASE_URL','postgresql://mfspx@/lucidota_state')
SCHEMA=ROOT/'06_SCHEMA'/'002_model_runtime.sql'
GOVERNOR_SQL='''
CREATE TABLE IF NOT EXISTS lucidota_runtime.load_governor_decision (
  decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  loadout_id text NOT NULL,
  target_gpu text NOT NULL DEFAULT '',
  budget_vram_mb integer NOT NULL,
  observed_used_mb integer,
  observed_free_mb integer,
  estimated_required_mb integer NOT NULL,
  headroom_mb integer NOT NULL,
  decision text NOT NULL CHECK (decision IN ('allow','defer','reject')),
  rationale text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS load_governor_decision_loadout_idx
  ON lucidota_runtime.load_governor_decision (loadout_id, created_at DESC);
'''

def gpu_memory() -> dict:
    if not shutil.which('nvidia-smi'):
        return {'status':'missing'}
    r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return {'status':'error','stderr':r.stderr[-200:]}
    name,total,used,free=[x.strip() for x in r.stdout.strip().splitlines()[0].split(',')[:4]]
    return {'status':'ok','name':name,'total_mb':int(total),'used_mb':int(used),'free_mb':int(free)}

def estimate_mb(parameter_count: int|None, quantization: str|None, explicit: int|None) -> int:
    if explicit is not None:
        return int(explicit)
    if not parameter_count:
        return 128
    q=(quantization or '').lower()
    bytes_per_param=0.5 if '4' in q else 1.0 if '8' in q else 2.0
    return max(64, int(parameter_count * bytes_per_param / 1_000_000 * 1.25) + 96)

def evaluate(conn, loadout_id: str|None, reserve_mb: int) -> dict:
    conn.execute(SCHEMA.read_text())
    conn.execute(GOVERNOR_SQL)
    loadout=conn.execute('''
      SELECT loadout_id, target_gpu, budget_vram_mb FROM lucidota_runtime.resident_loadout
      WHERE active=true OR loadout_id=COALESCE(%s, loadout_id)
      ORDER BY active DESC, created_at DESC LIMIT 1
    ''',(loadout_id,)).fetchone()
    if not loadout:
        raise SystemExit('no resident loadout found')
    lid,target_gpu,budget=loadout
    rows=conn.execute('''
      SELECT s.slot_name, s.instance_count, s.expected_vram_mb, m.model_id, m.role, m.parameter_count, m.quantization, m.expected_vram_mb
      FROM lucidota_runtime.resident_loadout_slot s
      JOIN lucidota_runtime.model_candidate m USING(model_id)
      WHERE s.loadout_id=%s
      ORDER BY s.priority ASC, s.slot_name
    ''',(lid,)).fetchall()
    slots=[]; required=0
    for slot, count, slot_mb, model_id, role, params, quant, model_mb in rows:
        each=estimate_mb(params, quant, slot_mb if slot_mb is not None else model_mb)
        total=each*count
        required += total
        slots.append({'slot':slot,'model_id':model_id,'role':role,'instances':count,'estimated_each_mb':each,'estimated_total_mb':total})
    gpu=gpu_memory()
    observed_used=gpu.get('used_mb') if gpu.get('status')=='ok' else None
    observed_free=gpu.get('free_mb') if gpu.get('status')=='ok' else None
    effective_budget=min(budget, gpu.get('total_mb', budget) if gpu.get('status')=='ok' else budget)
    headroom=effective_budget - required - reserve_mb
    decision='allow' if headroom >= 0 else 'defer' if required <= effective_budget else 'reject'
    rationale=f"required={required}MB reserve={reserve_mb}MB budget={effective_budget}MB headroom={headroom}MB"
    row=conn.execute('''
      INSERT INTO lucidota_runtime.load_governor_decision
        (loadout_id,target_gpu,budget_vram_mb,observed_used_mb,observed_free_mb,estimated_required_mb,headroom_mb,decision,rationale,detail)
      VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING decision_id::text
    ''',(lid,target_gpu,effective_budget,observed_used,observed_free,required,headroom,decision,rationale,json.dumps({'slots':slots,'gpu':gpu,'reserve_mb':reserve_mb}))).fetchone()
    conn.commit()
    return {'ok': True, 'decision_id': row[0], 'loadout_id': lid, 'decision': decision, 'rationale': rationale, 'gpu': gpu, 'slots': slots}

def main():
    ap=argparse.ArgumentParser(prog='lucidota-model-governor')
    ap.add_argument('--loadout-id')
    ap.add_argument('--reserve-mb', type=int, default=512)
    ap.add_argument('--json', action='store_true')
    args=ap.parse_args()
    with psycopg.connect(DB) as conn:
        report=evaluate(conn,args.loadout_id,args.reserve_mb)
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report['decision'] in ('allow','defer') else 1
if __name__=='__main__': raise SystemExit(main())
