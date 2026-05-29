#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import quality_work_order_compiler as qwo  # noqa:E402

def test_build_orders_turns_quality_rows_into_bounded_ouroboros_jobs(tmp_path:Path):
    src=tmp_path/'repair.jsonl'
    src.write_text('\n'.join(json.dumps(x) for x in [
        {'path':'scripts/a.py','verdict':'REPAIR','reason':'needs receipts','hash':'h1'},
        {'path':'scripts/b.py','verdict':'MERGE','reason':'duplicate','hash':'h2'},
    ])+'\n')
    rows=qwo.load_jsonl(src); orders=qwo.build_orders(rows,limit=2,receipt_root='05_OUTPUTS/qwo')
    assert [o['target_path'] for o in orders]==['scripts/a.py','scripts/b.py']
    assert all(o['handler']=='external_command' for o in orders)
    assert all(o['command'][1]=='scripts/lucidota_ouroboros_loop.py' for o in orders)
    assert all('--target' in o['command'] and '--receipt-root' in o['command'] for o in orders)

def test_execute_enqueue_uses_absurd_queue_spine_without_secret_leak(monkeypatch):
    calls=[]
    def fake_run(cmd, **kw):
        calls.append(cmd)
        class P: returncode=0; stdout='REPORT_PATH=05_OUTPUTS/absurd/fake.json\n'; stderr=''
        return P()
    monkeypatch.setattr(qwo.subprocess,'run',fake_run)
    order={'work_order_id':'qwo:test','handler':'external_command','command':[sys.executable,'scripts/lucidota_ouroboros_loop.py','--loops','1','--target','scripts/a.py'],'timeout_seconds':12}
    out=qwo.enqueue([order],queue='goal_swarm',workflow='quality_repair_queue_v1',timeout=12)
    assert out[0]['rc']==0
    cmd=' '.join(calls[0])
    assert 'scripts/absurd_queue_spine.py' in cmd and '--execute' in cmd
    assert 'GROQ_API_KEY' not in cmd and 'COHERE_API_KEY' not in cmd

def test_helper_stays_under_100_loc():
    lines=[l for l in Path(qwo.__file__).read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    assert len(lines)<=100
