#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from kernel_control_packet import absurd_enqueue_packet

def run(cmd, expect=0):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    assert p.returncode==expect, (cmd,p.returncode,p.stdout,p.stderr)
    return json.loads(p.stdout)

def test_pipeline_submission_requires_kernel_control_packet(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    base=tmp_path/'cases'
    run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'new-case','auth-case','--max-files','10'])
    missing=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'submit-absurd','auth-case','--source',str(src)], expect=2)
    assert missing['error']=='MISSING_KERNEL_CONTROL_PACKET'
    bad=absurd_enqueue_packet(queue_name='pipeline', lane='other_lane', source_path=str(src), idempotency_key='auth-case', authorized_by='operator_cli')
    bad_path=tmp_path/'bad.json'; bad_path.write_text(json.dumps(bad))
    mismatch=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'submit-absurd','auth-case','--source',str(src),'--control-packet',str(bad_path)], expect=2)
    assert mismatch['error']=='CONTROL_PACKET_SCOPE_MISMATCH'
    good=absurd_enqueue_packet(queue_name='pipeline', lane='product_pipeline', source_path=str(src), idempotency_key='auth-case', authorized_by='operator_cli')
    good_path=tmp_path/'good.json'; good_path.write_text(json.dumps(good))
    ok=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'submit-absurd','auth-case','--source',str(src),'--control-packet',str(good_path)])
    assert ok['status']=='PASSED' and ok['job_count']==6
