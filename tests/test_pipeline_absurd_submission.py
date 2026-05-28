#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from kernel_control_packet import absurd_enqueue_packet

def test_product_pipeline_expands_to_ordered_absurd_jobs(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    base=tmp_path/'cases'
    subprocess.run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'new-case','absurd-case','--max-files','10'],cwd=ROOT,check=True,capture_output=True,text=True)
    packet=absurd_enqueue_packet(queue_name='pipeline', lane='product_pipeline', source_path=str(src), idempotency_key='absurd-case', authorized_by='operator_cli')
    packet_path=tmp_path/'packet.json'; packet_path.write_text(json.dumps(packet))
    proc=subprocess.run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'submit-absurd','absurd-case','--source',str(src),'--control-packet',str(packet_path)],cwd=ROOT,check=True,capture_output=True,text=True)
    out=json.loads(proc.stdout)
    assert out['job_count']==6
    state=json.loads((ROOT/out['absurd_state']).read_text())
    assert len(state)==6
    jobs=sorted(state.values(), key=lambda j: ['intake','parse','timeline','staging','graph_candidate','case_packet'].index(j['payload']['stage']))
    assert jobs[0]['payload']['stage']=='intake'
    assert jobs[0]['depends_on']==[]
    assert all(j['state']=='QUEUED' for j in jobs)
    assert jobs[-1]['payload']['stage']=='case_packet'
    assert jobs[-1]['depends_on']==[jobs[-2]['job_id']]
