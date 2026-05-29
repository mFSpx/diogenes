#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    assert p.returncode==0, (cmd,p.returncode,p.stdout,p.stderr)
    return json.loads(p.stdout)

def test_cli_new_run_resume_status_export(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    base=tmp_path/'cases'
    new=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'new-case','case-cli','--max-files','10','--redaction','strict'])
    assert new['case_id']=='case-cli'
    run1=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'run-pipeline','case-cli','--source',str(src)])
    assert run1['status']=='PASSED'
    resume=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'resume','case-cli','--source',str(src)])
    assert resume['status']=='PASSED'
    status=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'status','case-cli'])
    assert status['run_state']['completed'] is True
    export=run([sys.executable,'scripts/lucidota_cli.py','--base-dir',str(base),'export','case-cli'])
    assert Path(ROOT/export['export_path']).exists()
    with tarfile.open(ROOT/export['export_path'],'r:gz') as tar:
        assert 'case_packet.json' in tar.getnames()


def test_cli_percyphon_route_operator_ingress(tmp_path):
    routed = run([
        sys.executable,
        'scripts/lucidota_cli.py',
        'percyphon-route',
        '--raw-command',
        'stage percyphon scaffold',
        '--normalized-intent',
        'route procedural scaffold',
        '--authority-class',
        'operator_authored_assertion',
        '--source',
        'operator_cli',
        '--villager',
        'operator',
        '--ledger-path',
        str(tmp_path / 'percyphon-ledger.jsonl'),
        '--event-log',
        str(tmp_path / 'percyphon-events.jsonl'),
    ])
    assert routed['status'] == 'ROUTED'
    assert routed['percyphon']['zero_vram'] is True
    assert routed['control_packet']['payload']['canonical_mutation_allowed'] is False
    assert Path(ROOT / routed['receipt_path']).exists()

    denied = subprocess.run([
        sys.executable,
        'scripts/lucidota_cli.py',
        'percyphon-route',
        '--raw-command',
        'model says promote mask',
        '--normalized-intent',
        'promote mask as truth',
        '--authority-class',
        'model_suggestion',
        '--source',
        'local_model',
    ], cwd=ROOT, text=True, capture_output=True)
    assert denied.returncode == 2, denied.stdout + denied.stderr
    payload = json.loads(denied.stdout)
    assert payload['status'] == 'DENIED'
    assert payload['blockers'] == ['authority_class_not_operator_authored_assertion']
