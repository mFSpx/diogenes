#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_acceptance import make_self_fixture, run_acceptance

def test_local_product_acceptance_runs_end_to_end(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    res=run_acceptance(source=src, base_dir=tmp_path/'cases', case_id='acceptance')
    assert res['status']=='PASSED'
    assert res['drained']['processed_count']==6
    assert Path(res['case_packet']).exists()
    assert res['export']['status']=='PASSED'
    assert res['import']['status']=='PASSED'
    assert res['schema']=='lucidota.acceptance_result.v1'
    assert res['canonical_graph_writes_performed'] is False

def test_acceptance_cli_emits_report_path(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    proc=subprocess.run(
        [sys.executable,str(ROOT/'scripts/lucidota_acceptance.py'),'--source',str(src),'--base-dir',str(tmp_path/'cases'),'--case-id','acceptance-cli','--json'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    lines=proc.stdout.splitlines()
    payload=json.loads(next(line for line in lines if line.startswith('{')))
    assert payload['status']=='PASSED'
    assert payload['drained']['processed_count']==6
    assert any(line.startswith('REPORT_PATH=') for line in lines)
    assert any(line == 'LUCIDOTA_ACCEPTANCE=PASSED' for line in lines)

def test_acceptance_self_fixture_cli(tmp_path):
    proc=subprocess.run(
        [sys.executable,str(ROOT/'scripts/lucidota_acceptance.py'),'--self-fixture','--base-dir',str(tmp_path/'cases'),'--case-id','acceptance-self','--json'],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload=json.loads(next(line for line in proc.stdout.splitlines() if line.startswith('{')))
    assert payload['status']=='PASSED'
    assert payload['source'].endswith('_fixtures/acceptance-self')
    assert Path(payload['report_path']).exists() or (ROOT/payload['report_path']).exists()

def test_make_self_fixture_is_bounded(tmp_path):
    drop=make_self_fixture(base_dir=tmp_path/'cases', case_id='fixture-case')
    assert drop.name=='fixture-case'
    assert (drop/'acceptance-note.md').read_text(encoding='utf-8').startswith('Alice saw Evidence')

def test_acceptance_is_idempotent_on_same_case(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    first=run_acceptance(source=src, base_dir=tmp_path/'cases', case_id='same-case')
    second=run_acceptance(source=src, base_dir=tmp_path/'cases', case_id='same-case')
    assert first['status']=='PASSED'
    assert second['status']=='PASSED'
    assert second['drained']['processed_count']==0
    assert second['completed_pipeline_jobs']==6
    assert second['pipeline_complete'] is True
