#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from case_workspace import CaseWorkspace
from spine_job_adapter import ABSURDJobAdapter
from work_order_importer import submit_pipeline_jobs
from absurd_worker_runner import drain

def test_worker_drain_executes_pipeline_stage_jobs_in_dependency_order(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    ws=CaseWorkspace.create('worker-case', base_dir=tmp_path/'cases')
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    submit_pipeline_jobs(adapter=adapter, case_id='worker-case', source_folder=str(src))
    res=drain(adapter, base_dir=tmp_path/'cases')
    assert res['processed_count']==6
    state=json.loads((ws.root/'absurd'/'jobs_state.json').read_text())
    assert all(j['state']=='COMPLETED' for j in state.values())
    assert (ws.root/'workspace'/'case_packet.json').exists()

def test_submit_pipeline_jobs_returns_queued_state(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    ws=CaseWorkspace.create('queued-case', base_dir=tmp_path/'cases')
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    jobs=submit_pipeline_jobs(adapter=adapter, case_id='queued-case', source_folder=str(src))
    assert len(jobs)==6
    assert all(j['state']=='QUEUED' for j in jobs)

def test_worker_drain_queues_created_ready_job_before_running(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    ws=CaseWorkspace.create('created-case', base_dir=tmp_path/'cases')
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    adapter.create_job(lane='pipeline.intake', payload={'case_id':'created-case','stage':'intake','source_folder':str(src)}, idempotency_key='created-intake')
    res=drain(adapter, base_dir=tmp_path/'cases')
    assert res['processed_count']==1
    state=json.loads((ws.root/'absurd'/'jobs_state.json').read_text())
    job=next(iter(state.values()))
    assert job['state']=='COMPLETED'
    events=[json.loads(line)['event'] for line in (ws.root/'absurd'/'jobs.jsonl').read_text().splitlines()]
    assert events[:3]==['CREATED','QUEUED','RUNNING']
