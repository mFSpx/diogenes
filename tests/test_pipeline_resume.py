#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_pipeline import LucidotaPipeline
from pipeline_run_store import PipelineRunStore

def test_pipeline_rerun_reuses_completed_stage_records(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice claims a fact.')
    store=PipelineRunStore(tmp_path/'runs')
    p1=LucidotaPipeline(case_id='case-a', run_id='run-a', output_root=tmp_path/'out', run_store=store)
    p1.run_fixture_pipeline(src, max_files=10)
    state1=store.load('run-a')
    first_receipt=state1['stages']['intake']['receipt_path']
    p2=LucidotaPipeline(case_id='case-a', run_id='run-a', output_root=tmp_path/'out', run_store=store)
    p2.run_fixture_pipeline(src, max_files=10)
    state2=store.load('run-a')
    assert state2['stages']['intake']['receipt_path']==first_receipt
    assert state2['completed'] is True
