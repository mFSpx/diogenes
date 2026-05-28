#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from pipeline_run_store import PipelineRunStore

def test_pipeline_run_store_records_stage_lifecycle(tmp_path):
    store=PipelineRunStore(tmp_path/'runs')
    store.start_stage('run-1','intake',{'source_folder':'x'})
    store.complete_stage('run-1','intake',{'package':{},'receipt_path':'r.json'},'r.json')
    state=store.load('run-1')
    assert state['stages']['intake']['status']=='PASSED'
    assert state['stages']['intake']['receipt_path']=='r.json'
    assert (tmp_path/'runs'/'run-1'/'events.jsonl').exists()
