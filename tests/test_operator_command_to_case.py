#!/usr/bin/env python3
from __future__ import annotations
import tempfile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from operator_command_router import route_operator_command

def test_operator_command_becomes_case_pipeline_job_graph(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    result=route_operator_command('create case from folder and build packet', case_id='operator-case', source_folder=str(src), base_dir=tmp_path/'cases', ledger_path=tmp_path/'ledger.jsonl', event_log=tmp_path/'events.jsonl')
    assert result['status']=='PASSED'
    assert result['cep']['command_id'].startswith('cep:')
    assert result['route']['route_plan']['status']=='ROUTED'
    assert result['job_count']==6
    assert (tmp_path/'cases'/'operator-case'/'absurd'/'jobs_state.json').exists()
