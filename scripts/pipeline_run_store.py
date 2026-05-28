#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import ROOT, now, append_jsonl, write_json, rel

class PipelineRunStore:
    def __init__(self, root: str|Path|None=None):
        self.root = Path(root) if root else ROOT/'05_OUTPUTS/pipeline_runs'
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        d = self.root/run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id)/'state.json'

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id)/'events.jsonl'

    def load(self, run_id: str) -> dict[str, Any]:
        p=self.state_path(run_id)
        if not p.exists():
            return {'run_id':run_id,'created_at':now(),'updated_at':now(),'stages':{},'failed_stage':None,'completed':False}
        return json.loads(p.read_text(encoding='utf-8'))

    def save(self, state: dict[str, Any]) -> None:
        state['updated_at']=now()
        write_json(self.state_path(state['run_id']), state)

    def start_stage(self, run_id: str, stage_name: str, input_refs: dict[str, Any]) -> None:
        state=self.load(run_id)
        state['stages'].setdefault(stage_name,{})
        state['stages'][stage_name].update({'status':'RUNNING','started_at':now(),'input_refs':input_refs})
        state['failed_stage']=None
        self.save(state)
        append_jsonl(self.events_path(run_id), {'run_id':run_id,'stage_name':stage_name,'event':'STARTED','input_refs':input_refs,'at':now()})

    def complete_stage(self, run_id: str, stage_name: str, output_refs: dict[str, Any], receipt_path: str|None) -> None:
        state=self.load(run_id)
        state['stages'].setdefault(stage_name,{})
        state['stages'][stage_name].update({'status':'PASSED','completed_at':now(),'output_refs':output_refs,'receipt_path':receipt_path})
        state['failed_stage']=None
        self.save(state)
        append_jsonl(self.events_path(run_id), {'run_id':run_id,'stage_name':stage_name,'event':'PASSED','output_refs':output_refs,'receipt_path':receipt_path,'at':now()})

    def fail_stage(self, run_id: str, stage_name: str, error: str) -> None:
        state=self.load(run_id)
        state['stages'].setdefault(stage_name,{})
        state['stages'][stage_name].update({'status':'FAILED','failed_at':now(),'error':error})
        state['failed_stage']=stage_name
        self.save(state)
        append_jsonl(self.events_path(run_id), {'run_id':run_id,'stage_name':stage_name,'event':'FAILED','error':error,'at':now()})

    def mark_completed(self, run_id: str) -> None:
        state=self.load(run_id); state['completed']=True; state['failed_stage']=None; self.save(state)
