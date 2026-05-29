#!/usr/bin/env python3
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from spine_common import ROOT, rel, stamp
from product_intake import intake_folder
from product_parse_pipeline import parse_custody_package
from timeline_compiler import compile_timeline_from_custody
from chunk_to_staging import stage_chunks_file
from graph_candidate import candidates_from_file
from case_packet_compiler import compile_case_packet
from pipeline_contracts import get_contract
from pipeline_run_store import PipelineRunStore
from content_store import ContentStore
from case_workspace import CaseWorkspace

@dataclass
class StageResult:
    stage_name: str
    status: str
    output: dict[str,Any]
    receipt_path: str|None = None
    error: str|None = None

@dataclass
class PipelineRun:
    case_id: str
    output_dir: Path
    stages: list[StageResult] = field(default_factory=list)
    failed_stage: str|None = None
    def add(self, r: StageResult):
        self.stages.append(r)
        if r.status!='PASSED': self.failed_stage=r.stage_name

class LucidotaPipeline:
    def __init__(self, *, case_id: str|None=None, output_root: str|Path|None=None, run_id: str|None=None, run_store: PipelineRunStore|None=None):
        self.case_id=case_id or 'case-'+stamp()
        self.workspace=CaseWorkspace.create(self.case_id, base_dir=(Path(output_root).parent if output_root else ROOT/'05_OUTPUTS/cases'))
        self.run_id=run_id or self.case_id
        self.output_root=Path(output_root) if output_root else self.workspace.root/'workspace'
        self.output_root.mkdir(parents=True,exist_ok=True)
        self.run_store=run_store or PipelineRunStore(self.workspace.root/'runs')
        self.content_store=ContentStore(self.workspace.root/'content_store', case_id=self.case_id)
    def _stage(self, run: PipelineRun, name: str, input_payload: dict[str,Any], fn: Callable[[],dict[str,Any]]) -> dict[str,Any]:
        contract=get_contract(name)
        contract.validate_input(input_payload)
        state=self.run_store.load(self.run_id)
        existing=state.get('stages',{}).get(name,{})
        if existing.get('status')=='PASSED' and existing.get('output_refs'):
            return existing['output_refs']
        self.run_store.start_stage(self.run_id,name,input_payload)
        try:
            out=fn(); contract.validate_output(out); out=dict(out); out['content_ref']=self.content_store.put_json(out, media_type=f'application/vnd.lucidota.pipeline.{name}+json')['content_ref']; self.run_store.complete_stage(self.run_id,name,out,out.get('receipt_path')); run.add(StageResult(name,'PASSED',out,out.get('receipt_path'))); return out
        except Exception as e:
            self.run_store.fail_stage(self.run_id,name,repr(e)); out={'status':'FAILED','error':repr(e)}; run.add(StageResult(name,'FAILED',out,error=repr(e))); raise
    def run_fixture_pipeline(self, source_folder: str|Path, *, max_files:int=100) -> PipelineRun:
        run=PipelineRun(self.case_id,self.output_root)
        custody=self._stage(run,'intake',{'source_folder':str(source_folder),'max_files':max_files},lambda: intake_folder(source_folder, output_dir=self.output_root/'custody', max_files=max_files))
        parse=self._stage(run,'parse',{'package_path':str(self.output_root/'custody'/'custody_package.json'),'source_root':str(source_folder)},lambda: parse_custody_package(self.output_root/'custody'/'custody_package.json', source_root=source_folder, output_dir=self.output_root/'parse'))
        timeline=self._stage(run,'timeline',{'package_path':str(self.output_root/'custody'/'custody_package.json'),'source_root':str(source_folder)},lambda: compile_timeline_from_custody(self.output_root/'custody'/'custody_package.json', source_root=source_folder, output_dir=self.output_root/'timeline'))
        staging=self._stage(run,'staging',{'chunks_path':str(ROOT/parse['package']['chunks_path'])},lambda: stage_chunks_file(ROOT/parse['package']['chunks_path'], self.output_root/'claims'/'staging.jsonl'))
        graph=self._stage(run,'graph_candidate',{'staging_path':str(ROOT/staging['staging_path'])},lambda: candidates_from_file(ROOT/staging['staging_path'], self.output_root/'graph'/'graph_candidates.jsonl'))
        self._stage(run,'case_packet',{'case_id':self.case_id,'custody':custody,'parse':parse,'timeline':timeline,'staging':staging,'graph':graph},lambda: compile_case_packet(case_id=self.case_id,custody=custody,parse=parse,timeline=timeline,staging=staging,graph=graph,output_path=self.output_root/'case_packet.json'))
        self.run_store.mark_completed(self.run_id)
        (self.output_root/'pipeline_run.json').write_text(json.dumps({'case_id':run.case_id,'run_id':self.run_id,'stages':[s.__dict__ for s in run.stages],'failed_stage':run.failed_stage},indent=2,sort_keys=True),encoding='utf-8')
        return run
