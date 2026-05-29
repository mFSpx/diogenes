#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any
from case_workspace import CaseWorkspace
from product_intake import intake_folder
from product_parse_pipeline import parse_custody_package
from timeline_compiler import compile_timeline_from_custody
from chunk_to_staging import stage_chunks_file
from graph_candidate import candidates_from_file
from case_packet_compiler import compile_case_packet
from spine_common import ROOT

KNOWN_LANES={'pipeline.intake','pipeline.parse','pipeline.timeline','pipeline.staging','pipeline.graph_candidate','pipeline.case_packet'}

def dispatch(lane: str, payload: dict[str,Any], *, base_dir: str|Path|None=None) -> dict[str,Any]:
    if lane not in KNOWN_LANES:
        raise ValueError(f'unknown lane: {lane}')
    case_id=payload['case_id']; source=Path(payload['source_folder']); ws=CaseWorkspace.create(case_id, base_dir=base_dir); out=ws.root/'workspace'
    if lane=='pipeline.intake': return intake_folder(source, output_dir=out/'custody', max_files=ws.load_import_policy().max_files)
    if lane=='pipeline.parse': return parse_custody_package(out/'custody'/'custody_package.json', source_root=source, output_dir=out/'parse')
    if lane=='pipeline.timeline': return compile_timeline_from_custody(out/'custody'/'custody_package.json', source_root=source, output_dir=out/'timeline')
    if lane=='pipeline.staging': return stage_chunks_file(ROOT/json_path(out/'parse'/'parse_package.json','chunks_path'), out/'claims'/'staging.jsonl')
    if lane=='pipeline.graph_candidate': return candidates_from_file(out/'claims'/'staging.jsonl', out/'graph'/'graph_candidates.jsonl')
    if lane=='pipeline.case_packet':
        import json
        custody={'package':json.loads((out/'custody'/'custody_package.json').read_text()),'receipt_path':None}
        parse={'package':json.loads((out/'parse'/'parse_package.json').read_text()),'receipt_path':None}
        timeline=json.loads((out/'timeline'/'timeline.json').read_text())
        timeline={'timeline_path':str(out/'timeline'/'timeline.json'),'claim_count':timeline['claim_count'],'receipt_path':None}
        staging={'staging_path':str(out/'claims'/'staging.jsonl'),'claim_count':sum(1 for _ in (out/'claims'/'staging.jsonl').open()),'receipt_path':None}
        graph={'graph_candidates_path':str(out/'graph'/'graph_candidates.jsonl'),'candidate_count':sum(1 for _ in (out/'graph'/'graph_candidates.jsonl').open()),'receipt_path':None}
        return compile_case_packet(case_id=case_id,custody=custody,parse=parse,timeline=timeline,staging=staging,graph=graph,output_path=out/'case_packet.json')
    raise ValueError(lane)

def json_path(file: Path, key: str) -> Path:
    import json
    return Path(json.loads(file.read_text())[key])
