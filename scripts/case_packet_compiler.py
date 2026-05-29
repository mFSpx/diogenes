#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, receipt, write_json, sha256_json


def _case_number(case_id: str, output_path: str | Path | None) -> str:
    import json
    if output_path:
        p = Path(output_path).resolve()
        for parent in [p.parent, *p.parents]:
            meta = parent / 'case_workspace.json'
            if meta.exists():
                try:
                    n = json.loads(meta.read_text(encoding='utf-8')).get('case_number')
                    if n:
                        return str(n)
                except json.JSONDecodeError:
                    pass
    return case_id

def compile_case_packet(*, case_id: str, custody: dict[str,Any], parse: dict[str,Any], timeline: dict[str,Any]|None, staging: dict[str,Any], graph: dict[str,Any], output_path: str|Path|None=None) -> dict[str,Any]:
    case_number = _case_number(case_id, output_path)
    packet={'schema':'lucidota.case_packet.v1','case_id':case_id,'case_number':case_number,'packet_id':'case:'+sha256_json({'case_id':case_id,'case_number':case_number,'custody':custody.get('package',{}),'parse':parse.get('package',{}),'staging':staging.get('claim_count'), 'graph':graph.get('candidate_count')})[:24], 'custody_refs':custody.get('package',{}), 'parse_refs':parse.get('package',{}), 'timeline_refs':{'timeline_path':timeline.get('timeline_path'), 'claim_count':timeline.get('claim_count')} if timeline else {}, 'claim_refs':{'staging_path':staging.get('staging_path'),'claim_count':staging.get('claim_count')}, 'graph_candidate_refs':{'graph_candidates_path':graph.get('graph_candidates_path'),'candidate_count':graph.get('candidate_count')}, 'receipt_refs':[x for x in [custody.get('receipt_path'),parse.get('receipt_path'),timeline.get('receipt_path') if timeline else None,staging.get('receipt_path'),graph.get('receipt_path')] if x]}
    out=Path(output_path) if output_path else ROOT/'05_OUTPUTS/case_packets'/f'{packet["packet_id"]}.json'
    write_json(out, packet)
    rec={'status':'PASSED','case_packet_path':rel(out),'packet':packet}
    rp=receipt('case_packet', rec, root='05_OUTPUTS/case_packets'); rec['receipt_path']=rel(rp); return rec
