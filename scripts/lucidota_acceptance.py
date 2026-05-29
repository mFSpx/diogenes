#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from case_workspace import CaseWorkspace
from spine_job_adapter import ABSURDJobAdapter
from work_order_importer import submit_pipeline_jobs
from absurd_worker_runner import drain
from case_dashboard_data import compile_dashboard_data
from case_packet_renderer import render_case_packet
from export_bundle import export_bundle, verify_bundle
from import_export_bundle import import_bundle_read_only
from spine_common import now, rel, write_json

def make_self_fixture(*, base_dir: str|Path, case_id: str) -> Path:
    drop=Path(base_dir)/'_fixtures'/case_id
    drop.mkdir(parents=True,exist_ok=True)
    note=drop/'acceptance-note.md'
    note.write_text('Alice saw Evidence. Bob disputes missing acceptance evidence. Chrono should preserve the dated fixture.\\n',encoding='utf-8')
    return drop

def run_acceptance(*, source: str|Path, base_dir: str|Path, case_id: str='acceptance-case') -> dict:
    ws=CaseWorkspace.create(case_id, base_dir=base_dir)
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    submit_pipeline_jobs(adapter=adapter, case_id=case_id, source_folder=str(source))
    drained=drain(adapter, base_dir=base_dir)
    job_state=json.loads(adapter.state_path.read_text(encoding='utf-8')) if adapter.state_path.exists() else {}
    completed_pipeline_jobs=sum(1 for j in job_state.values() if str(j.get('lane','')).startswith('pipeline.') and j.get('state')=='COMPLETED')
    pipeline_complete=drained['processed_count']==6 or completed_pipeline_jobs==6
    packet_path=ws.root/'workspace'/'case_packet.json'
    packet=json.loads(packet_path.read_text())
    dashboard=compile_dashboard_data(case_id=case_id,custody={'package':packet.get('custody_refs',{}),'quarantine':[]},timeline={'claim_count':packet.get('timeline_refs',{}).get('claim_count',0),'timeline':[]},claims=[],contradictions=[],next_actions=[],receipts=packet.get('receipt_refs',[]),output_path=ws.root/'workspace'/'dashboard_data.json')
    render=render_case_packet(packet, md_path=ws.root/'workspace'/'case_packet.md', html_path=ws.root/'workspace'/'case_packet.html')
    exp=export_bundle(files=[packet_path, ws.root/'workspace'/'dashboard_data.json'], output_path=ws.root/'exports'/'case.tar.gz')
    ver=verify_bundle(ws.root/'exports'/'case.tar.gz') if exp['status']=='PASSED' else {'status':'SKIPPED'}
    imp=import_bundle_read_only(ws.root/'exports'/'case.tar.gz', case_id=case_id+'-imported', base_dir=Path(base_dir)) if ver['status']=='PASSED' else {'status':'SKIPPED'}
    report_path=ws.root/'acceptance_result.json'
    result={'schema':'lucidota.acceptance_result.v1','generated_at':now(),'status':'PASSED' if pipeline_complete and exp['status']=='PASSED' and ver['status']=='PASSED' and imp['status']=='PASSED' else 'FAILED','case_id':case_id,'source':str(source),'base_dir':str(base_dir),'drained':drained,'completed_pipeline_jobs':completed_pipeline_jobs,'pipeline_complete':pipeline_complete,'case_packet':str(packet_path),'dashboard':dashboard['dashboard_data_path'],'render':render,'export':exp,'verify':ver,'import':imp,'db_writes_performed':False,'canonical_graph_writes_performed':False,'report_path':rel(report_path)}
    write_json(report_path,result); return result

def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Run the local LUCIDOTA product acceptance harness end-to-end.')
    p.add_argument('--source',help='Folder containing input fixture/drop files.')
    p.add_argument('--self-fixture',action='store_true',help='Create and use a bounded local fixture under base-dir/_fixtures/case-id.')
    p.add_argument('--base-dir',default='05_OUTPUTS/acceptance_cases',help='Case workspace root.')
    p.add_argument('--case-id',default='acceptance-case')
    p.add_argument('--json',action='store_true')
    return p

def main() -> int:
    args=build_parser().parse_args()
    source=args.source
    if args.self_fixture:
        source=str(make_self_fixture(base_dir=args.base_dir,case_id=args.case_id))
    if not source:
        raise SystemExit('--source is required unless --self-fixture is set')
    result=run_acceptance(source=source,base_dir=args.base_dir,case_id=args.case_id)
    if args.json: print(json.dumps(result,sort_keys=True))
    print('REPORT_PATH='+result['report_path'])
    print('LUCIDOTA_ACCEPTANCE='+result['status'])
    return 0 if result['status']=='PASSED' else 4

if __name__=='__main__': raise SystemExit(main())
