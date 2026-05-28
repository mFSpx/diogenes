#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil, tarfile, contextlib, io, sys
from pathlib import Path
from case_workspace import CaseWorkspace
from import_policy import ImportPolicy
from lucidota_pipeline import LucidotaPipeline
from pipeline_run_store import PipelineRunStore
from spine_job_adapter import ABSURDJobAdapter
from work_order_importer import submit_pipeline_jobs
from kernel_control_packet import verify_control_packet
from spine_common import rel, write_json

def cmd_new_case(a):
    ws=CaseWorkspace.create(a.case_id, base_dir=a.base_dir)
    policy=ImportPolicy(max_files=a.max_files,max_bytes=a.max_bytes,allowed_extensions=tuple(a.allowed_ext or []),quarantine_extensions=tuple(a.quarantine_ext or ['.zip','.db','.sqlite','.sqlite3']),ocr_allowed=a.ocr_allowed,model_allowed=a.model_allowed,export_redaction_level=a.redaction)
    ws.write_import_policy(policy)
    print(json.dumps({'status':'PASSED','case_id':ws.case_id,'case_number':ws.case_number,'workspace':rel(ws.root),'import_policy':policy.as_dict()},sort_keys=True)); return 0

def cmd_run_pipeline(a):
    ws=CaseWorkspace.create(a.case_id, base_dir=a.base_dir)
    policy=ws.load_import_policy()
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        run=LucidotaPipeline(case_id=a.case_id, output_root=ws.root/'workspace', run_id=a.run_id or a.case_id, run_store=PipelineRunStore(ws.root/'runs')).run_fixture_pipeline(a.source, max_files=policy.max_files)
    noisy=buf.getvalue().strip()
    if noisy: print(noisy, file=sys.stderr)
    print(json.dumps({'status':'PASSED' if not run.failed_stage else 'FAILED','case_id':a.case_id,'case_number':ws.case_number,'run_id':a.run_id or a.case_id,'failed_stage':run.failed_stage,'stage_count':len(run.stages),'workspace':rel(ws.root)},sort_keys=True)); return 0 if not run.failed_stage else 2

def cmd_resume(a):
    return cmd_run_pipeline(a)

def cmd_submit_absurd(a):
    if not a.dry_run and not a.control_packet:
        print(json.dumps({'status':'BLOCKED','error':'MISSING_KERNEL_CONTROL_PACKET'},sort_keys=True)); return 2
    if a.control_packet:
        packet=json.loads(Path(a.control_packet).read_text())
        ok,err=verify_control_packet(packet)
        if not ok:
            print(json.dumps({'status':'BLOCKED','error':'INVALID_KERNEL_CONTROL_PACKET','detail':err},sort_keys=True)); return 2
        payload=packet.get('payload',{})
        if payload.get('source_path') != a.source or payload.get('lane') != 'product_pipeline' or payload.get('queue_name') != 'pipeline':
            print(json.dumps({'status':'BLOCKED','error':'CONTROL_PACKET_SCOPE_MISMATCH'},sort_keys=True)); return 2
    ws=CaseWorkspace.create(a.case_id, base_dir=a.base_dir)
    adapter=ABSURDJobAdapter(ws.root/'absurd')
    jobs=submit_pipeline_jobs(adapter=adapter, case_id=a.case_id, source_folder=a.source)
    print(json.dumps({'status':'PASSED','case_id':a.case_id,'case_number':ws.case_number,'job_count':len(jobs),'jobs':[j['job_id'] for j in jobs],'absurd_state':rel(adapter.state_path)},sort_keys=True)); return 0

def cmd_status(a):
    ws=CaseWorkspace.create(a.case_id, base_dir=a.base_dir)
    store=PipelineRunStore(ws.root/'runs')
    state=store.load(a.run_id or a.case_id)
    print(json.dumps({'status':'PASSED','case_id':a.case_id,'case_number':ws.case_number,'run_state':state},sort_keys=True)); return 0

def cmd_export(a):
    ws=CaseWorkspace.create(a.case_id, base_dir=a.base_dir)
    export_dir=ws.root/'exports'; export_dir.mkdir(exist_ok=True)
    bundle=export_dir/f'{a.case_id}.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for name in ['case_workspace.json','import_policy.json']:
            p=ws.root/name
            if p.exists(): tar.add(p, arcname=name)
        p=ws.root/'workspace'/'case_packet.json'
        if p.exists(): tar.add(p, arcname='case_packet.json')
    print(json.dumps({'status':'PASSED','case_id':a.case_id,'case_number':ws.case_number,'export_path':rel(bundle)},sort_keys=True)); return 0

def cmd_language(a):
    from language_router import route_text, write
    r=write(route_text(a.text,a.channel,a.template,verbosity=a.verbosity,refresh_ontology=a.refresh_ontology,source_surfaces=['cli','operator']))
    print(json.dumps(r,sort_keys=True) if a.json else f"LANGUAGE_ROUTER=PASS REPORT_PATH={r['report_path']}"); return 0

def cmd_percyphon_route(a):
    from percyphon_kernel_bridge import build_bridge
    r=build_bridge(raw_command=a.raw_command, normalized_intent=a.normalized_intent, authority_class=a.authority_class, source=a.source, villagers=a.villager, fluid_slots=a.fluid_slots, ledger_path=a.ledger_path, event_log=a.event_log)
    print(json.dumps(r,sort_keys=True)); return 0 if r['status']=='ROUTED' else 2

def build_parser():
    ap=argparse.ArgumentParser(); ap.add_argument('--base-dir',default=None); sub=ap.add_subparsers(dest='cmd',required=True)
    n=sub.add_parser('new-case'); n.add_argument('case_id'); n.add_argument('--max-files',type=int,default=100); n.add_argument('--max-bytes',type=int,default=10_000_000); n.add_argument('--allowed-ext',action='append'); n.add_argument('--quarantine-ext',action='append'); n.add_argument('--ocr-allowed',action='store_true'); n.add_argument('--model-allowed',action='store_true'); n.add_argument('--redaction',choices=['none','local_paths','strict'],default='local_paths'); n.set_defaults(fn=cmd_new_case)
    for name,fn in [('run-pipeline',cmd_run_pipeline),('resume',cmd_resume)]:
        p=sub.add_parser(name); p.add_argument('case_id'); p.add_argument('--source',required=True); p.add_argument('--run-id'); p.set_defaults(fn=fn)
    j=sub.add_parser('submit-absurd'); j.add_argument('case_id'); j.add_argument('--source',required=True); j.add_argument('--control-packet'); j.add_argument('--dry-run',action='store_true'); j.set_defaults(fn=cmd_submit_absurd)
    s=sub.add_parser('status'); s.add_argument('case_id'); s.add_argument('--run-id'); s.set_defaults(fn=cmd_status)
    e=sub.add_parser('export'); e.add_argument('case_id'); e.set_defaults(fn=cmd_export)
    l=sub.add_parser('language'); l.add_argument('--text',required=True); l.add_argument('--channel',default='operator'); l.add_argument('--template'); l.add_argument('--verbosity',choices=['terse','brief','normal','full'],default='normal'); l.add_argument('--refresh-ontology',action='store_true'); l.add_argument('--json',action='store_true'); l.set_defaults(fn=cmd_language)
    pr=sub.add_parser('percyphon-route'); pr.add_argument('--raw-command',required=True); pr.add_argument('--normalized-intent',required=True); pr.add_argument('--authority-class',default='operator_authored_assertion'); pr.add_argument('--source',default='operator_cli'); pr.add_argument('--villager',action='append',default=[]); pr.add_argument('--fluid-slots',type=int,default=8); pr.add_argument('--ledger-path'); pr.add_argument('--event-log'); pr.set_defaults(fn=cmd_percyphon_route)
    return ap

def main():
    a=build_parser().parse_args(); return a.fn(a)
if __name__=='__main__': raise SystemExit(main())
