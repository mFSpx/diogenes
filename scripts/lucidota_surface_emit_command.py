#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/surfaces'

def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def sh(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true',default=False)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--database-url',default=os.environ.get('DATABASE_URL'))
    p.add_argument('--surface-key',default='marrow_loop_status')
    p.add_argument('--intent',default='surface.command.sample')
    p.add_argument('--payload-json',default='{}')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    payload=json.loads(args.payload_json)
    envelope={'protocol':'lucidota.command_envelope.v1','surface_key':args.surface_key,'intent':args.intent,'payload':payload,'canonical_target':'lucidota_control.conversation_command','direct_db_write':False,'direct_graph_write':False,'staging_only':True}
    report={'generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'mode':'execute' if args.execute else 'dry_run','database_url_source':'env' if args.database_url else 'unset','payload_sha256':sh(envelope),'command_envelope':envelope,'db_writes_performed':False,'graph_writes_performed':False,'execute_performed':False,'blockers':[]}
    if args.execute:
        report['blockers'].append('execute_not_implemented_until_conversation_command_reachable')
    out=OUT/f'surface_emit_command_{ts()}.json'; out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'REPORT_PATH={out.relative_to(ROOT)}')
    return 0 if not args.execute else 1
if __name__=='__main__': raise SystemExit(main())
