#!/usr/bin/env python3
"""Extract deterministic topology findings from existing design_atom rows."""
from __future__ import annotations
import argparse, json, os, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/phase05'
SCHEMAS=[ROOT/'06_SCHEMA/030_phase05_brain_archaeology.sql', ROOT/'06_SCHEMA/083_topology_finding_extractor.sql']
PATTERNS=[
 ('infinite_sink', re.compile(r'\binfinite\s+sink\b|resource-asymmetrical|unreciprocated rescue', re.I), 'api_rate_limiting'),
 ('anchor_weight', re.compile(r'\banchor\s+weight\b|self-preservation override', re.I), 'resource_flow_review'),
 ('client_server', re.compile(r'\bclient/server\b|client\s*-\s*server', re.I), 'resource_flow_review'),
 ('legacy_client', re.compile(r'\blegacy\s+client\b', re.I), 'server_wipe'),
 ('transactional_isolation', re.compile(r'transactional isolation|boundary assertion', re.I), 'boundary_assertion'),
 ('apex_peer_candidate', re.compile(r'apex peer|peer-to-peer|peer to peer', re.I), 'peer_to_peer_probe'),
]
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def db(a): return a.database_url or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def rel(p):
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)
def write(n,d):
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'topology_finding_extractor_{n}_{stamp()}.json'; d.setdefault('generated_at',now()); d['report_path']=rel(p); p.write_text(json.dumps(d,indent=2,sort_keys=False,default=str)); print(f'REPORT_PATH={rel(p)}'); return p
def init_schema(a):
    if a.execute:
      with psycopg.connect(db(a)) as c:
        with c.cursor() as cur:
          for s in SCHEMAS: cur.execute(s.read_text())
        c.commit()
    write('init_schema_execute' if a.execute else 'init_schema_dry_run', {'action':'init_schema','execute_performed':bool(a.execute),'schemas':[rel(s) for s in SCHEMAS]}); return 0
def candidates(cur, limit):
    cur.execute('''SELECT atom_uuid::text,title,normalized_claim,raw_excerpt,authority_class,confidence_bps
      FROM lucidota_archaeology.design_atom ORDER BY created_at DESC LIMIT %s''',(limit,))
    out=[]
    for r in cur.fetchall():
      text=' '.join(str(r[k] or '') for k in ['title','normalized_claim','raw_excerpt'])
      for model,rx,proto in PATTERNS:
        if rx.search(text):
          out.append((dict(r),model,proto,text[:500]))
    return out
def extract(a):
    prepared=[]; inserted=0
    with psycopg.connect(db(a), row_factory=dict_row) as c:
      with c.cursor() as cur:
        prepared=candidates(cur,a.limit)
        if a.execute:
          for atom,model,proto,snippet in prepared:
            cur.execute('''INSERT INTO lucidota_archaeology.topology_finding(source_node_alias,target_node_alias,topology_model,finding_text,measured_dimensions,evidence,method,authority_class,confidence_bps,recommended_protocol,external_action_authorized)
              VALUES ('Operator','system_relation',%s,%s,%s::jsonb,%s::jsonb,'mixed',%s,%s,%s,false) RETURNING topology_uuid::text''', (model, f'Design atom indicates {model} topology signal.', json.dumps({'deterministic_pattern':model}), json.dumps([{'atom_uuid':atom['atom_uuid'],'excerpt':snippet}]), atom['authority_class'], min(9000,max(5000,int(atom['confidence_bps'] or 5000))), proto))
            top=cur.fetchone()['topology_uuid']
            cur.execute('''INSERT INTO lucidota_archaeology.topology_finding_source_receipt(atom_uuid,topology_uuid,topology_model,detail)
              VALUES (%s::uuid,%s::uuid,%s,%s::jsonb) ON CONFLICT(atom_uuid,topology_model) DO NOTHING RETURNING receipt_uuid::text''',(atom['atom_uuid'],top,model,json.dumps({'script':'scripts/topology_finding_extractor.py'})))
            if cur.fetchone(): inserted += 1
          c.commit()
    report={'action':'extract','execute_performed':bool(a.execute),'db_writes_performed':bool(a.execute),'graph_writes_performed':False,'atoms_scanned_limit':a.limit,'candidate_count':len(prepared),'inserted_findings':inserted,'candidates':[{'atom_uuid':x[0]['atom_uuid'],'model':x[1],'recommended_protocol':x[2]} for x in prepared[:10]],'blockers':[]}
    write('extract_execute' if a.execute else 'extract_dry_run', report); print(f'TOPOLOGY_CANDIDATES={len(prepared)}'); print(f'INSERTED_FINDINGS={inserted}'); return 0
def main():
    p=argparse.ArgumentParser(); p.add_argument('--database-url'); sub=p.add_subparsers(dest='cmd',required=True)
    sp=sub.add_parser('init-schema'); sp.add_argument('--execute',action='store_true'); sp.set_defaults(func=init_schema)
    sp=sub.add_parser('extract'); sp.add_argument('--execute',action='store_true'); sp.add_argument('--limit',type=int,default=500); sp.set_defaults(func=extract)
    a=p.parse_args(); return a.func(a)
if __name__=='__main__': raise SystemExit(main())
