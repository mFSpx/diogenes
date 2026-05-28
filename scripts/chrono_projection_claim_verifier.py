#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'05_OUTPUTS/chrono_ledger'
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p):
 try: return str(Path(p).resolve().relative_to(ROOT))
 except Exception: return str(p)
def db(a): return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--database-url'); a=ap.parse_args(); blockers=[]
 with psycopg.connect(db(a), row_factory=dict_row) as c:
  with c.cursor() as cur:
   cur.execute("SELECT to_regclass('lucidota_korpus.current_chrono_timeline_projection') reg"); exists=cur.fetchone()['reg'] is not None
   if not exists: blockers.append('PROJECTION_TABLE_MISSING'); total=missing=invalid=0; sample=[]
   else:
    cur.execute('SELECT count(*) total FROM lucidota_korpus.current_chrono_timeline_projection'); total=int(cur.fetchone()['total'])
    cur.execute('''SELECT p.file_uuid::text,p.selected_claim_uuid::text FROM lucidota_korpus.current_chrono_timeline_projection p LEFT JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid=p.selected_claim_uuid WHERE tc.claim_uuid IS NULL LIMIT 20'''); sample=[dict(r) for r in cur.fetchall()]; missing=len(sample)
    cur.execute('''SELECT count(*) invalid FROM lucidota_korpus.current_chrono_timeline_projection p JOIN lucidota_korpus.temporal_claim tc ON tc.claim_uuid=p.selected_claim_uuid WHERE coalesce(tc.invalid,false)=true'''); invalid=int(cur.fetchone()['invalid'])
 if missing: blockers.append('PROJECTION_SELECTED_CLAIM_MISSING')
 if invalid: blockers.append('PROJECTION_SELECTED_INVALID_CLAIM')
 payload={'action':'chrono_projection_claim_verify','projection_rows':total,'missing_claim_links':missing,'invalid_selected_claims':invalid,'sample':sample,'blockers':blockers,'status':'PASS' if not blockers else 'FAIL'}
 OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'chrono_projection_claim_verifier_{ts()}.json'; payload['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'); payload['report_path']=rel(p); p.write_text(json.dumps(payload,indent=2),encoding='utf-8'); print('REPORT_PATH='+rel(p)); print('CHRONO_PROJECTION_CLAIMS='+payload['status']); return 0 if not blockers else 4
if __name__=='__main__': raise SystemExit(main())
