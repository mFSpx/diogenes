#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/retention'
SCAN_ROOT=ROOT/'05_OUTPUTS'
EXCLUDE={'raw','tmp','.cache'}
def ts(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p:Path)->str:
    try: return str(p.resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
def classify(path:Path)->str:
    parts=path.relative_to(SCAN_ROOT).parts
    return parts[0] if parts else 'root'
def collect(limit_hash_bytes:int=8_000_000):
    records=[]; skipped=[]
    for path in SCAN_ROOT.rglob('*'):
        if not path.is_file(): continue
        if any(part in EXCLUDE for part in path.parts):
            skipped.append(rel(path)); continue
        try: st=path.stat()
        except OSError as exc:
            skipped.append(f'{rel(path)}::{exc}'); continue
        rec={'path':rel(path),'category':classify(path),'size_bytes':st.st_size,'mtime':datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat().replace('+00:00','Z'),'sha256':None,'hash_skipped':False}
        if st.st_size <= limit_hash_bytes:
            try: rec['sha256']=sha256_file(path)
            except OSError as exc: rec['hash_error']=str(exc)
        else:
            rec['hash_skipped']=True
        records.append(rec)
    return records, skipped
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); ap.add_argument('--jsonl',action='store_true'); args=ap.parse_args()
    records, skipped=collect(); counts=Counter(r['category'] for r in records); total=sum(r['size_bytes'] for r in records)
    payload={'schema':'lucidota.report_retention_index.v1','generated_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'execute_performed':args.execute,'scan_root':rel(SCAN_ROOT),'total_reports':len(records),'total_size_bytes':total,'category_counts':dict(sorted(counts.items())),'records':records,'skipped':skipped,'retention_policy':{'default':'retain','destructive_delete_allowed':False,'large_files':'indexed_without_hash_above_limit'}}
    OUT.mkdir(parents=True,exist_ok=True); p=OUT/f'report_retention_index_{ts()}.json'; payload['report_path']=rel(p)
    if args.execute: p.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('REPORT_PATH='+rel(p)); print('REPORT_RETENTION_INDEX='+('PASS' if records else 'EMPTY'))
    return 0
if __name__=='__main__': raise SystemExit(main())
