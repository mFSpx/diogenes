#!/usr/bin/env python3
"""Bounded KRAMPUSCHEWING inventory with safe hashing and error capture."""
from __future__ import annotations
import argparse, fnmatch, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_OUTPUTS/krampus_inventory'
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def rel(p:Path)->str:
    try: return str(p.resolve().relative_to(ROOT))
    except Exception: return str(p)
def sha_file(path:Path, max_bytes:int)->tuple[str|None,str]:
    try:
        st=path.stat()
        if st.st_size>max_bytes: return None,'HASH_SKIPPED_SIZE'
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest(),'HASHED'
    except Exception as exc:
        return None,f'HASH_ERROR:{type(exc).__name__}'
def allowed(rp:str, includes:list[str], excludes:list[str])->bool:
    if includes and not any(fnmatch.fnmatch(rp, pat) for pat in includes): return False
    if excludes and any(fnmatch.fnmatch(rp, pat) for pat in excludes): return False
    return True
def inventory(args)->dict[str,Any]:
    target=(ROOT/args.target).resolve() if not Path(args.target).is_absolute() else Path(args.target).resolve()
    includes=args.include or []
    excludes=args.exclude or []
    params={'target':str(target),'max_files':args.max_files,'max_bytes':args.max_bytes,'include':includes,'exclude':excludes}
    inv_key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    jsonl_path=Path(args.jsonl_output) if args.jsonl_output else OUT/f'krampus_bounded_inventory_{stamp()}.jsonl'
    if not jsonl_path.is_absolute(): jsonl_path=ROOT/jsonl_path
    counts={'seen':0,'emitted':0,'errors':0,'hashed':0,'hash_skipped':0,'dirs_skipped':0}
    errors=[]
    suffixes={}
    with jsonl_path.open('w',encoding='utf-8') as out:
        if not target.exists():
            errors.append({'path':str(target),'error':'TARGET_MISSING'}); counts['errors']+=1
        else:
            for dirpath, dirnames, filenames in os.walk(target):
                # never descend common toxic/cache dirs by default
                dirnames[:] = [d for d in dirnames if d not in {'.git','node_modules','__pycache__','.venv','venv','target','.cache'}]
                for name in filenames:
                    counts['seen']+=1
                    path=Path(dirpath)/name
                    rp=rel(path)
                    if not allowed(rp,includes,excludes): continue
                    rec={'schema':'diogenes.krampus_inventory.entry.v1','inventory_key':inv_key,'path':rp,'suffix':path.suffix.lower() or '[none]','status':'OK','size_bytes':None,'sha256':None,'hash_status':None,'error':None,'mtime':None}
                    try:
                        st=path.stat(); rec['size_bytes']=st.st_size; rec['mtime']=datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat().replace('+00:00','Z')
                        rec['sha256'],rec['hash_status']=sha_file(path,args.max_bytes)
                        if rec['hash_status']=='HASHED': counts['hashed']+=1
                        else: counts['hash_skipped']+=1
                    except Exception as exc:
                        rec['status']='ERROR'; rec['error']=repr(exc); counts['errors']+=1; errors.append({'path':rp,'error':repr(exc)})
                    suffixes[rec['suffix']]=suffixes.get(rec['suffix'],0)+1
                    out.write(json.dumps(rec,sort_keys=True)+'\n')
                    counts['emitted']+=1
                    if counts['emitted']>=args.max_files: break
                if counts['emitted']>=args.max_files: break
    receipt={'schema':'diogenes.krampus_inventory.receipt.v1','generated_at':now(),'execute_performed':bool(args.execute),'dry_run':not args.execute,'target':rel(target),'inventory_key':inv_key,'max_files':args.max_files,'max_bytes':args.max_bytes,'include':includes,'exclude':excludes,'jsonl_output':rel(jsonl_path),'counts':counts,'suffixes':dict(sorted(suffixes.items(), key=lambda kv:(-kv[1],kv[0]))[:50]),'errors':errors[:50],'idempotency_basis':params,'status':'PASSED' if counts['errors']==0 or counts['emitted']>0 else 'FAILED'}
    receipt_path=OUT/f'krampus_bounded_inventory_receipt_{stamp()}.json'
    receipt['receipt_path']=rel(receipt_path); receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt
def main():
    p=argparse.ArgumentParser(); p.add_argument('--target',default='KRAMPUSCHEWING'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--execute',action='store_true'); p.add_argument('--max-files',type=int,default=500); p.add_argument('--max-bytes',type=int,default=1024*1024); p.add_argument('--include',action='append'); p.add_argument('--exclude',action='append'); p.add_argument('--jsonl-output')
    a=p.parse_args(); r=inventory(a); print('RECEIPT_PATH='+r['receipt_path']); print('JSONL_OUTPUT='+r['jsonl_output']); print('KRAMPUS_INVENTORY='+r['status']); return 0 if r['status']=='PASSED' else 2
if __name__=='__main__': raise SystemExit(main())
