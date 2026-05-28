#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from spine_common import ROOT, now, rel, sha256_bytes, sha256_json, write_json, receipt
from krampus_extension_policy import classify_path
from source_bundle import source_bundle_from_intake
from import_policy import ImportPolicy

def _safe_read(path: Path, max_bytes: int) -> tuple[bytes|None,str|None]:
    try:
        size=path.stat().st_size
        if size>max_bytes: return None, 'MAX_BYTES_EXCEEDED'
        return path.read_bytes(), None
    except Exception as e:
        return None, f'READ_ERROR:{type(e).__name__}'

def intake_folder(source: str|Path, *, output_dir: str|Path|None=None, max_files: int=100, max_bytes: int=10_000_000, include: list[str]|None=None, exclude: list[str]|None=None, import_policy: ImportPolicy|dict[str,Any]|None=None) -> dict[str,Any]:
    src=Path(source)
    policy = ImportPolicy.from_dict(import_policy) if isinstance(import_policy, dict) else (import_policy or ImportPolicy(max_files=max_files, max_bytes=max_bytes))
    max_files = min(max_files, policy.max_files)
    max_bytes = min(max_bytes, policy.max_bytes)
    if not src.exists() or not src.is_dir(): raise ValueError(f'source folder missing: {source}')
    if src.name == 'KRAMPUSCHEWING' and not max_files: raise ValueError('max_files required for KRAMPUSCHEWING')
    out=Path(output_dir) if output_dir else ROOT/'05_OUTPUTS/product_intake'/sha256_bytes(str(src.resolve()).encode())[:12]
    out.mkdir(parents=True, exist_ok=True)
    normal=[]; quarantine=[]; duplicates={}; seen={}; cursor={'source':rel(src),'max_files':max_files,'processed':0,'complete':False}
    files=[]
    for p in sorted(x for x in src.rglob('*') if x.is_file() or x.is_symlink()):
        rp=str(p.relative_to(src))
        if exclude and any(Path(rp).match(e) for e in exclude): continue
        if include and not any(Path(rp).match(i) for i in include): continue
        files.append(p)
        if len(files)>=max_files: break
    for p in files:
        rp=str(p.relative_to(src)); ext_policy=classify_path(p); stat_size=0
        try: stat_size=p.stat().st_size
        except Exception: pass
        decision=policy.decision_for_path(p)
        b,err=_safe_read(p,max_bytes)
        row={'relative_path':rp,'absolute_path_redacted':'<LOCAL_ROOT>/'+rel(p),'size_bytes':stat_size,'kind':ext_policy['kind'],'lane':ext_policy['lane'],'occurrence_id':sha256_json({'source':rel(src),'path':rp}),'content_id':None,'sha256':None,'duplicate_of':None,'quarantine_reason':ext_policy['reason'] if ext_policy['quarantine'] else '', 'policy_decision':decision}
        if not decision['allowed']:
            row['quarantine_reason']=decision['reason']; quarantine.append(row); continue
        if b is None:
            row['quarantine_reason']=err or 'READ_ERROR'; quarantine.append(row); continue
        row['sha256']=sha256_bytes(b); row['content_id']='sha256:'+row['sha256']
        if row['sha256'] in seen:
            row['duplicate_of']=seen[row['sha256']]; duplicates.setdefault(seen[row['sha256']],[]).append(row['occurrence_id'])
        else: seen[row['sha256']]=row['occurrence_id']
        if decision['force_quarantine']:
            row['quarantine_reason']=decision['reason']
            quarantine.append(row)
        elif ext_policy['quarantine']: quarantine.append(row)
        else: normal.append(row)
    cursor.update({'processed':len(files),'complete':len(files)<max_files})
    package={'schema':'lucidota.product.custody_package.v1','generated_at':now(),'source_root':rel(src),'normal_manifest_path':rel(out/'normal_manifest.jsonl'),'quarantine_manifest_path':rel(out/'quarantine_manifest.jsonl'),'duplicate_index_path':rel(out/'duplicate_index.json'),'resume_cursor_path':rel(out/'resume_cursor.json'),'normal_count':len(normal),'quarantine_count':len(quarantine),'duplicate_groups':len(duplicates)}
    for name,rows in [('normal_manifest.jsonl',normal),('quarantine_manifest.jsonl',quarantine)]:
        with (out/name).open('w',encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    write_json(out/'duplicate_index.json',duplicates); write_json(out/'resume_cursor.json',cursor); write_json(out/'custody_package.json',package)
    rec={'status':'PASSED','package':package,'normal':normal,'quarantine':quarantine,'duplicates':duplicates,'cursor':cursor}
    rec['import_policy']=policy.as_dict()
    bundle=source_bundle_from_intake(rec, safety_policy=policy.as_dict())
    write_json(out/'source_bundle.json', bundle)
    rec['source_bundle']=bundle
    rec['package']['source_bundle_path']=rel(out/'source_bundle.json')
    rp=receipt('product_intake',rec,root='05_OUTPUTS/product_intake'); rec['receipt_path']=rel(rp); return rec

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('source'); ap.add_argument('--output-dir'); ap.add_argument('--max-files',type=int,default=100); ap.add_argument('--max-bytes',type=int,default=10_000_000); a=ap.parse_args(); intake_folder(a.source, output_dir=a.output_dir, max_files=a.max_files, max_bytes=a.max_bytes); return 0
if __name__=='__main__': raise SystemExit(main())
