#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from spine_common import ROOT, write_json, receipt, rel, sha256_json
from document_parse_router import parse_file
from text_chunker import chunk_text

def _load_jsonl(path: Path) -> list[dict[str,Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def parse_custody_package(package_path: str|Path, *, source_root: str|Path|None=None, output_dir: str|Path|None=None, max_chunk_chars:int=500) -> dict[str,Any]:
    pkg= json.loads(Path(package_path).read_text(encoding='utf-8')) if Path(package_path).is_file() else package_path
    if not isinstance(pkg, dict): raise ValueError('package_path must be custody_package.json path')
    src=Path(source_root or pkg['source_root'])
    out=Path(output_dir) if output_dir else ROOT/'05_OUTPUTS/product_parse'/sha256_json(pkg)[:12]
    out.mkdir(parents=True, exist_ok=True)
    normal=_load_jsonl(ROOT/pkg['normal_manifest_path'])
    quarantine=_load_jsonl(ROOT/pkg['quarantine_manifest_path'])
    parse_records=[]; chunks=[]; blocked=[]
    for row in normal + quarantine:
        p=src/row['relative_path']
        try:
            parsed=parse_file(p, custody_ref=row)
        except Exception as e:
            parsed={'source_path':rel(p),'custody_id':row.get('occurrence_id'),'content_id':row.get('content_id'),'status':'FAILED','text':'','parse_method':'exception','error':repr(e)}
        parse_records.append(parsed)
        if parsed['status']=='PARSED' and parsed.get('text'):
            chunks.extend(chunk_text(parsed['text'], source_ref={'custody_id':parsed.get('custody_id'),'source_path':parsed['source_path'],'source_sha256':parsed['source_sha256']}, max_chars=max_chunk_chars))
        if parsed['status']!='PARSED': blocked.append(parsed)
    with (out/'parse_records.jsonl').open('w',encoding='utf-8') as f:
        for r in parse_records: f.write(json.dumps(r,sort_keys=True)+'\n')
    with (out/'chunks.jsonl').open('w',encoding='utf-8') as f:
        for c in chunks: f.write(json.dumps(c,sort_keys=True)+'\n')
    with (out/'blocked_parse_records.jsonl').open('w',encoding='utf-8') as f:
        for b in blocked: f.write(json.dumps(b,sort_keys=True)+'\n')
    result={'schema':'lucidota.product.parse_package.v1','source_package':rel(package_path),'parse_records_path':rel(out/'parse_records.jsonl'),'chunks_path':rel(out/'chunks.jsonl'),'blocked_parse_records_path':rel(out/'blocked_parse_records.jsonl'),'parse_count':len(parse_records),'chunk_count':len(chunks),'blocked_count':len(blocked)}
    write_json(out/'parse_package.json', result)
    rec={'status':'PASSED','package':result,'parse_records':parse_records,'chunks':chunks,'blocked':blocked}
    rp=receipt('product_parse_pipeline', rec, root='05_OUTPUTS/product_parse'); rec['receipt_path']=rel(rp); return rec

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('package_path'); ap.add_argument('--source-root'); ap.add_argument('--output-dir'); ap.add_argument('--max-chunk-chars',type=int,default=500); a=ap.parse_args(); parse_custody_package(a.package_path, source_root=a.source_root, output_dir=a.output_dir, max_chunk_chars=a.max_chunk_chars); return 0
if __name__=='__main__': raise SystemExit(main())
