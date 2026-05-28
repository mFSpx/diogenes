#!/usr/bin/env python3
from __future__ import annotations
import json, tarfile
from pathlib import Path
from typing import Any
from spine_common import sha256_bytes, write_json, rel
from secret_scanner import scan_paths

def export_bundle(*, files: list[str|Path], output_path: str|Path, manifest_extra: dict[str,Any]|None=None) -> dict[str,Any]:
    scan=scan_paths(files)
    if not scan['clean']:
        return {'status':'BLOCKED','error':'SECRET_SCAN_FAILED','scan':scan}
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'lucidota.export_bundle.v1','files':[], **(manifest_extra or {})}
    with tarfile.open(out,'w:gz') as tar:
        for f in files:
            p=Path(f); data=p.read_bytes(); arc=p.name
            tar.add(p,arcname=arc); manifest['files'].append({'arcname':arc,'sha256':sha256_bytes(data),'size_bytes':len(data)})
        mp=out.parent/'export_manifest.json'; write_json(mp,manifest); tar.add(mp,arcname='export_manifest.json')
    return {'status':'PASSED','export_path':rel(out),'manifest':manifest}

def verify_bundle(bundle_path: str|Path) -> dict[str,Any]:
    with tarfile.open(bundle_path,'r:gz') as tar:
        manifest=json.loads(tar.extractfile('export_manifest.json').read().decode())
        for f in manifest['files']:
            data=tar.extractfile(f['arcname']).read()
            if sha256_bytes(data)!=f['sha256']: return {'status':'FAILED','error':'HASH_MISMATCH','file':f['arcname']}
    return {'status':'PASSED','manifest':manifest}
