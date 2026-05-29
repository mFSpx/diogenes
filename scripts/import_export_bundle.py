#!/usr/bin/env python3
from __future__ import annotations
import json, tarfile
from pathlib import Path
from typing import Any
from case_workspace import CaseWorkspace
from export_bundle import verify_bundle
from spine_common import write_json

def import_bundle_read_only(bundle_path: str|Path, *, case_id: str, base_dir: str|Path) -> dict[str,Any]:
    verified=verify_bundle(bundle_path)
    if verified['status']!='PASSED': return {'status':'FAILED','verify':verified}
    ws=CaseWorkspace.create(case_id, base_dir=base_dir, read_only=True)
    import_dir=ws.root/'imported'; import_dir.mkdir(exist_ok=True)
    with tarfile.open(bundle_path,'r:gz') as tar:
        tar.extractall(import_dir)
    write_json(ws.root/'import_status.json', {'status':'PASSED','read_only':True,'verify':verified})
    return {'status':'PASSED','case_id':case_id,'read_only':True,'workspace':str(ws.root),'verify':verified}
