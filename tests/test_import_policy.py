#!/usr/bin/env python3
from __future__ import annotations
import sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from import_policy import ImportPolicy
from product_intake import intake_folder
from case_workspace import CaseWorkspace

def test_case_policy_blocks_archives_caps_files_and_records_redaction(tmp_path):
    src=tmp_path/'drop'; src.mkdir()
    (src/'a.md').write_text('a'); (src/'b.txt').write_text('b'); (src/'c.py').write_text('c')
    with zipfile.ZipFile(src/'archive.zip','w') as z: z.writestr('x','x')
    policy=ImportPolicy(max_files=3,max_bytes=1000,allowed_extensions=('.md','.txt','.zip'),quarantine_extensions=('.zip',),ocr_allowed=False,model_allowed=False,export_redaction_level='strict')
    ws=CaseWorkspace.create('policy-case', base_dir=tmp_path/'cases'); ws.write_import_policy(policy)
    r=intake_folder(src, output_dir=ws.root/'custody', max_files=99, max_bytes=9999, import_policy=ws.load_import_policy())
    assert r['cursor']['processed']==3
    assert r['import_policy']['ocr_allowed'] is False
    assert r['import_policy']['export_redaction_level']=='strict'
    assert any(q['quarantine_reason']=='POLICY_QUARANTINE_EXTENSION' for q in r['quarantine'])
    assert not any(n['relative_path']=='c.py' for n in r['normal'])
