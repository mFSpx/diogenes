#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from export_bundle import export_bundle
from import_export_bundle import import_bundle_read_only

def test_export_import_roundtrip_read_only_workspace(tmp_path):
    f=tmp_path/'case.json'; f.write_text('{"case":"ok"}')
    exp=export_bundle(files=[f], output_path=tmp_path/'case.tar.gz')
    assert exp['status']=='PASSED'
    imp=import_bundle_read_only(tmp_path/'case.tar.gz', case_id='imported-case', base_dir=tmp_path/'cases')
    assert imp['status']=='PASSED'
    assert imp['read_only'] is True
    assert (tmp_path/'cases'/'imported-case'/'imported'/'export_manifest.json').exists()
    assert (tmp_path/'cases'/'imported-case'/'import_status.json').exists()
