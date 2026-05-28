#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from export_bundle import export_bundle, verify_bundle

def test_export_bundle_hashes_and_blocks_secret_fixture(tmp_path):
    clean=tmp_path/'case.json'; clean.write_text('{"case":"ok"}')
    res=export_bundle(files=[clean], output_path=tmp_path/'case.tar.gz')
    assert res['status']=='PASSED'
    assert verify_bundle(tmp_path/'case.tar.gz')['status']=='PASSED'
    secret=tmp_path/'secret.env'; secret.write_text('API_KEY=abcd1234SECRET')
    blocked=export_bundle(files=[secret], output_path=tmp_path/'bad.tar.gz')
    assert blocked['status']=='BLOCKED'
    assert blocked['error']=='SECRET_SCAN_FAILED'
