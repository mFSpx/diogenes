#!/usr/bin/env python3
from __future__ import annotations
import json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from product_intake import intake_folder

def test_messy_folder_becomes_deterministic_custody_package(tmp_path):
    src=tmp_path/'messy'; src.mkdir();
    (src/'note.md').write_text('hello')
    (src/'data.json').write_text('{"a":1}')
    (src/'paper.pdf').write_bytes(b'%PDF-1.4')
    (src/'image.png').write_bytes(b'PNG')
    (src/'movie.mp4').write_bytes(b'MP4')
    (src/'local.db').write_bytes(b'SQLite format 3')
    (src/'dup1.txt').write_text('same')
    (src/'dup2.txt').write_text('same')
    with zipfile.ZipFile(src/'archive.zip','w') as z: z.writestr('x.txt','x')
    r1=intake_folder(src, output_dir=tmp_path/'out1', max_files=50, max_bytes=100000)
    r2=intake_folder(src, output_dir=tmp_path/'out2', max_files=50, max_bytes=100000)
    assert r1['package']['normal_count'] == r2['package']['normal_count']
    assert r1['package']['quarantine_count'] == r2['package']['quarantine_count']
    assert r1['duplicates']
    assert any(x['kind']=='archive' for x in r1['quarantine'])
    assert any(x['kind']=='database' for x in r1['quarantine'])
    assert Path(ROOT/r1['package']['normal_manifest_path']).exists()
    assert Path(ROOT/r1['package']['quarantine_manifest_path']).exists()
    assert r1['receipt_path']
