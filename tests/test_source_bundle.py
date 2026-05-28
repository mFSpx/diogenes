#!/usr/bin/env python3
from __future__ import annotations
import sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from product_intake import intake_folder
from source_bundle import source_bundle_from_intake

def test_source_bundle_identity_counts_and_policy(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'a.md').write_text('a'); (src/'b.db').write_bytes(b'db'); (src/'dup1.txt').write_text('same'); (src/'dup2.txt').write_text('same')
    receipt=intake_folder(src, output_dir=tmp_path/'out', max_files=10)
    b=receipt['source_bundle']
    assert b['bundle_id'].startswith('bundle:')
    assert b['file_count']==4
    assert b['normal_count']>=2
    assert b['quarantine_count']>=1
    assert b['duplicate_groups']>=1
    assert b['safety_policy']['destructive_actions_allowed'] is False
    assert Path(ROOT/receipt['package']['source_bundle_path']).exists()
