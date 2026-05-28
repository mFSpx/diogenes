#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys, time, os
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from product_intake import intake_folder
from timeline_compiler import compile_timeline_from_custody

def test_inventory_times_and_filename_hints_become_source_backed_timeline(tmp_path):
    src=tmp_path/'corpus'; src.mkdir()
    f1=src/'2026-05-15-note.md'; f1.write_text('event')
    f2=src/'plain.md'; f2.write_text('plain')
    os.utime(f1,(1000000000,1000000000)); os.utime(f2,(1000000100,1000000100))
    intake=intake_folder(src, output_dir=tmp_path/'custody', max_files=10)
    tl=compile_timeline_from_custody(tmp_path/'custody'/'custody_package.json', source_root=src, output_dir=tmp_path/'timeline')
    assert tl['claim_count'] >= 3
    assert any(c['evidence_source']=='filename_date_hint' for c in tl['timeline'])
    assert any(c['evidence_source']=='filesystem_mtime' for c in tl['timeline'])
    assert all(c['source_ref'].get('custody_id') for c in tl['timeline'])
    assert tl['timeline'] == sorted(tl['timeline'], key=lambda c:(c['candidate_timestamp'], c['claim_id']))
    assert tl['receipt_path']
