#!/usr/bin/env python3
from __future__ import annotations
import sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from product_intake import intake_folder
from timeline_compiler import compile_timeline_from_custody
from event_candidate_registry import build_event_registry

def test_chrono_claims_become_event_candidates_and_timeline_uses_them(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); f=src/'2026-05-17-note.md'; f.write_text('event'); os.utime(f,(1000000000,1000000000))
    intake_folder(src, output_dir=tmp_path/'custody', max_files=10)
    tl=compile_timeline_from_custody(tmp_path/'custody'/'custody_package.json', source_root=src, output_dir=tmp_path/'timeline')
    assert tl['event_count'] >= 1
    assert Path(ROOT/tl['event_candidates_path']).exists()
    assert all(e['status']=='CANDIDATE' for e in tl['events'])
    reg=build_event_registry(chrono_claims=tl['timeline'])
    assert reg
