#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_pipeline import LucidotaPipeline

def test_pipeline_stage_outputs_have_content_refs(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'note.md').write_text('Alice saw Evidence.')
    run=LucidotaPipeline(case_id='case-content', output_root=tmp_path/'out').run_fixture_pipeline(src, max_files=10)
    assert all(s.output.get('content_ref','').startswith('cas:') for s in run.stages)
