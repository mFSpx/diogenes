#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from case_workspace import CaseWorkspace
from lucidota_pipeline import LucidotaPipeline


def test_same_filenames_in_two_cases_are_isolated(tmp_path):
    src1=tmp_path/'drop1'; src2=tmp_path/'drop2'; src1.mkdir(); src2.mkdir()
    (src1/'note.md').write_text('Alice saw Evidence.')
    (src2/'note.md').write_text('Alice saw Different Evidence.')
    ws1=CaseWorkspace.create('case-one', base_dir=tmp_path/'cases')
    ws2=CaseWorkspace.create('case-two', base_dir=tmp_path/'cases')
    assert ws1.root != ws2.root
    r1=LucidotaPipeline(case_id='case-one', output_root=ws1.root/'workspace').run_fixture_pipeline(src1,max_files=10)
    r2=LucidotaPipeline(case_id='case-two', output_root=ws2.root/'workspace').run_fixture_pipeline(src2,max_files=10)
    assert (ws1.root/'case_workspace.json').exists()
    assert (ws2.root/'case_workspace.json').exists()
    assert (ws1.root/'workspace'/'case_packet.json').exists()
    assert (ws2.root/'workspace'/'case_packet.json').exists()
    assert (ws1.root/'workspace'/'case_packet.json').read_text() != (ws2.root/'workspace'/'case_packet.json').read_text()
    assert all(s.output.get('content_ref','').startswith('cas:') for s in r1.stages)
    assert all(s.output.get('content_ref','').startswith('cas:') for s in r2.stages)
