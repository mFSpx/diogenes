#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_pipeline import LucidotaPipeline

def test_pipeline_api_runs_named_product_stages(tmp_path):
    src=tmp_path/'drop'; src.mkdir(); (src/'2026-05-17-note.md').write_text('Alice saw Evidence. Maybe Bob disagrees.'); (src/'image.png').write_bytes(b'PNG')
    run=LucidotaPipeline(case_id='fixture-case', output_root=tmp_path/'out').run_fixture_pipeline(src, max_files=10)
    assert run.failed_stage is None
    names=[s.stage_name for s in run.stages]
    assert names == ['intake','parse','timeline','staging','graph_candidate','case_packet']
    assert all(s.status=='PASSED' for s in run.stages)
    assert (tmp_path/'out'/'pipeline_run.json').exists()
    assert (tmp_path/'out'/'case_packet.json').exists()
    packet=(tmp_path/'out'/'case_packet.json').read_text()
    assert 'claim_refs' in packet and 'graph_candidate_refs' in packet
