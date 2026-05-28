#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_pipeline import LucidotaPipeline

def test_demo_product_snapshot_normalized_outputs(tmp_path):
    expected=json.loads((ROOT/'tests/snapshots/demo_product_snapshot.json').read_text())
    run=LucidotaPipeline(case_id='demo-snapshot', output_root=tmp_path/'out').run_fixture_pipeline(ROOT/'tests/fixtures/demo_corpus', max_files=10)
    assert [s.stage_name for s in run.stages] == expected['stage_names']
    assert len(run.stages) >= expected['min_stage_count']
    staging=next(s.output for s in run.stages if s.stage_name=='staging')
    graph=next(s.output for s in run.stages if s.stage_name=='graph_candidate')
    assert staging['claim_count'] >= expected['min_claims']
    assert graph['candidate_count'] >= expected['min_graph_candidates']
