#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from case_packet_renderer import render_case_packet

def test_case_packet_renders_to_markdown_and_html_from_structured_json(tmp_path):
    packet={'case_id':'case','packet_id':'case:1','receipt_refs':['r1'],'timeline_refs':{'timeline_path':'t'},'claim_refs':{'claim_count':1},'graph_candidate_refs':{'candidate_count':1}}
    out=render_case_packet(packet, md_path=tmp_path/'case.md', html_path=tmp_path/'case.html')
    md=(tmp_path/'case.md').read_text(); html=(tmp_path/'case.html').read_text()
    assert '# Case Packet case' in md
    assert 'Evidence Refs' in md
    assert '<html>' in html
    assert out['template_contract']=='jinja_like_minimal_v1'
    assert Path(out['markdown_path']).exists() and Path(out['html_path']).exists()
