#!/usr/bin/env python3
from __future__ import annotations
import html, json
from pathlib import Path
from typing import Any
from template_contract import render_template

MARKDOWN_TEMPLATE = """# Case Packet {{ case_id }}

Packet: `{{ packet_id }}`

## Evidence Refs
{% for ref in receipt_refs %}- `{{ ref }}`
{% endfor %}
## Timeline
{{ timeline_json }}

## Claims
{{ claims_json }}

## Graph Candidates
{{ graph_candidates_json }}
"""

HTML_TEMPLATE = """<!doctype html><html><body><h1>Case Packet {{ case_id }}</h1><p>Packet: <code>{{ packet_id }}</code></p><h2>Evidence Refs</h2><ul>{% for ref in receipt_refs_html %}<li><code>{{ ref }}</code></li>{% endfor %}</ul><h2>Timeline</h2><pre>{{ timeline_json_html }}</pre><h2>Claims</h2><pre>{{ claims_json_html }}</pre><h2>Graph Candidates</h2><pre>{{ graph_candidates_json_html }}</pre></body></html>"""

def render_case_packet(packet: dict[str,Any], *, md_path: str|Path, html_path: str|Path) -> dict[str,str]:
    timeline_json=json.dumps(packet.get('timeline_refs',{}),indent=2,sort_keys=True)
    claims_json=json.dumps(packet.get('claim_refs',{}),indent=2,sort_keys=True)
    graph_json=json.dumps(packet.get('graph_candidate_refs',{}),indent=2,sort_keys=True)
    ctx={
        'case_id':packet.get('case_id'),
        'packet_id':packet.get('packet_id'),
        'receipt_refs':packet.get('receipt_refs',[]),
        'receipt_refs_html':[html.escape(str(r)) for r in packet.get('receipt_refs',[])],
        'timeline_json':timeline_json,
        'claims_json':claims_json,
        'graph_candidates_json':graph_json,
        'timeline_json_html':html.escape(timeline_json),
        'claims_json_html':html.escape(claims_json),
        'graph_candidates_json_html':html.escape(graph_json),
    }
    md=render_template(MARKDOWN_TEMPLATE,ctx)
    rendered_html=render_template(HTML_TEMPLATE,ctx)
    Path(md_path).parent.mkdir(parents=True,exist_ok=True); Path(html_path).parent.mkdir(parents=True,exist_ok=True)
    Path(md_path).write_text(md,encoding='utf-8')
    Path(html_path).write_text(rendered_html,encoding='utf-8')
    return {'markdown_path':str(md_path),'html_path':str(html_path),'template_contract':'jinja_like_minimal_v1'}
