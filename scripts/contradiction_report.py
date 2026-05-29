#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import sha256_json, now
NEGATORS=['did not ', 'does not ', 'is not ', 'are not ', 'not ','never ','false','no ']

def _norm(text): return ' '.join(text.lower().replace('.','').split())
def _base(text):
    t=_norm(text)
    for n in NEGATORS: t=t.replace(n,'')
    words=[]
    for w in t.split():
        if w.endswith('ed') and len(w)>4: w=w[:-2]
        words.append(w)
    return ' '.join(words).strip()
def _neg(text): return any(n in _norm(text) for n in NEGATORS)

def detect_contradictions(
    claims: list[dict[str,Any]],
    *,
    max_reports: int | None = 1000,
    max_pairs_per_base: int = 25,
) -> list[dict[str,Any]]:
    """Detect exact negation contradictions without an O(n^2) full scan.

    The old implementation compared every claim pair. That is fine for tiny
    packets but explodes on real case files. Grouping by normalized base text
    preserves the exact-matching semantics while keeping full-case passes
    bounded and deterministic.
    """
    out=[]
    grouped: dict[str, dict[bool, list[dict[str,Any]]]] = {}
    for claim in claims:
        grouped.setdefault(_base(claim['claim_text']), {False: [], True: []})[_neg(claim['claim_text'])].append(claim)
    for base_key in sorted(grouped):
        groups = grouped[base_key]
        if not groups[False] or not groups[True]:
            continue
        emitted_for_base = 0
        for a in groups[False]:
            for b in groups[True]:
                if max_reports is not None and len(out) >= max_reports:
                    return out
                if emitted_for_base >= max_pairs_per_base:
                    break
                emitted_for_base += 1
                rec={'contradiction_id':'contra:'+sha256_json({'a':a['claim_id'],'b':b['claim_id']})[:24], 'severity':'HIGH', 'claim_ids':[a['claim_id'],b['claim_id']], 'entity_refs':list(set(a.get('entity_candidates',[])+b.get('entity_candidates',[]))), 'event_refs':[], 'source_refs':[a.get('source_ref'),b.get('source_ref')], 'resolution_status':'OPEN', 'created_at':now()}
                out.append(rec)
            if emitted_for_base >= max_pairs_per_base:
                break
    return out

def set_resolution(record: dict[str,Any], status: str, *, reason: str) -> dict[str,Any]:
    if status not in {'RESOLVED','DISMISSED'}: raise ValueError('invalid resolution status')
    r=dict(record); r['resolution_status']=status; r['resolution_reason']=reason; r['resolved_at']=now(); return r
