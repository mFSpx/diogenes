#!/usr/bin/env python3
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from spine_common import sha256_json
ISO_PATTERNS=[re.compile(r'(20\d{2})[-_](\d{2})[-_](\d{2})'), re.compile(r'(20\d{6})T?(\d{6})?')]

def _iso_from_match(m):
    g=m.groups()
    if len(g)>=3 and g[1] and len(g[1])==2:
        return f'{g[0]}-{g[1]}-{g[2]}T00:00:00Z'
    raw=g[0]; t=g[1] or '000000'
    return f'{raw[0:4]}-{raw[4:6]}-{raw[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z'

def claims_from_inventory_rows(rows: list[dict[str,Any]], *, source_root: str|Path|None=None) -> list[dict[str,Any]]:
    claims=[]
    for row in rows:
        source_ref={'custody_id':row.get('occurrence_id'),'content_id':row.get('content_id'),'relative_path':row.get('relative_path')}
        # filesystem mtime from fixture/source when available; still a claim, not truth.
        if source_root:
            p=Path(source_root)/row['relative_path']
            if p.exists():
                ts=datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat().replace('+00:00','Z')
                claims.append(_claim(ts,'filesystem_mtime',1000,source_ref,'inventory_stat_mtime'))
        for pat in ISO_PATTERNS:
            m=pat.search(row.get('relative_path',''))
            if m:
                claims.append(_claim(_iso_from_match(m),'filename_date_hint',8000,source_ref,'regex_filename_date'))
                break
    return claims

def _claim(ts, source, confidence, source_ref, method):
    base={'candidate_timestamp':ts,'evidence_source':source,'confidence_bps':confidence,'source_ref':source_ref,'method':method,'conflict':False}
    base['claim_id']='chrono:'+sha256_json(base)[:24]
    return base
