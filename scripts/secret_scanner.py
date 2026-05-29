#!/usr/bin/env python3
from __future__ import annotations
import re
from pathlib import Path
SECRET_RE=[re.compile(r'(?i)(api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9_\-]{8,}'), re.compile(r'-----BEGIN (RSA |OPENSSH |DSA |EC )?PRIVATE KEY-----')]

def scan_paths(paths: list[str|Path]) -> dict:
    findings=[]
    for path in paths:
        p=Path(path)
        if not p.exists() or not p.is_file(): continue
        text=p.read_text(errors='ignore')[:200000]
        for i,rx in enumerate(SECRET_RE):
            if rx.search(text): findings.append({'path':str(p),'rule':f'secret_rule_{i}','redacted':True})
    return {'clean':not findings,'findings':findings}
