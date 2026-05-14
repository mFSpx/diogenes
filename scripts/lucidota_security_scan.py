#!/usr/bin/env python3
"""Lightweight repo security tripwire for LUCIDOTA tracked docs/scripts."""
from __future__ import annotations
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCAN_DIRS=['00_PROJECT_BRAIN','02_RECORDS_OFFICE','06_SCHEMA','scripts']
PATTERNS={
 'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
 'env_assignment_secret': re.compile(r'(?i)\b(api[_-]?key|secret|token|password)\s*=\s*[^\s#]{12,}'),
 'google_oauth_file': re.compile('(?i)(' + 'auth' + r'\.json|' + 'credentials' + r'\.json|' + 'oauth' + r'_?tokens?\.json)'),
 'drive_secret_archive': re.compile('(?i)(' + 'CURRENT' + '_' + 'SECRETS|' + 'secrets' + '_' + 'LOCAL' + '_' + 'PRIVATE|' + 'sandbox' + '_' + 'users|' + 'cap' + '_' + 'sid)'),
}
ALLOWLIST={
 ('02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md','drive_secret_archive'),
 ('02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md','google_oauth_file'),
 ('02_RECORDS_OFFICE/CYBERSECURITY_RISK_REGISTER.md','google_oauth_file'),
 ('02_RECORDS_OFFICE/CYBERSECURITY_RISK_REGISTER.md','drive_secret_archive'),
}
findings=[]
for d in SCAN_DIRS:
    for path in (ROOT/d).rglob('*'):
        if not path.is_file(): continue
        rel=path.relative_to(ROOT).as_posix()
        if rel == 'scripts/lucidota_security_scan.py' or '__pycache__' in rel or path.suffix in {'.pyc', '.pyo'}:
            continue
        text=path.read_text(errors='ignore')
        for name,pat in PATTERNS.items():
            for m in pat.finditer(text):
                if (rel,name) in ALLOWLIST: continue
                line=text.count('\n',0,m.start())+1
                findings.append({'file':rel,'line':line,'rule':name,'snippet':m.group(0)[:120]})
print(json.dumps({'ok': not findings, 'findings': findings[:100], 'count': len(findings)}, indent=2, sort_keys=True))
raise SystemExit(1 if findings else 0)
