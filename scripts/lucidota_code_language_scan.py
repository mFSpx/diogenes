#!/usr/bin/env python3
"""Plain-language gate for executable/schema paths.

Purpose: keep code, schemas, and harness text technical and boring. Project lore
and immutable ontology records are out of scope for this scanner.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCAN_DIRS=[Path('scripts'), Path('06_SCHEMA'), Path('ALGOS')]
SCAN_FILES=[Path('check_diogenes.sh')]
SKIP_NAMES={'lucidota_code_language_scan.py'}
PATTERNS={
    'hype_language': re.compile(r'(?i)\b(godly|sick as|vibe|vibes|slop|scream|brutal|blood|weaponized|kill-shot|maniac|frantic)\b'),
    'insult_language': re.compile(r'(?i)\b(bitch|cunt|slut|whore|retard|fat chick|ass burger)\b'),
    'misleading_security_language': re.compile(r'(?i)\b(stealth|aggressive mode|attack surface|target dossier)\b'),
}
ALLOWLIST={
    ('scripts/lucidota_security_scan.py','hype_language'),
    ('ALGOS/krampus_stickers.py','hype_language'),
    ('ALGOS/krampus_stickers.py','insult_language'),
}
findings=[]
paths=[]
for d in SCAN_DIRS:
    root=ROOT/d
    if root.exists():
        paths.extend(p for p in root.rglob('*') if p.is_file())
paths.extend(ROOT/p for p in SCAN_FILES if (ROOT/p).exists())
for path in sorted(paths):
    rel=path.relative_to(ROOT).as_posix()
    if path.name in SKIP_NAMES or '__pycache__' in rel or path.suffix in {'.pyc','.pyo'}:
        continue
    text=path.read_text(errors='ignore')
    for name,pat in PATTERNS.items():
        if (rel,name) in ALLOWLIST:
            continue
        for m in pat.finditer(text):
            findings.append({'file':rel,'line':text.count('\n',0,m.start())+1,'rule':name,'snippet':m.group(0)[:120]})
print(json.dumps({'ok':not findings,'count':len(findings),'findings':findings[:100]}, indent=2, sort_keys=True))
raise SystemExit(1 if findings else 0)
