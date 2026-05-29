#!/usr/bin/env python3
"""Lightweight repo security tripwire for LUCIDOTA.

This scanner must be boring and non-leaky: report locations and masked matches
only. Never echo live-looking credential material back to stdout.
"""
from __future__ import annotations
import argparse
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCAN_DIRS=['00_PROJECT_BRAIN','02_RECORDS_OFFICE','03_VAULT','04_RUNTIME','05_OUTPUTS','06_SCHEMA','scripts','ALGOS','BOOKS']
SKIP_PARTS={'.git','.venv','__pycache__','models','cas','.tmp','plugins','vendor_imports','.sandbox-bin'}
SKIP_SUFFIXES={'.pyc','.pyo','.exe','.dll','.so','.gguf','.safetensors','.pt','.pth','.ckpt','.onnx','.sqlite','.sqlite3','.db','.pkl','.zip','.png','.jpg','.jpeg','.webp','.pdf'}
PATTERNS={
 'private_key': re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
 'anthropic_token': re.compile(r'sk-ant-[A-Za-z0-9_-]{8,}-[A-Za-z0-9_-]{20,}'),
 'openai_key': re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{24,}'),
 'google_api_key': re.compile(r'AIza[0-9A-Za-z_-]{30,45}'),
 'github_token': re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
 'huggingface_token': re.compile(r'hf_[A-Za-z0-9]{20,}'),
 'env_assignment_secret': re.compile(r'(?i)\b(api[_-]?key|client[_-]?secret|secret|token|password|app[_-]?password)\s*[:=]\s*[^\s#,"\'}]{12,}'),
 'google_oauth_file': re.compile('(?i)(' + 'auth' + r'\.json|' + 'credentials' + r'\.json|' + 'oauth' + r'_?tokens?\.json)'),
 'drive_secret_archive': re.compile('(?i)(' + 'CURRENT' + '_' + 'SECRETS|' + 'secrets' + '_' + 'LOCAL' + '_' + 'PRIVATE|' + 'sandbox' + '_' + 'users|' + 'cap' + '_' + 'sid)'),
}
PLACEHOLDERS=('example','dummy','placeholder','your_','<api','<token','not-needed','no-key-required','test-secret','sk-this-is-the-secret-key','hf_xxxxx','sk-question')
ALLOWLIST={
 ('02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md','drive_secret_archive'),
 ('02_RECORDS_OFFICE/DRIVE_MAP_STATUS.md','google_oauth_file'),
 ('02_RECORDS_OFFICE/CYBERSECURITY_RISK_REGISTER.md','google_oauth_file'),
 ('02_RECORDS_OFFICE/CYBERSECURITY_RISK_REGISTER.md','drive_secret_archive'),
}

def skipped(path: Path) -> bool:
    rel_parts=set(path.relative_to(ROOT).parts)
    return bool(rel_parts & SKIP_PARTS) or path.suffix.lower() in SKIP_SUFFIXES

def masked(value: str) -> str:
    compact=re.sub(r'\s+', ' ', value.strip())
    if len(compact) <= 12:
        return '[masked]'
    return f"{compact[:4]}…{compact[-4:]} len={len(compact)}"

def placeholder(value: str) -> bool:
    low=value.lower()
    return any(token in low for token in PLACEHOLDERS)

def scan_text(rel: str, text: str, findings: list[dict]) -> None:
    for name,pat in PATTERNS.items():
        for m in pat.finditer(text):
            if (rel,name) in ALLOWLIST:
                continue
            snippet=m.group(0)[:500]
            if placeholder(snippet):
                continue
            line=text.count('\n',0,m.start())+1
            findings.append({'file':rel,'line':line,'rule':name,'snippet_mask':masked(snippet)})

def scan_repo(*, max_findings: int | None = None, scan_dirs: list[str] | None = None) -> list[dict]:
    findings: list[dict] = []
    for d in scan_dirs or SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if not path.is_file():
                continue
            rel=path.relative_to(ROOT).as_posix()
            if rel == 'scripts/lucidota_security_scan.py' or rel == 'scripts/lucidota_secret_quarantine.py' or skipped(path):
                continue
            try:
                with path.open('r', encoding='utf-8', errors='ignore') as fh:
                    for lineno, line in enumerate(fh, 1):
                        if '"encrypted_content"' in line and '"aggregated_output"' not in line and '"output"' not in line:
                            continue
                        for name,pat in PATTERNS.items():
                            for m in pat.finditer(line):
                                if (rel,name) in ALLOWLIST:
                                    continue
                                snippet=m.group(0)[:500]
                                if placeholder(snippet):
                                    continue
                                findings.append({'file':rel,'line':lineno,'rule':name,'snippet_mask':masked(snippet)})
                                if max_findings and len(findings) >= max_findings:
                                    return findings
            except OSError as exc:
                findings.append({'file':rel,'line':0,'rule':'scan_error','snippet_mask':str(exc)[:120]})
                if max_findings and len(findings) >= max_findings:
                    return findings
    return findings


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-findings', type=int, default=0)
    args=ap.parse_args()
    findings=scan_repo(max_findings=args.max_findings or None)
    print(json.dumps({'ok': not findings, 'findings': findings[:100], 'count': len(findings)}, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == '__main__':
    raise SystemExit(main())
