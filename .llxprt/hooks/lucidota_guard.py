#!/usr/bin/env python3
"""LUCIDOTA LLxprt hook: audit tool calls and block obvious secret/law-file damage.

No secret values are logged; suspicious strings are redacted before appending to
04_RUNTIME/llxprt_hook_audit.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(os.environ.get('LLXPRT_PROJECT_DIR') or '/home/mfspx/LUCIDOTA').resolve()
AUDIT = PROJECT / '04_RUNTIME' / 'llxprt_hook_audit.jsonl'
SENSITIVE_PATHS = [
    Path.home() / '.config' / 'lucidota' / 'secrets.env',
    Path.home() / '.llxprt' / 'provider-keys',
    Path.home() / '.local' / 'share' / 'llxprt-code' / 'secure-store',
]
PROTECTED_PROJECT_FILES = [
    PROJECT / '00_PROJECT_BRAIN' / 'TICKLETRUNK.json',
    PROJECT / '00_PROJECT_BRAIN' / 'TICKLETRUNK.md',
    PROJECT / '00_PROJECT_BRAIN' / 'BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md',
]
SECRET_RE = re.compile(r'(?i)(api[_-]?key|token|secret|password|authorization)\s*[:=]\s*([^\s,;]+)')
KEYISH_RE = re.compile(r'(?i)\b(?:gsk_[A-Za-z0-9_\-]{20,}|co_[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})\b')


def resolve_path(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        p = PROJECT / p
    return p.resolve(strict=False)


def under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base.resolve(strict=False))
        return True
    except Exception:
        return False


def redact(value: Any) -> Any:
    if isinstance(value, str):
        value = SECRET_RE.sub(lambda m: f"{m.group(1)}=<redacted>", value)
        value = KEYISH_RE.sub('<redacted-key>', value)
        return value
    if isinstance(value, dict):
        return {k: ('<redacted>' if re.search(r'(?i)(key|token|secret|password|authorization)', str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def candidate_paths(tool_input: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for key in ('path', 'file_path', 'filename', 'target_file', 'target'):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append(resolve_path(value))
    return paths


def deny(reason: str) -> None:
    print(json.dumps({'decision': 'deny', 'reason': reason}))
    raise SystemExit(2)


def allow() -> None:
    print(json.dumps({'decision': 'allow'}))
    raise SystemExit(0)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        deny(f'LUCIDOTA guard could not parse hook JSON: {exc}')

    tool = str(payload.get('tool_name') or payload.get('tool') or '')
    tool_input = payload.get('tool_input') if isinstance(payload.get('tool_input'), dict) else {}

    record = {
        'ts': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
        'event': payload.get('hook_event_name') or payload.get('event') or 'unknown',
        'tool': tool,
        'input': redact(tool_input),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')

    # Block direct file-tool changes to startup-law files unless operator explicitly works around it.
    for p in candidate_paths(tool_input):
        if any(p == protected.resolve(strict=False) for protected in PROTECTED_PROJECT_FILES):
            deny(f'LUCIDOTA startup law file is protected from direct LLxprt mutation: {p}')
        if any(p == sp.resolve(strict=False) or under(p, sp) for sp in SENSITIVE_PATHS):
            deny(f'Sensitive credential path is protected from LLxprt tool access: {p}')

    # Block shell commands that obviously print/edit the configured secret store.
    command = str(tool_input.get('command') or '')
    if command:
        secret_path_texts = [str(p) for p in SENSITIVE_PATHS]
        reads_or_writes = re.search(r'\b(cat|sed|awk|grep|rg|less|more|head|tail|cp|mv|rm|chmod|chown|tee|>|>>|python|node)\b', command)
        if reads_or_writes and any(t in command or t.replace(str(Path.home()), '~') in command for t in secret_path_texts):
            deny('Shell command touches LUCIDOTA credential paths; blocked to prevent secret exposure/mutation.')
        if re.search(r'\brm\s+-rf\s+(/|~|\$HOME)\b', command):
            deny('Dangerous recursive deletion target blocked by LUCIDOTA guard.')

    allow()


if __name__ == '__main__':
    main()
