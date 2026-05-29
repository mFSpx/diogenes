#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]

def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')

def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def rel(p: str|Path) -> str:
    try: return str(Path(p).resolve().relative_to(ROOT))
    except Exception: return str(p)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_json(v: Any) -> str:
    return sha256_bytes(json.dumps(v, sort_keys=True, separators=(',',':'), default=str).encode())

def read_json(path: str|Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path: str|Path, data: Any) -> Path:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding='utf-8'); return p

def append_jsonl(path: str|Path, row: dict[str,Any]) -> Path:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f: f.write(json.dumps(row, sort_keys=True, default=str)+'\n')
    return p

def receipt(component: str, payload: dict[str,Any], *, root: str|Path='05_OUTPUTS/test_runs') -> Path:
    p=ROOT/root/f'{component}_{stamp()}.json'
    payload.setdefault('generated_at', now()); payload['receipt_path']=rel(p); write_json(p,payload); print(f'RECEIPT_PATH={rel(p)}'); return p

def source_hash(path: str|Path) -> str:
    return sha256_bytes(Path(path).read_bytes())
