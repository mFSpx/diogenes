#!/usr/bin/env python3
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from spine_common import ROOT, now, rel, write_json, sha256_json
from import_policy import ImportPolicy

CASE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
STANDARD_CASE_RE = re.compile(r'^KE26-[0-9]{5}$')


def validate_case_id(case_id: str) -> str:
    if not CASE_ID_RE.match(case_id):
        raise ValueError('invalid case_id')
    return case_id


def _case_number_registry(base: Path) -> Path:
    return base / '_case_number_registry.json'


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'schema': 'lucidota.case_number_registry.v1', 'prefix': 'KE26', 'aliases': {}}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        data = {'schema': 'lucidota.case_number_registry.v1', 'prefix': 'KE26', 'aliases': {}}
    data.setdefault('schema', 'lucidota.case_number_registry.v1')
    data.setdefault('prefix', 'KE26')
    data.setdefault('aliases', {})
    return data


def _known_case_numbers(base: Path, registry: dict[str, Any]) -> set[str]:
    numbers = {str(v) for v in registry.get('aliases', {}).values() if STANDARD_CASE_RE.match(str(v))}
    if base.exists():
        for child in base.iterdir():
            if child.is_dir() and STANDARD_CASE_RE.match(child.name):
                numbers.add(child.name)
            meta = child / 'case_workspace.json'
            if meta.exists():
                try:
                    n = json.loads(meta.read_text(encoding='utf-8')).get('case_number', '')
                    if STANDARD_CASE_RE.match(str(n)):
                        numbers.add(str(n))
                except json.JSONDecodeError:
                    pass
    return numbers


def standard_case_number(case_id: str, base: Path) -> str:
    """Return a stable KE26-##### case number without renaming legacy aliases."""
    case_id = validate_case_id(case_id)
    if STANDARD_CASE_RE.match(case_id):
        return case_id
    base.mkdir(parents=True, exist_ok=True)
    registry_path = _case_number_registry(base)
    registry = _load_registry(registry_path)
    aliases = registry.setdefault('aliases', {})
    if case_id in aliases and STANDARD_CASE_RE.match(str(aliases[case_id])):
        return str(aliases[case_id])
    used = _known_case_numbers(base, registry)
    n = 1
    while f'KE26-{n:05d}' in used:
        n += 1
    number = f'KE26-{n:05d}'
    aliases[case_id] = number
    registry['updated_at'] = now()
    write_json(registry_path, registry)
    return number


@dataclass(frozen=True)
class CaseWorkspace:
    case_id: str
    case_number: str
    root: Path
    read_only: bool = False

    @classmethod
    def create(cls, case_id: str, *, base_dir: str | Path | None = None, read_only: bool = False) -> 'CaseWorkspace':
        case_id = validate_case_id(case_id)
        base = Path(base_dir) if base_dir else ROOT / '05_OUTPUTS/cases'
        case_number = standard_case_number(case_id, base)
        root = base / case_id
        root.mkdir(parents=True, exist_ok=True)
        for sub in ['content_store', 'receipts', 'runs', 'exports', 'workspace']:
            (root / sub).mkdir(parents=True, exist_ok=True)
        case_hash = sha256_json({'case_id': case_id, 'case_number': case_number, 'root': rel(root)})
        meta = {
            'schema':'lucidota.case_workspace.v1',
            'case_id':case_id,
            'case_number':case_number,
            'case_file_number':case_number,
            'case_hash':'sha256:'+case_hash,
            'root':rel(root),
            'read_only':read_only,
            'created_or_seen_at':now(),
            'organization':['content_store','receipts','runs','exports','workspace'],
        }
        write_json(root / 'case_workspace.json', meta)
        return cls(case_id=case_id, case_number=case_number, root=root, read_only=read_only)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def assert_writable(self) -> None:
        if self.read_only:
            raise PermissionError(f'case workspace {self.case_id} is read-only')

    def scoped_ref(self, payload: Any) -> str:
        return f'case:{self.case_number}:{sha256_json(payload)[:24]}'

    def write_import_policy(self, policy: ImportPolicy) -> Path:
        self.assert_writable()
        return write_json(self.root / 'import_policy.json', policy.as_dict())

    def load_import_policy(self) -> ImportPolicy:
        p=self.root / 'import_policy.json'
        if not p.exists():
            return ImportPolicy()
        import json
        return ImportPolicy.from_dict(json.loads(p.read_text(encoding='utf-8')))
