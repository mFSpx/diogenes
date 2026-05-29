#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ImportPolicy:
    max_files: int = 100
    max_bytes: int = 10_000_000
    allowed_extensions: tuple[str, ...] = ()
    quarantine_extensions: tuple[str, ...] = ('.zip','.db','.sqlite','.sqlite3')
    ocr_allowed: bool = False
    model_allowed: bool = False
    export_redaction_level: str = 'local_paths'

    def __post_init__(self):
        if self.max_files <= 0: raise ValueError('max_files must be positive')
        if self.max_bytes <= 0: raise ValueError('max_bytes must be positive')
        if self.export_redaction_level not in {'none','local_paths','strict'}: raise ValueError('invalid export_redaction_level')

    def as_dict(self) -> dict[str, Any]:
        d=asdict(self); d['allowed_extensions']=list(self.allowed_extensions); d['quarantine_extensions']=list(self.quarantine_extensions); return d

    @classmethod
    def from_dict(cls, d: dict[str,Any]) -> 'ImportPolicy':
        return cls(max_files=int(d.get('max_files',100)), max_bytes=int(d.get('max_bytes',10_000_000)), allowed_extensions=tuple(x.lower() for x in d.get('allowed_extensions',[]) or []), quarantine_extensions=tuple(x.lower() for x in d.get('quarantine_extensions',[]) or []), ocr_allowed=bool(d.get('ocr_allowed',False)), model_allowed=bool(d.get('model_allowed',False)), export_redaction_level=d.get('export_redaction_level','local_paths'))

    def decision_for_path(self, path: str|Path) -> dict[str,Any]:
        ext=Path(path).suffix.lower()
        if self.allowed_extensions and ext not in self.allowed_extensions:
            return {'allowed':False,'force_quarantine':True,'reason':'EXTENSION_NOT_ALLOWED'}
        if ext in self.quarantine_extensions:
            return {'allowed':True,'force_quarantine':True,'reason':'POLICY_QUARANTINE_EXTENSION'}
        return {'allowed':True,'force_quarantine':False,'reason':''}
