#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Callable, Any

def snapshot(root: str|Path) -> set[str]:
    r=Path(root)
    if not r.exists(): return set()
    return {str(p.resolve()) for p in r.rglob('*') if p.is_file()}

def enforce_write_scope(*, repo_root: str|Path, allowed_roots: list[str|Path], fn: Callable[[],Any]) -> dict:
    before=snapshot(repo_root)
    result=None; error=None
    try: result=fn()
    except Exception as e: error=repr(e)
    after=snapshot(repo_root)
    new_files=after-before
    allowed=[Path(a).resolve() for a in allowed_roots]
    violations=[p for p in new_files if not any(Path(p).is_relative_to(a) for a in allowed)]
    if violations: raise RuntimeError(f'write_scope_violation:{violations[:5]}')
    if error: raise RuntimeError(error)
    return {'status':'PASSED','new_files':sorted(new_files),'result':result}
