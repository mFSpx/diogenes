#!/usr/bin/env python3
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Any
from spine_common import ROOT, sha256_bytes, sha256_json, write_json, rel, now

class ContentStore:
    def __init__(self, root: str|Path|None=None, *, case_id: str|None=None):
        base=Path(root) if root else ROOT/'05_OUTPUTS/content_store'
        self.root=base/(case_id or '_global')
        (self.root/'objects').mkdir(parents=True,exist_ok=True)
        (self.root/'refs').mkdir(parents=True,exist_ok=True)
    def put_bytes(self, data: bytes, *, media_type: str='application/octet-stream', suffix: str='.bin') -> dict[str,Any]:
        digest=sha256_bytes(data); obj=self.root/'objects'/digest[:2]/(digest+suffix); obj.parent.mkdir(parents=True,exist_ok=True)
        if not obj.exists(): obj.write_bytes(data)
        meta={'content_ref':'cas:'+digest,'sha256':digest,'size_bytes':len(data),'media_type':media_type,'path':rel(obj),'stored_at':now()}
        write_json(self.root/'refs'/(digest+'.json'), meta); return meta
    def put_json(self, value: Any, *, media_type: str='application/json') -> dict[str,Any]:
        return self.put_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode(), media_type=media_type, suffix='.json')
    def put_text(self, text: str, *, media_type: str='text/plain') -> dict[str,Any]:
        return self.put_bytes(text.encode(), media_type=media_type, suffix='.txt')
    def get(self, content_ref: str) -> bytes:
        digest=content_ref.split(':',1)[1]
        for p in (self.root/'objects'/digest[:2]).glob(digest+'*'):
            return p.read_bytes()
        raise FileNotFoundError(content_ref)
