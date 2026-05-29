#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Any
from spine_common import ROOT, rel, sha256_bytes
MODEL_EXT={'.gguf','.bin','.safetensors'}

def inventory_models(paths: list[str|Path]|None=None, *, max_files:int=200) -> list[dict[str,Any]]:
    roots=[Path(p) for p in (paths or ['03_VAULT/models','MODELS','01_REPOS/llama.cpp/models'])]
    out=[]
    for root in roots:
        if not (ROOT/root).exists() and not root.exists(): continue
        base=(ROOT/root) if (ROOT/root).exists() else root
        for p in sorted(base.rglob('*')):
            if len(out)>=max_files: break
            if p.is_file() and p.suffix.lower() in MODEL_EXT:
                # Hash small files only; large model provenance uses size/path until explicit model-intake pass.
                h=sha256_bytes(p.read_bytes()) if p.stat().st_size < 20_000_000 else None
                out.append({'model_id':'model:'+rel(p),'path':rel(p),'size_bytes':p.stat().st_size,'extension':p.suffix.lower(),'sha256':h,'status':'FOUND'})
    return out
