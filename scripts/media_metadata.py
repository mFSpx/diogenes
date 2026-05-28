#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from spine_common import source_hash, rel
MEDIA_EXT={'.mp4','.mov','.mkv','.opus','.mp3','.wav','.m4a'}

def extract_media_metadata(path: str|Path) -> dict:
    p=Path(path)
    return {'source_path':rel(p),'source_sha256':source_hash(p),'kind':'media','size_bytes':p.stat().st_size,'metadata_status':'METADATA_ONLY','text_extractable':False}
