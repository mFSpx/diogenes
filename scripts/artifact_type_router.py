from __future__ import annotations

from pathlib import Path

TEXT_PLAIN_EXTS: frozenset[str] = frozenset({'.txt', '.md', '.rst', '.log', '.text', '.org', '.wiki'})
RICH_DOC_EXTS: frozenset[str] = frozenset({'.pdf', '.docx', '.doc', '.odt', '.rtf', '.pptx', '.ppt', '.xlsx', '.xls', '.odp', '.ods', '.epub'})
CODE_EXTS: frozenset[str] = frozenset({'.py', '.rs', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.c', '.cpp', '.h', '.hpp',
               '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.sql', '.sh', '.bash', '.zsh',
               '.ps1', '.cmd', '.bat', '.lua', '.vim', '.el', '.clj', '.ex', '.exs', '.hs', '.ml'})
STRUCTURED_EXTS: frozenset[str] = frozenset({'.json', '.jsonl', '.ndjson', '.csv', '.yaml', '.yml', '.toml', '.xml', '.ini',
                     '.cfg', '.conf', '.env', '.tsv'})
IMAGE_EXTS: frozenset[str] = frozenset({'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.tiff', '.tif', '.bmp', '.heic',
                '.heif', '.ico', '.raw', '.cr2', '.nef', '.arw', '.dng'})
AUDIO_EXTS: frozenset[str] = frozenset({'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.opus', '.wma', '.aiff', '.aif', '.ape'})
VIDEO_EXTS: frozenset[str] = frozenset({'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v', '.wmv', '.flv', '.3gp', '.ts', '.mts', '.m2ts'})
ARCHIVE_EXTS: frozenset[str] = frozenset({'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.tgz', '.tbz2', '.zst'})
ASSET_EXTS: frozenset[str] = frozenset({'.psd', '.ai', '.xcf', '.sketch', '.fig', '.blend', '.fbx', '.glb', '.gltf',
                '.indd', '.afdesign', '.afpub', '.afphoto', '.eps', '.ttf', '.otf', '.woff', '.woff2'})

_EXT_MAP: dict[frozenset[str], str] = {
    TEXT_PLAIN_EXTS: 'text_plain',
    RICH_DOC_EXTS: 'rich_doc',
    CODE_EXTS: 'code',
    STRUCTURED_EXTS: 'structured',
    IMAGE_EXTS: 'image',
    AUDIO_EXTS: 'audio',
    VIDEO_EXTS: 'video',
    ARCHIVE_EXTS: 'archive',
    ASSET_EXTS: 'asset',
}

_MIME_PREFIX_MAP: dict[str, str] = {
    'image/': 'image',
    'audio/': 'audio',
    'video/': 'video',
    'text/': 'text_plain',
}

_MIME_EXACT_MAP: dict[str, str] = {
    'application/pdf': 'rich_doc',
    'application/zip': 'archive',
    'application/x-tar': 'archive',
    'application/gzip': 'archive',
    'application/x-bzip2': 'archive',
    'application/x-xz': 'archive',
    'application/x-7z-compressed': 'archive',
    'application/x-rar-compressed': 'archive',
    'application/vnd.rar': 'archive',
    'application/json': 'structured',
    'application/xml': 'structured',
    'application/x-sh': 'code',
}


def classify_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext:
        for ext_set, label in _EXT_MAP.items():
            if ext in ext_set:
                return label

    try:
        import magic
        mime: str = magic.from_file(str(path), mime=True)
    except Exception:
        return 'binary_unknown'

    if exact := _MIME_EXACT_MAP.get(mime):
        return exact
    for prefix, label in _MIME_PREFIX_MAP.items():
        if mime.startswith(prefix):
            return label

    return 'binary_unknown'


if __name__ == '__main__':
    print("BUILT: artifact_type_router.py")
