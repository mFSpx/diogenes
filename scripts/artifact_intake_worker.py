from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import artifact_envelope
import artifact_type_router
import artifact_domain_classifier
from lucidota_cas_journal import append_record

import psycopg

_DSN = "postgresql:///lucidota_storage"

_UPSERT = """
INSERT INTO lucidota_korpus.file_object (
    file_uuid, sha256, size_bytes, mime, file_kind, status,
    suffix, cas_uri, first_seen_path, first_seen_at, last_seen_at,
    acquisition_location, t_thing, t_possession, t_ingested,
    domain, sub_domain, media_type_class, xattr_stamped, detail
) VALUES (
    %(file_uuid)s, %(sha256)s, %(size_bytes)s, %(mime)s, %(file_kind)s, %(status)s,
    %(suffix)s, %(cas_uri)s, %(first_seen_path)s, %(first_seen_at)s, %(last_seen_at)s,
    %(acquisition_location)s, %(t_thing)s, %(t_possession)s, %(t_ingested)s,
    %(domain)s, %(sub_domain)s, %(media_type_class)s, %(xattr_stamped)s, %(detail)s
)
ON CONFLICT (sha256) DO UPDATE SET
    domain               = EXCLUDED.domain,
    sub_domain           = EXCLUDED.sub_domain,
    media_type_class     = EXCLUDED.media_type_class,
    acquisition_location = EXCLUDED.acquisition_location,
    t_thing              = EXCLUDED.t_thing,
    t_possession         = EXCLUDED.t_possession,
    t_ingested           = EXCLUDED.t_ingested,
    xattr_stamped        = EXCLUDED.xattr_stamped,
    last_seen_at         = EXCLUDED.last_seen_at,
    updated_at           = NOW()
RETURNING file_uuid
"""


def _extract_text_snippet(path: Path, media_type_class: str) -> str:
    if media_type_class in ("text_plain", "code", "structured"):
        try:
            return path.read_text(errors="ignore")[:500]
        except OSError:
            return ""
    if media_type_class == "rich_doc":
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
            result = DocumentConverter().convert(str(path))
            text = result.document.export_to_text()
            return text[:500]
        except Exception:
            return ""
    return ""


def process_artifact(
    path: Path,
    acquisition_location: str = "Laptop",
    groq_api_key: str = "",
) -> dict[str, Any]:
    env = artifact_envelope.make_envelope(path, acquisition_location)
    media_type_class: str = artifact_type_router.classify_media_type(path)
    text_snippet = _extract_text_snippet(path, media_type_class)
    domain, sub_domain = artifact_domain_classifier.classify_domain(
        path, text_snippet, groq_api_key
    )

    import mimetypes
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    row: dict[str, Any] = {
        "file_uuid":           env.artifact_id,
        "sha256":              env.sha256,
        "size_bytes":          env.size_bytes,
        "mime":                mime,
        "file_kind":           media_type_class,
        "status":              "indexed",
        "suffix":              path.suffix.lower() or ".bin",
        "cas_uri":             "",
        "first_seen_path":     str(path),
        "first_seen_at":       env.t_possession,
        "last_seen_at":        env.t_ingested,
        "acquisition_location": acquisition_location,
        "t_thing":             env.t_thing,
        "t_possession":        env.t_possession,
        "t_ingested":          env.t_ingested,
        "domain":              domain,
        "sub_domain":          sub_domain,
        "media_type_class":    media_type_class,
        "xattr_stamped":       env.xattr_stamped,
        "detail":              "{}",
    }

    with psycopg.connect(_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(_UPSERT, row)
            conn.commit()

    journal_record = {
        "artifact_id":         str(env.artifact_id),
        "sha256":              env.sha256,
        "source_path":         str(path),
        "acquisition_location": acquisition_location,
        "domain":              domain,
        "sub_domain":          sub_domain,
        "media_type_class":    media_type_class,
        "t_thing":             str(env.t_thing),
        "t_possession":        str(env.t_possession),
        "t_ingested":          str(env.t_ingested),
        "xattr_stamped":       env.xattr_stamped,
    }
    append_record(journal_record)

    return {
        "artifact_id":      str(env.artifact_id),
        "sha256":           env.sha256,
        "domain":           domain,
        "sub_domain":       sub_domain,
        "media_type_class": media_type_class,
        "t_thing":          str(env.t_thing),
        "t_possession":     str(env.t_possession),
        "t_ingested":       str(env.t_ingested),
        "xattr_stamped":    env.xattr_stamped,
        "source_path":      str(path),
    }


if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        print("usage: artifact_intake_worker.py <path> [acquisition_location]")
        sys.exit(1)
    target = Path(sys.argv[1])
    loc = sys.argv[2] if len(sys.argv) > 2 else "Laptop"
    receipt = process_artifact(target, loc)
    print(json.dumps(receipt, indent=2))
