from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ArtifactEnvelope:
    artifact_id: uuid.UUID
    sha256: str
    size_bytes: int
    acquisition_location: str
    t_thing: datetime | None
    t_possession: datetime
    t_ingested: datetime
    source_path: str
    media_type_class: str = "binary_unknown"
    domain: str = "unknown"
    sub_domain: str | None = None
    xattr_stamped: bool = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_t_thing(path: Path) -> datetime | None:
    suffix = path.suffix.lower()

    # a. EXIF for images
    if suffix in {".jpg", ".jpeg", ".tiff", ".tif", ".png", ".heic", ".heif"}:
        try:
            from PIL import Image
            img = Image.open(path)
            exif = img._getexif()  # type: ignore[attr-defined]
            if exif and 36867 in exif:
                return datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
        except Exception:
            pass

    # b. PDF metadata
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            meta = reader.metadata
            if meta and meta.creation_date:
                cd = meta.creation_date
                if isinstance(cd, datetime):
                    return cd.replace(tzinfo=timezone.utc) if cd.tzinfo is None else cd
            # fallback: raw /CreationDate string
            raw = getattr(meta, "creation_date_raw", None) or (
                meta.get("/CreationDate") if meta else None
            )
            if raw and isinstance(raw, str) and raw.startswith("D:"):
                return datetime.strptime(raw[2:16], "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )
        except Exception:
            pass

    # c. Filename date regex: YYYYMMDD anywhere in name
    m = re.search(r"(20[12][0-9]{5})", path.name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


def make_envelope(
    path: Path, acquisition_location: str = "Laptop"
) -> ArtifactEnvelope:
    stat = path.stat()
    size_bytes = stat.st_size
    t_ingested = datetime.now(timezone.utc)
    t_possession = datetime.fromtimestamp(stat.st_mtime, timezone.utc)

    if size_bytes > 500 * 1024 * 1024:
        digest = "SKIPPED_LARGE"
    else:
        digest = sha256_file(path)

    artifact_id = uuid.uuid4()
    t_thing = extract_t_thing(path)

    xattr_stamped = False
    path_bytes = os.fsencode(path)
    try:
        os.setxattr(path_bytes, b"user.lucidota.sha256", digest.encode())
        os.setxattr(
            path_bytes, b"user.lucidota.artifact_id", str(artifact_id).encode()
        )
        xattr_stamped = True
    except OSError:
        pass

    return ArtifactEnvelope(
        artifact_id=artifact_id,
        sha256=digest,
        size_bytes=size_bytes,
        acquisition_location=acquisition_location,
        t_thing=t_thing,
        t_possession=t_possession,
        t_ingested=t_ingested,
        source_path=str(path),
    )


