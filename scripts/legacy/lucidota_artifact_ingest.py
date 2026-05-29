#!/usr/bin/env python3
"""DIOGENES investigative case + artifact workflow.

Deterministic local software, not agent sprawl:
- Create proper case folders/files.
- Hash artifact bytes and lock them into CAS.
- Extract metadata/EXIF/OCR/document/video text.
- Normalize sidecars as JSON/YAML/JSONL.
- Extract entities and write GO graph anchors/edges.
- Run local pivot searches from entities.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html.parser
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import tarfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import psycopg

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
OPERATOR_UUID = os.environ.get("LUCIDOTA_OPERATOR_UUID", "00000000-0000-4000-8000-000000000414")
GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
VAULT_SCHEMA = ROOT / "06_SCHEMA" / "005_cas_manifest.sql"
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
INVESTIGATION_SCHEMA = ROOT / "06_SCHEMA" / "018_investigation_artifact.sql"
CAS_ROOT = ROOT / "03_VAULT" / "cas"
CASE_ROOT = ROOT / "03_VAULT" / "cases"
OUTPUT_ROOT = ROOT / "05_OUTPUTS" / "investigation_artifacts"
FRAME_ROOT = ROOT / "04_RUNTIME" / "artifact_frames"
MAX_TEXT_CHARS = 2_000_000
GRAPH_APPROVAL_MODE = (
    "approved"
    if os.environ.get("LUCIDOTA_GRAPH_APPROVAL_MODE", "").strip().lower() == "approved"
    and os.environ.get("LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL", "").strip().lower() in {"1", "true", "yes", "on"}
    else "staged"
)
ACTOR_ENTITY_KINDS = {"name", "phone", "email", "alias", "address", "organization", "domain", "ip", "identifier"}
NON_ACTOR_ENTITY_KINDS = {"date", "money", "hash", "other", "url"}
NOISE_DOMAINS = {
    "static.xx.fbcdn.net",
    "scontent.xx.fbcdn.net",
    "fbcdn.net",
    "googleapis.com",
    "gstatic.com",
    "cloudfront.net",
    "githubusercontent.com",
    "gravatar.com",
}
NOISE_DOMAIN_PREFIXES = ("static.", "cdn.", "assets.", "asset.", "img.", "images.", "media.", "fonts.")
NOISE_NAME_WORDS = {
    "chunk", "comment", "comments", "reaction", "reactions", "like", "likes", "viewed", "visited",
    "activity", "information", "browser", "cookie", "cookies", "marketplace", "search", "history",
    "account", "login", "security", "profile", "photo", "photos", "video", "videos", "message",
}
ADDRESS_FALSE_SECOND_WORDS = {"min", "mins", "minute", "minutes", "hr", "hrs", "hour", "hours", "sec", "second", "seconds", "all"}
IDENTIFIER_ACTOR_PATTERNS = [
    re.compile(r"\b\d{5,12}\s+(?:BC|B\.C\.|ALBERTA|ONTARIO|CANADA)\s+(?:LTD|INC|CORP|LIMITED|CORPORATION)\b", re.I),
    re.compile(r"\b(?:BC|B\.C\.)\s*\d{5,12}\b", re.I),
]

ENTITY_TO_GO = {
    "name": "ENTITY",
    "alias": "ENTITY",
    "organization": "ENTITY",
    "location": "ENTITY",
    "address": "ENTITY",
    "phone": "ATOMIC_ID",
    "ip": "ATOMIC_ID",
    "email": "ATOMIC_ID",
    "url": "ATOMIC_ID",
    "domain": "ATOMIC_ID",
    "identifier": "ATOMIC_ID",
    "hash": "ATOMIC_ID",
    "date": "TIME",
    "money": "ATTRIBUTE",
    "other": "ENTITY",
}

CASE_DIRS = [
    "00_CASEFILE",
    "01_EVIDENCE_LOCKED",
    "02_NORMALIZED",
    "03_ENTITIES",
    "04_PIVOTS",
    "05_REPORTS",
    "06_EXPORTS",
    "99_AUDIT",
]

FORMAL_CASE_RE = re.compile(r"^KE26-[0-9]{5}$")
LEGACY_ONE_OFF_CASE_RE = re.compile(r"^CASE-[0-9]{8}-[A-Z0-9][A-Z0-9_-]{1,80}$")

COMMON_NAME_FALSE_POSITIVES = {
    "case", "evidence", "artifact", "subject", "report", "analysis", "phone", "email", "date",
    "address", "street", "avenue", "road", "drive", "court", "lane", "file", "page", "figure",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "pdf", "image", "video", "document", "hello", "lorem", "ipsum",
}


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def set_graph_approval_mode(mode: str | None) -> None:
    """Set GO write mode for this process.

    Default is staged. Direct approved writes are intentionally gated because
    graph approval is supposed to be a human/governance promotion step, not a
    side effect of byte ingestion.
    """
    global GRAPH_APPROVAL_MODE
    if not mode:
        return
    mode = mode.strip().lower()
    if mode not in {"staged", "approved"}:
        raise SystemExit(f"invalid graph approval mode: {mode}")
    if mode == "approved" and os.environ.get("LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise SystemExit("direct GO graph approval requires LUCIDOTA_ALLOW_DIRECT_GRAPH_APPROVAL=1")
    GRAPH_APPROVAL_MODE = mode


def graph_direct_approved() -> bool:
    return GRAPH_APPROVAL_MODE == "approved"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run_tool(cmd: list[str], *, timeout: int = 60, text: bool = True) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, timeout=timeout, check=False)
        out = proc.stdout if text else proc.stdout.decode("utf-8", errors="ignore")
        err = proc.stderr if text else proc.stderr.decode("utf-8", errors="ignore")
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": out, "stderr": err, "cmd": cmd}
    except FileNotFoundError:
        return {"ok": False, "returncode": 127, "stdout": "", "stderr": f"missing tool: {cmd[0]}", "cmd": cmd}
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if not isinstance(out, str):
            out = out.decode("utf-8", errors="ignore")
        if not isinstance(err, str):
            err = err.decode("utf-8", errors="ignore")
        return {"ok": False, "returncode": 124, "stdout": out, "stderr": f"timeout: {err}", "cmd": cmd}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def slugify(text: str, *, fallback: str = "CASE") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", text.upper()).strip("-")
    text = re.sub(r"-+", "-", text)
    return (text or fallback)[:80]


def normalize_case_key(raw: str, title: str = "") -> str:
    raw = (raw or "").strip()
    upper = raw.upper()
    if FORMAL_CASE_RE.match(upper) or LEGACY_ONE_OFF_CASE_RE.match(upper):
        return upper
    source = raw or title or "UNTITLED"
    return f"CASE-{dt.datetime.now().strftime('%Y%m%d')}-{slugify(source, fallback='UNTITLED')}"


def case_folder(case_key: str) -> Path:
    return CASE_ROOT / case_key


def ensure_case_shape(case_key: str) -> dict[str, str]:
    base = case_folder(case_key)
    for sub in CASE_DIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return {
        "folder": rel(base),
        "casefile": rel(base / "00_CASEFILE" / f"{case_key}.case.yaml"),
        "evidence_locked": rel(base / "01_EVIDENCE_LOCKED"),
        "normalized": rel(base / "02_NORMALIZED"),
        "entities": rel(base / "03_ENTITIES"),
        "pivots": rel(base / "04_PIVOTS"),
        "reports": rel(base / "05_REPORTS"),
        "exports": rel(base / "06_EXPORTS"),
        "audit": rel(base / "99_AUDIT"),
    }


def classify_kind(path: Path, mime: str) -> str:
    low = mime.lower()
    suffix = path.suffix.lower()
    if low.startswith("image/"):
        return "image"
    if low.startswith("video/"):
        return "video"
    if low.startswith("audio/"):
        return "audio"
    if low.startswith("text/") or suffix in {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".log"}:
        return "text"
    if low in {"application/pdf", "application/rtf"} or suffix in {".pdf", ".docx", ".odt", ".rtf", ".html", ".htm"}:
        return "document"
    if suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"}:
        return "archive"
    if low == "application/octet-stream":
        return "binary"
    return "document" if "officedocument" in low or "opendocument" in low else "unknown"


def detect_mime(path: Path) -> tuple[str, str]:
    out = run_tool(["file", "--mime-type", "-b", str(path)], timeout=10)
    mime = out["stdout"].strip() if out["ok"] else ""
    if not mime:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    desc = run_tool(["file", "-b", str(path)], timeout=10)
    return mime, desc["stdout"].strip() if desc["ok"] else ""


def exiftool_json(path: Path) -> dict[str, Any]:
    out = run_tool(["exiftool", "-json", "-n", str(path)], timeout=30)
    if not out["ok"]:
        return {"engine": "exiftool", "status": "failed", "error": out["stderr"][-500:]}
    try:
        data = json.loads(out["stdout"])
        return data[0] if isinstance(data, list) and data else {"engine": "exiftool", "status": "empty"}
    except Exception as exc:
        return {"engine": "exiftool", "status": "parse_failed", "error": str(exc)}


def parse_exif_datetime(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    patterns = [
        "%Y:%m:%d %H:%M:%S%z",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    cleaned = raw.replace("Z", "+0000")
    cleaned = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", cleaned)
    for pat in patterns:
        try:
            parsed = dt.datetime.strptime(cleaned[:26], pat)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except (TypeError, ValueError):
            continue
    return None


def choose_evidence_date(path: Path, explicit: str, exif: dict[str, Any]) -> tuple[str | None, str]:
    if explicit:
        parsed = parse_exif_datetime(explicit)
        if parsed:
            return parsed.isoformat().replace("+00:00", "Z"), "operator_explicit"
        return explicit, "operator_explicit_unparsed"
    keys = [
        "DateTimeOriginal", "CreateDate", "CreationDate", "MediaCreateDate", "TrackCreateDate",
        "ModifyDate", "FileModifyDate", "FileInodeChangeDate",
    ]
    for key in keys:
        parsed = parse_exif_datetime(exif.get(key))
        if parsed:
            return parsed.isoformat().replace("+00:00", "Z"), f"exif:{key}"
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return mtime.isoformat().replace("+00:00", "Z"), "filesystem_mtime_fallback"


def title_from_metadata(path: Path, explicit: str, exif: dict[str, Any]) -> str:
    if explicit.strip():
        return explicit.strip()[:240]
    for key in ("Title", "ObjectName", "Headline", "FileName", "DocumentName"):
        val = str(exif.get(key) or "").strip()
        if val:
            return val[:240]
    return path.stem[:240]


class TextHTMLParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.parts: list[str] = []
    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in {"script", "style", "template", "noscript"}:
            self.skip += 1
    def handle_endtag(self, tag: str):
        if tag.lower() in {"script", "style", "template", "noscript"} and self.skip:
            self.skip -= 1
    def handle_data(self, data: str):
        if not self.skip:
            self.parts.append(data)


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def read_text_bytes(path: Path, limit: int = MAX_TEXT_CHARS) -> str:
    data = path.read_bytes()[:limit]
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc, errors="replace")
        except (LookupError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


def strip_xml_text(xml: str) -> str:
    text = re.sub(r"<[^>]+>", " ", xml)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return normalize_ws(text)


def extract_document_text(path: Path, mime: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if mime == "application/pdf" or suffix == ".pdf":
        out = run_tool(["pdftotext", "-layout", str(path), "-"], timeout=120)
        return {"engine": "pdftotext", "status": "ok" if out["ok"] else "failed", "text": out["stdout"][:MAX_TEXT_CHARS], "stderr": out["stderr"][-500:]}
    if suffix in {".txt", ".md", ".log", ".json", ".jsonl", ".yaml", ".yml"} or mime.startswith("text/"):
        return {"engine": "direct_text", "status": "ok", "text": read_text_bytes(path)}
    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        text = read_text_bytes(path)
        try:
            sample = "\n".join(", ".join(row) for row in list(csv.reader(text.splitlines(), delimiter=delim))[:5000])
            return {"engine": "csv", "status": "ok", "text": sample[:MAX_TEXT_CHARS]}
        except (csv.Error, UnicodeError):
            return {"engine": "direct_text", "status": "ok", "text": text}
    if suffix in {".html", ".htm"}:
        parser = TextHTMLParser()
        parser.feed(read_text_bytes(path))
        return {"engine": "html.parser", "status": "ok", "text": normalize_ws(" ".join(parser.parts))[:MAX_TEXT_CHARS]}
    if suffix == ".docx" and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            chunks = []
            for name in ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml"):
                if name in zf.namelist():
                    chunks.append(zf.read(name).decode("utf-8", errors="ignore"))
            return {"engine": "docx-zip-xml", "status": "ok", "text": strip_xml_text("\n".join(chunks))[:MAX_TEXT_CHARS]}
    if suffix == ".odt" and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            if "content.xml" in zf.namelist():
                return {"engine": "odt-zip-xml", "status": "ok", "text": strip_xml_text(zf.read("content.xml").decode("utf-8", errors="ignore"))[:MAX_TEXT_CHARS]}
    if suffix == ".rtf":
        text = read_text_bytes(path)
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
        return {"engine": "rtf-rough", "status": "ok", "text": normalize_ws(re.sub(r"[{}]", " ", text))[:MAX_TEXT_CHARS]}
    return {"engine": "document_text", "status": "unsupported", "text": ""}


def image_analysis(path: Path) -> dict[str, Any]:
    ident = run_tool(["identify", "-format", "%m %w %h %[colorspace]", str(path)], timeout=30)
    ocr = run_tool(["tesseract", str(path), "stdout"], timeout=120)
    return {
        "identify": {"status": "ok" if ident["ok"] else "failed", "summary": ident["stdout"].strip(), "error": ident["stderr"][-500:]},
        "ocr": {"engine": "tesseract", "status": "ok" if ocr["ok"] else "failed", "text": ocr["stdout"][:MAX_TEXT_CHARS], "error": ocr["stderr"][-500:]},
        "image_recognition": {
            "engine": "deterministic_metadata_ocr_heuristic",
            "status": "metadata_and_ocr_only",
            "note": "No local object-recognition model is configured; this lane records dimensions/colorspace/OCR without inventing labels.",
        },
    }


def ffprobe_json(path: Path) -> dict[str, Any]:
    out = run_tool(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)], timeout=60)
    if not out["ok"]:
        return {"engine": "ffprobe", "status": "failed", "error": out["stderr"][-500:]}
    try:
        data = json.loads(out["stdout"] or "{}")
        data["engine"] = "ffprobe"
        data["status"] = "ok"
        return data
    except Exception as exc:
        return {"engine": "ffprobe", "status": "parse_failed", "error": str(exc)}


def video_frame_ocr(path: Path, digest: str) -> dict[str, Any]:
    frame_dir = FRAME_ROOT / digest[:16]
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.jpg"):
        try:
            old.unlink()
        except OSError as exc:
            print(f"warning: unable to remove stale frame {old}: {exc}", file=sys.stderr)
    out = run_tool([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", "fps=1/20", "-frames:v", "5", str(frame_dir / "frame_%03d.jpg")
    ], timeout=120)
    frames = []
    for frame in sorted(frame_dir.glob("frame_*.jpg")):
        ia = image_analysis(frame)
        frames.append({"frame": rel(frame), "analysis": ia})
    return {"engine": "ffmpeg+tesseract", "status": "ok" if frames else "no_frames", "ffmpeg": out["stderr"][-500:], "frames": frames}


def archive_listing(path: Path) -> dict[str, Any]:
    entries: list[str] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                entries = zf.namelist()[:1000]
            return {"engine": "zipfile", "status": "ok", "entries": entries, "entry_count_returned": len(entries)}
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                entries = [m.name for m in tf.getmembers()[:1000]]
            return {"engine": "tarfile", "status": "ok", "entries": entries, "entry_count_returned": len(entries)}
    except Exception as exc:
        return {"engine": "archive", "status": "failed", "error": str(exc)}
    return {"engine": "archive", "status": "unsupported", "entries": []}


def analyze_artifact(path: Path, mime: str, file_kind: str, digest: str) -> tuple[str, dict[str, Any]]:
    analysis: dict[str, Any] = {"file_kind": file_kind, "processors": []}
    text_parts: list[str] = []
    if file_kind == "image":
        ia = image_analysis(path)
        analysis["image"] = ia
        analysis["processors"].extend(["identify", "tesseract"])
        text_parts.append(ia.get("ocr", {}).get("text", ""))
    elif file_kind in {"document", "text"}:
        doc = extract_document_text(path, mime)
        analysis["document"] = {k: v for k, v in doc.items() if k != "text"}
        analysis["processors"].append(doc.get("engine", "document_text"))
        text_parts.append(str(doc.get("text", "")))
    elif file_kind in {"video", "audio"}:
        probe = ffprobe_json(path)
        analysis["ffprobe"] = probe
        analysis["processors"].append("ffprobe")
        if file_kind == "video":
            frame = video_frame_ocr(path, digest)
            analysis["frame_ocr"] = frame
            analysis["processors"].extend(["ffmpeg", "tesseract"])
            for item in frame.get("frames", []):
                text_parts.append(item.get("analysis", {}).get("ocr", {}).get("text", ""))
        # Local ASR is intentionally not faked.
        if file_kind == "audio":
            analysis["transcription"] = {"status": "queued_unavailable", "note": "Local ASR model not configured; ffprobe metadata captured."}
    elif file_kind == "archive":
        analysis["archive"] = archive_listing(path)
        analysis["processors"].append("archive_listing")
    return normalize_ws("\n\n".join(t for t in text_parts if t))[:MAX_TEXT_CHARS], analysis


@dataclass(frozen=True)
class Entity:
    entity_kind: str
    value: str
    normalized_value: str
    confidence_bps: int
    source: str
    context: str
    detail: dict[str, Any]


def context_for(text: str, start: int, end: int, radius: int = 80) -> str:
    return normalize_ws(text[max(0, start - radius): min(len(text), end + radius)])[:320]


def normalize_entity(kind: str, value: str) -> str:
    value = normalize_ws(value)
    if kind == "phone":
        return re.sub(r"\D+", "", value)
    if kind in {"email", "domain", "hash", "identifier"}:
        return value.lower()
    if kind == "url":
        try:
            p = urlparse(value.strip())
            netloc = p.netloc.lower().removeprefix("www.")
            path = p.path.rstrip("/")
            return urlunparse((p.scheme.lower() or "https", netloc, path, "", p.query, ""))
        except ValueError:
            return value.lower()
    return value.lower()


def _entity_domain(kind: str, value: str) -> str:
    if kind == "domain":
        return value.lower().removeprefix("www.").strip(" ./")
    if kind == "url":
        try:
            return urlparse(value).netloc.lower().removeprefix("www.").strip(" ./")
        except ValueError:
            return ""
    if kind == "email" and "@" in value:
        return value.split("@", 1)[1].lower().removeprefix("www.").strip(" ./")
    return ""


def _is_noise_domain(domain: str) -> bool:
    if not domain or "." not in domain:
        return True
    if domain in NOISE_DOMAINS:
        return True
    if any(domain.endswith("." + d) or domain == d for d in NOISE_DOMAINS):
        return True
    return any(domain.startswith(prefix) for prefix in NOISE_DOMAIN_PREFIXES)


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _looks_like_numbered_company(value: str) -> bool:
    return any(rx.search(value) for rx in IDENTIFIER_ACTOR_PATTERNS)


def actor_entity_class(ent: dict[str, Any]) -> str:
    """Return the graph-worthy actor class, or empty string for raw-only facts.

    Entity means an actor/contact surface/organization/address/website node: a thing
    that can act, be operated, be owned, convey action, or anchor an actor.
    Dates, comments, likes, reactions, raw money mentions, hashes, pages, and chunk
    titles stay as evidence attributes unless an operator/model later promotes them.
    """
    kind = str(ent.get("entity_kind") or "").lower()
    value = normalize_ws(str(ent.get("value") or ent.get("normalized_value") or ""))
    norm = normalize_entity(kind, value)
    source = str(ent.get("source") or "")
    if kind in NON_ACTOR_ENTITY_KINDS:
        return ""
    if kind == "phone":
        return "contact_phone" if len(norm) >= 7 else ""
    if kind == "email":
        return "contact_email" if "@" in value else ""
    if kind == "alias":
        return "handle_alias" if len(norm) >= 2 else ""
    if kind == "address":
        parts = norm.split()
        if len(parts) >= 2 and parts[1].strip(".,") in ADDRESS_FALSE_SECOND_WORDS:
            return ""
        if "<" in norm or len(norm) > 120:
            return ""
        if not re.search(r"\b(street|st|avenue|ave|road|rd|boulevard|blvd|court|ct|lane|ln|way|place|pl|highway|hwy)\b", norm, re.I):
            return ""
        return "address_actor_anchor" if len(norm) >= 8 else ""
    if kind == "organization":
        return "business_organization"
    if kind == "domain":
        domain = _entity_domain(kind, value)
        return "" if _is_noise_domain(domain) else "website_actor_anchor"
    if kind == "ip":
        return "network_endpoint" if _is_public_ip(norm) else ""
    if kind == "identifier":
        return "numbered_company_or_registered_asset" if _looks_like_numbered_company(value) else ""
    if kind == "name":
        words = [w.strip(".,'\"()[]{}").lower() for w in value.split()]
        strong_words = [w for w in words if len(w) >= 2 and w not in NOISE_NAME_WORDS]
        if source == "metadata:Title":
            return ""
        if len(strong_words) < 2:
            return ""
        confidence = int(ent.get("confidence_bps") or 0)
        trusted_source = source in {"metadata:Author", "metadata:Creator", "metadata:Artist", "metadata:OwnerName", "metadata:Company"}
        heuristic_override = os.environ.get("LUCIDOTA_PROMOTE_HEURISTIC_NAMES", "").strip().lower() in {"1", "true", "yes", "on"}
        if confidence < 50 and not trusted_source and not heuristic_override:
            return ""
        return "human_or_business_name_candidate"
    return ""


def should_promote_entity_to_graph(ent: dict[str, Any]) -> bool:
    return bool(actor_entity_class(ent))


def entity_graph_term(ent: dict[str, Any]) -> str:
    kind = str(ent.get("entity_kind") or "").lower()
    if kind in {"phone", "email", "alias", "domain", "ip", "identifier"}:
        return "ATOMIC_ID"
    if kind == "address":
        return "LOCATION"
    if kind == "organization":
        return "GROUP"
    return "ENTITY"


def entity_graph_location(ent: dict[str, Any]) -> str:
    kind = str(ent.get("entity_kind") or "").lower()
    norm = str(ent.get("normalized_value") or normalize_entity(kind, str(ent.get("value") or "")))
    actor_class = actor_entity_class(ent) or "raw"
    digest = hashlib.sha256(f"{kind}:{norm}".encode("utf-8", errors="ignore")).hexdigest()[:24]
    return f"korpus-actor://{actor_class}/{kind}/{digest}"


def entity_graph_payload(ent: dict[str, Any], file_sha256: str) -> dict[str, Any]:
    actor_class = actor_entity_class(ent)
    kind = str(ent.get("entity_kind") or "").lower()
    return {
        "kind": "korpus_entity",
        "entity": ent,
        "entity_is_actor": bool(actor_class),
        "actor_entity_class": actor_class,
        "first_class_actor_entity": bool(actor_class),
        "contact_surface_node": kind in {"phone", "email", "alias", "domain", "ip", "address"},
        "identity_resolution_policy": "Contact surfaces are preserved as graph nodes by default. Collapse into a person/organization is an explicit operator/case decision, never an automatic ingest side effect. No evidence is deleted; node status/canonical links are reversible graph state.",
        "file_sha256": file_sha256,
        "evidence_note": "Actor/contact-surface entity extracted by deterministic KORPUS pipeline under Operator entity ontology.",
    }


def add_entity(found: dict[tuple[str, str], Entity], kind: str, value: str, confidence: int, source: str, text: str, start: int = 0, end: int = 0, detail: dict[str, Any] | None = None) -> None:
    value = normalize_ws(value).strip(" ,;:()[]{}<>")
    if not value:
        return
    norm = normalize_entity(kind, value)
    if kind == "phone" and len(norm) < 7:
        return
    if kind == "name":
        words = [w.lower().strip(".,") for w in value.split()]
        if not words or any(w in COMMON_NAME_FALSE_POSITIVES for w in words):
            return
    key = (kind, norm)
    ent = Entity(kind, value, norm, confidence, source, context_for(text, start, end) if text else "", detail or {})
    prior = found.get(key)
    if prior is None or ent.confidence_bps > prior.confidence_bps:
        found[key] = ent


def extract_entities(text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], Entity] = {}
    scan = text[:MAX_TEXT_CHARS]
    patterns = [
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), 69),
        ("url", re.compile(r"\bhttps?://[^\s<>'\")]+", re.I), 69),
        ("phone", re.compile(r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\w)"), 69),
        ("ip", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 69),
        ("hash", re.compile(r"\b[a-fA-F0-9]{32,64}\b"), 69),
        ("date", re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b", re.I), 50),
        ("money", re.compile(r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{2})?\b"), 50),
        ("address", re.compile(r"\b\d{1,6}\s+[A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*){0,6}\s+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Court|Ct\.?|Lane|Ln\.?|Way|Place|Pl\.?|Highway|Hwy\.?)\b(?:[^\n,;]{0,80})?", re.I), 50),
        ("identifier", re.compile(r"\b[A-Z]{2,10}[-_ ]?\d{3,12}\b|\b\d{8,16}\b"), 50),
    ]
    for kind, rx, conf in patterns:
        for m in rx.finditer(scan):
            add_entity(found, kind, m.group(0), conf, f"regex:{kind}", scan, m.start(), m.end())
    for m in re.finditer(r"\b(?:aka|a/k/a|alias|known as|also known as)[:\s]+([A-Z][A-Za-z0-9_'\-.]+(?:\s+[A-Z][A-Za-z0-9_'\-.]+){0,4})", scan, re.I):
        alias = re.split(r"\s+(?:met|with|called|contacted|emailed|at|on|from|near)\b", m.group(1), maxsplit=1, flags=re.I)[0]
        add_entity(found, "alias", alias, 69, "regex:alias", scan, m.start(1), m.start(1) + len(alias))
    for m in re.finditer(r"\b([A-Z][a-z][A-Za-z'\-.]+(?:\s+[A-Z][a-z][A-Za-z'\-.]+){1,3})\b", scan):
        add_entity(found, "name", m.group(1), 10, "heuristic:capitalized_sequence", scan, m.start(1), m.end(1))
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9&.,'\- ]{2,80}\s+(?:Inc\.?|Ltd\.?|LLC|Corp\.?|Corporation|Society|Association|Agency|Commission|Party|Union|Foundation))\b", scan):
        add_entity(found, "organization", m.group(1), 50, "regex:organization_suffix", scan, m.start(1), m.end(1))
    # Domain pivots from URLs and emails.
    for ent in list(found.values()):
        domain = ""
        if ent.entity_kind == "url":
            domain = urlparse(ent.value).netloc.lower().removeprefix("www.")
        elif ent.entity_kind == "email" and "@" in ent.value:
            domain = ent.value.split("@", 1)[1].lower()
        if domain:
            add_entity(found, "domain", domain, 69, f"derived:{ent.entity_kind}", scan, 0, 0, {"derived_from": ent.value})
    # Metadata title/author fields.
    for key in ("Title", "Author", "Creator", "Artist", "OwnerName", "Company"):
        val = str(metadata.get(key) or "").strip()
        if val:
            add_entity(found, "other" if key == "Title" else "name", val, 10, f"metadata:{key}", scan, 0, 0)
    return [asdict(ent) for ent in sorted(found.values(), key=lambda e: (e.entity_kind, e.normalized_value, e.value))]


def ensure_storage(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(VAULT_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(GO_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(INVESTIGATION_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def ensure_state(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        row = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
            VALUES (%s,%s,%s,%s,'lucidota_artifact_ingest',%s::jsonb)
            RETURNING event_id::text
            """,
            (workflow_id, run_id, phase, status, jdump(detail)),
        ).fetchone()
        conn.commit()
    return str(row[0])


def insert_graph_item(conn: psycopg.Connection, term: str, label: str, location: str, payload: dict[str, Any], *, layer: str = "map", role: str = "investigation") -> str:
    mode = GRAPH_APPROVAL_MODE
    item_payload = {
        **payload,
        "graph_write_mode": mode,
        "approval_required": not graph_direct_approved(),
        "approval_source": "lucidota_artifact_ingest",
    }
    existing = conn.execute(
        """
        SELECT uuid::text
        FROM lucidota_go.graph_item
        WHERE location_at_on_graph=%s
          AND status <> 'lost'
        ORDER BY
          CASE status
            WHEN 'approved' THEN 0
            WHEN 'staged' THEN 1
            WHEN 'located' THEN 2
            ELSE 3
          END,
          created_at ASC
        LIMIT 1
        """,
        (location,),
    ).fetchone()
    inserted = existing is None
    if existing:
        uuid_ = str(existing[0])
        conn.execute(
            """
            UPDATE lucidota_go.graph_item
            SET updated_at=now(),
                payload = CASE
                  WHEN payload ? 'last_seen_by' THEN payload || %s::jsonb
                  ELSE payload || %s::jsonb
                END
            WHERE uuid=%s::uuid
            """,
            (
                jdump({"last_seen_by": "lucidota_artifact_ingest", "last_seen_payload_kind": item_payload.get("kind", "")}),
                jdump({"last_seen_by": "lucidota_artifact_ingest", "last_seen_payload_kind": item_payload.get("kind", "")}),
                uuid_,
            ),
        )
    elif graph_direct_approved():
        row = conn.execute(
            """
            INSERT INTO lucidota_go.graph_item (
              term, label, status, time_on_graph, location_at_on_graph,
              location_real_at_added, time_approved, location_real_at_approved,
              approval_scope, operator_uuid, payload
            ) VALUES (%s,%s,'approved',now(),%s,%s::jsonb,now(),%s::jsonb,'current',%s,%s::jsonb)
            RETURNING uuid::text
            """,
            (
                term,
                label[:240],
                location,
                jdump({"surface": "investigation", "host": "local"}),
                jdump({"surface": "investigation", "host": "local"}),
                OPERATOR_UUID,
                jdump(item_payload),
            ),
        ).fetchone()
        uuid_ = str(row[0])
    else:
        row = conn.execute(
            """
            INSERT INTO lucidota_go.graph_item (
              term, label, status, time_on_graph, location_at_on_graph,
              location_real_at_added, operator_uuid, payload
            ) VALUES (%s,%s,'staged',now(),%s,%s::jsonb,%s,%s::jsonb)
            RETURNING uuid::text
            """,
            (
                term,
                label[:240],
                location,
                jdump({"surface": "investigation", "host": "local"}),
                OPERATOR_UUID,
                jdump(item_payload),
            ),
        ).fetchone()
        uuid_ = str(row[0])
    conn.execute(
        """
        INSERT INTO lucidota_go.graph_item_layer(uuid, layer, role, confidence_bps, operator_uuid, detail)
        VALUES (%s,%s,%s,50,%s,%s::jsonb)
        ON CONFLICT(uuid, layer, role) DO UPDATE SET detail=EXCLUDED.detail, confidence_bps=EXCLUDED.confidence_bps
        """,
        (uuid_, layer, role, OPERATOR_UUID, jdump({"source": "lucidota_artifact_ingest"})),
    )
    if inserted:
        conn.execute(
            """
            INSERT INTO lucidota_go.graph_journal(item_uuid, operator_uuid, action, reason, after_state)
            VALUES (%s,%s,%s,%s,%s::jsonb)
            """,
            (
                uuid_,
                OPERATOR_UUID,
                "approve" if graph_direct_approved() else "stage",
                "artifact ingest graph anchor",
                jdump({"term": term, "label": label[:240], "status": mode, "location": location}),
            ),
        )
    return uuid_


def insert_graph_edge(conn: psycopg.Connection, source_uuid: str, target_uuid: str, edge_type: str, detail: dict[str, Any], evidence_uuid: str | None = None) -> str:
    mode = GRAPH_APPROVAL_MODE
    edge_detail = {**detail, "graph_write_mode": mode, "approval_required": not graph_direct_approved()}
    row = conn.execute(
        """
        INSERT INTO lucidota_go.graph_edge(
          source_uuid, target_uuid, edge_type, term, status, current_status,
          current_unknown, evidence_uuid, operator_uuid, detail
        ) VALUES (%s,%s,%s,'RELATIONSHIP',%s,%s,%s,%s,%s,%s::jsonb)
        RETURNING edge_uuid::text
        """,
        (
            source_uuid,
            target_uuid,
            edge_type,
            mode,
            "yes" if graph_direct_approved() else "unknown",
            not graph_direct_approved(),
            evidence_uuid,
            OPERATOR_UUID,
            jdump(edge_detail),
        ),
    ).fetchone()
    edge_uuid = str(row[0])
    conn.execute(
        """
        INSERT INTO lucidota_go.graph_journal(edge_uuid, operator_uuid, action, reason, after_state)
        VALUES (%s,%s,%s,%s,%s::jsonb)
        """,
        (
            edge_uuid,
            OPERATOR_UUID,
            "approve" if graph_direct_approved() else "stage",
            "artifact ingest graph edge",
            jdump({"edge_type": edge_type, "status": mode, "source_uuid": source_uuid, "target_uuid": target_uuid}),
        ),
    )
    return edge_uuid


def insert_actor_relation_edge(
    conn: psycopg.Connection,
    actor_uuid: str,
    entity_uuid: str,
    relation: str,
    detail: dict[str, Any],
    *,
    evidence_uuid: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    current_status: str = "unknown",
    confirmed: bool = False,
    collapse_contact_surface: bool = False,
) -> str:
    """Attach actor nodes to contact surfaces/assets without automatic deletion.

    Operator ontology: phone/address/email/handle/domain nodes generally stay up.
    Collapse into a person/org only when the Operator or a case-specific workflow
    explicitly sets collapse_contact_surface=True. Ingest never destroys evidence.
    """
    relation = relation.strip().upper()
    allowed = {
        "USES_CONTACT_SURFACE",
        "OWNS_CONTACT_SURFACE",
        "OPERATES_WEBSITE",
        "OPERATES_ADDRESS",
        "CONTROLS_ASSET",
        "FORMERLY_USED_CONTACT_SURFACE",
        "SHARED_CONTACT_SURFACE",
        "UNKNOWN_ACTOR_BEHIND_CONTACT_SURFACE",
    }
    if relation not in allowed:
        raise ValueError(f"unsupported actor relation: {relation}")
    if current_status not in {"yes", "no", "unknown"}:
        raise ValueError(f"unsupported current_status: {current_status}")
    mode = "approved" if confirmed and graph_direct_approved() else GRAPH_APPROVAL_MODE
    edge_detail = {
        **detail,
        "actor_relation_policy": "Contact surfaces are preserved as graph nodes by default; collapse into a person/org is explicit operator/case state, never automatic ingest cleanup.",
        "collapse_contact_surface_requested": collapse_contact_surface,
        "confirmed": confirmed,
        "graph_write_mode": mode,
    }
    row = conn.execute(
        """
        INSERT INTO lucidota_go.graph_edge(
          source_uuid, target_uuid, edge_type, term, relationship_family, status,
          valid_from, valid_to, current_status, current_unknown, evidence_uuid,
          operator_uuid, detail
        ) VALUES (%s::uuid,%s::uuid,%s,'RELATIONSHIP','possessive',%s,%s::timestamptz,%s::timestamptz,%s,%s,%s::uuid,%s,%s::jsonb)
        RETURNING edge_uuid::text
        """,
        (
            actor_uuid,
            entity_uuid,
            relation,
            "approved" if confirmed and graph_direct_approved() else mode,
            valid_from,
            valid_to,
            current_status,
            current_status == "unknown",
            evidence_uuid,
            OPERATOR_UUID,
            jdump(edge_detail),
        ),
    ).fetchone()
    edge_uuid = str(row[0])
    conn.execute(
        """
        INSERT INTO lucidota_go.graph_journal(edge_uuid, operator_uuid, action, reason, after_state)
        VALUES (%s,%s,%s,%s,%s::jsonb)
        """,
        (
            edge_uuid,
            OPERATOR_UUID,
            "approve" if confirmed and graph_direct_approved() else "stage",
            "actor/contact-surface relation",
            jdump({"relation": relation, "current_status": current_status, "valid_from": valid_from, "valid_to": valid_to}),
        ),
    )
    if collapse_contact_surface and confirmed and current_status == "yes" and relation in {
        "USES_CONTACT_SURFACE",
        "OWNS_CONTACT_SURFACE",
        "OPERATES_WEBSITE",
        "OPERATES_ADDRESS",
        "CONTROLS_ASSET",
    }:
        conn.execute(
            """
            UPDATE lucidota_go.graph_item
            SET status='collapsed',
                canonical_uuid=%s::uuid,
                payload = payload || %s::jsonb,
                updated_at=now()
            WHERE uuid=%s::uuid
              AND payload->>'contact_surface_node'='true'
            """,
            (
                actor_uuid,
                jdump(
                    {
                        "collapsed_into_actor_uuid": actor_uuid,
                        "collapse_reason": "operator_requested_confirmed_current_single_responsible_actor",
                        "collapse_relation": relation,
                        "collapse_policy": "explicit_operator_or_case_decision_contact_surface_vanishes_into_person_or_org",
                        "collapse_at": now_utc(),
                    }
                ),
                entity_uuid,
            ),
        )
    return edge_uuid


def reactivate_contact_surface_node(
    conn: psycopg.Connection,
    entity_uuid: str,
    reason: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Make a formerly collapsed contact surface stand alone again when owner is gone/unknown."""
    conn.execute(
        """
        UPDATE lucidota_go.graph_item
        SET status='staged',
            canonical_uuid=NULL,
            payload = payload || %s::jsonb,
            updated_at=now()
        WHERE uuid=%s::uuid
          AND payload->>'contact_surface_node'='true'
        """,
        (
            jdump(
                {
                    "reactivated_contact_surface_node": True,
                    "reactivation_reason": reason,
                    "reactivation_policy": "contact_surface_is_node_when_person_or_org_is_gone_unknown_shared_or_ambiguous",
                    "reactivation_detail": detail or {},
                    "reactivated_at": now_utc(),
                }
            ),
            entity_uuid,
        ),
    )


def write_casefile(conn: psycopg.Connection, case_uuid: str) -> Path:
    row = conn.execute(
        """
        SELECT case_key, title, status, opened_at::text, summary, folder_relative_path, casefile_relative_path, detail
        FROM lucidota_investigation.case_file WHERE case_uuid=%s::uuid
        """,
        (case_uuid,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"case not found: {case_uuid}")
    case_key, title, status, opened_at, summary, folder_rel, casefile_rel, detail = row
    artifacts = conn.execute(
        """
        SELECT ca.evidence_id, a.sha256, ca.artifact_title, a.mime, a.file_kind, ca.evidence_date::text, ca.status
        FROM lucidota_investigation.case_artifact ca
        JOIN lucidota_investigation.artifact a ON a.artifact_uuid=ca.artifact_uuid
        WHERE ca.case_uuid=%s::uuid
        ORDER BY ca.evidence_id
        """,
        (case_uuid,),
    ).fetchall()
    caps = conn.execute(
        """
        SELECT capability_group, capability_name, lifecycle_status, run_state, workflow_name, command
        FROM lucidota_investigation.capability_registry
        ORDER BY capability_group, capability_name
        """
    ).fetchall()
    data = {
        "case_key": case_key,
        "title": title,
        "status": status,
        "opened_at": opened_at,
        "summary": summary,
        "folder_shape": detail.get("folder_shape", {}) if isinstance(detail, dict) else {},
        "nomenclature": {
            "case_key": "CASE-YYYYMMDD-SLUG",
            "evidence_id": f"{case_key}-EVID-000001",
            "sidecar_prefix": "<EVIDENCE_ID>.<artifact-sha12>",
        },
        "artifact_count": len(artifacts),
        "artifacts": [
            {"evidence_id": r[0], "sha256": r[1], "title": r[2], "mime": r[3], "kind": r[4], "evidence_date": r[5], "status": r[6]}
            for r in artifacts
        ],
        "capability_modules": [
            {"group": r[0], "name": r[1], "lifecycle": r[2], "run_state": r[3], "workflow": r[4], "command": r[5]}
            for r in caps
        ],
        "updated_at": now_utc(),
    }
    path = ROOT / str(casefile_rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        path.write_text(jdump(data), encoding="utf-8")
    return path


def ensure_case(conn: psycopg.Connection, case_arg: str, title: str = "", summary: str = "") -> dict[str, Any]:
    case_key = normalize_case_key(case_arg, title)
    shape = ensure_case_shape(case_key)
    existing = conn.execute(
        "SELECT case_uuid::text, graph_item_uuid::text, title FROM lucidota_investigation.case_file WHERE case_key=%s",
        (case_key,),
    ).fetchone()
    if existing:
        case_uuid, graph_uuid, old_title = existing
        if title and title != old_title:
            conn.execute("UPDATE lucidota_investigation.case_file SET title=%s, updated_at=now() WHERE case_uuid=%s::uuid", (title, case_uuid))
    else:
        graph_uuid = None
        case_uuid = str(conn.execute(
            """
            INSERT INTO lucidota_investigation.case_file(case_key, title, summary, folder_relative_path, casefile_relative_path, detail)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            RETURNING case_uuid::text
            """,
            (case_key, title or case_key, summary, shape["folder"], shape["casefile"], jdump({"folder_shape": shape})),
        ).fetchone()[0])
    if not graph_uuid:
        graph_uuid = insert_graph_item(
            conn,
            "GROUP",
            f"Case {case_key}: {title or case_key}",
            f"case://{case_key}",
            {"kind": "investigation_case", "case_key": case_key, "title": title or case_key, "evidence_note": "Investigative case container created locally."},
            layer="map",
            role="case",
        )
        conn.execute("UPDATE lucidota_investigation.case_file SET graph_item_uuid=%s::uuid, updated_at=now() WHERE case_uuid=%s::uuid", (graph_uuid, case_uuid))
    write_casefile(conn, case_uuid)
    return {"case_uuid": case_uuid, "case_key": case_key, "graph_item_uuid": graph_uuid, "folder_shape": shape}


def store_locked_cas(path: Path, digest: str, mime: str, source: str) -> tuple[str, Path]:
    subdir = CAS_ROOT / digest[:2] / digest[2:4]
    subdir.mkdir(parents=True, exist_ok=True)
    locked = subdir / digest
    if not locked.exists():
        tmp = locked.with_name(f"{locked.name}.{os.getpid()}.{time.time_ns()}.tmp")
        shutil.copyfile(path, tmp)
        os.chmod(tmp, 0o440)
        try:
            os.link(tmp, locked)
        except FileExistsError:
            if sha256_file(locked) != digest:
                raise RuntimeError(f"CAS collision/corruption at {locked}")
        finally:
            tmp.unlink(missing_ok=True)
    if not locked.exists():
        raise RuntimeError(f"CAS lock failed: {locked}")
    if sha256_file(locked) != digest:
        raise RuntimeError(f"CAS digest verification failed for {locked}")
    os.chmod(locked, 0o440)
    cas_uri = f"cas://sha256/{digest}"
    return cas_uri, locked


def lock_case_evidence_file(cas_path: Path, digest: str, case_key: str, evidence_id: str, original_name: str) -> Path:
    """Expose a case-local locked evidence file without duplicating bytes when possible."""
    evidence_dir = case_folder(case_key) / "01_EVIDENCE_LOCKED"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name).suffix[:32]
    target = evidence_dir / f"{evidence_id}.{digest[:12]}{suffix}"
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f"case evidence path exists with wrong digest: {target}")
        os.chmod(target, 0o440)
        return target
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        try:
            os.link(cas_path, target)
        except OSError:
            shutil.copyfile(cas_path, tmp)
            os.chmod(tmp, 0o440)
            os.replace(tmp, target)
        if sha256_file(target) != digest:
            raise RuntimeError(f"case evidence digest verification failed: {target}")
        os.chmod(target, 0o440)
        return target
    finally:
        tmp.unlink(missing_ok=True)


def next_evidence_id(conn: psycopg.Connection, case_key: str, case_uuid: str) -> str:
    row = conn.execute(
        "SELECT count(*) FROM lucidota_investigation.case_artifact WHERE case_uuid=%s::uuid",
        (case_uuid,),
    ).fetchone()
    return f"{case_key}-EVID-{int(row[0]) + 1:06d}"


def write_sidecars(base: Path, prefix: str, normalized: dict[str, Any], entities: list[dict[str, Any]], text: str, custody: dict[str, Any]) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "normalized_json": base / f"{prefix}.normalized.json",
        "normalized_yaml": base / f"{prefix}.normalized.yaml",
        "entities_jsonl": base / f"{prefix}.entities.jsonl",
        "custody_jsonl": base / f"{prefix}.custody.jsonl",
        "text_extract": base / f"{prefix}.text.txt",
    }
    paths["normalized_json"].write_text(json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    if yaml:
        paths["normalized_yaml"].write_text(yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        paths["normalized_yaml"].write_text(json.dumps(normalized, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")
    paths["entities_jsonl"].write_text("".join(json.dumps(e, ensure_ascii=False, sort_keys=True, default=str) + "\n" for e in entities), encoding="utf-8")
    paths["custody_jsonl"].write_text(json.dumps(custody, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    paths["text_extract"].write_text(text, encoding="utf-8", errors="ignore")
    return paths


def upsert_sidecars(conn: psycopg.Connection, artifact_uuid: str, case_uuid: str | None, paths: dict[str, Path]) -> None:
    mime_map = {"normalized_json": "application/json", "normalized_yaml": "application/x-yaml", "entities_jsonl": "application/jsonl", "custody_jsonl": "application/jsonl", "text_extract": "text/plain"}
    for kind, path in paths.items():
        digest = sha256_file(path)
        conn.execute(
            """
            INSERT INTO lucidota_investigation.artifact_sidecar(artifact_uuid, case_uuid, sidecar_kind, relative_path, sha256, mime)
            VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s)
            ON CONFLICT(artifact_uuid, case_uuid, sidecar_kind, relative_path) DO UPDATE SET
              sha256=EXCLUDED.sha256, mime=EXCLUDED.mime
            """,
            (artifact_uuid, case_uuid, kind, rel(path), digest, mime_map.get(kind, "application/octet-stream")),
        )


def insert_tags(conn: psycopg.Connection, artifact_uuid: str, case_uuid: str | None, file_kind: str, entities: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    tags = [(f"artifact:{file_kind if file_kind in {'image','document','video','audio','text','binary'} else 'binary'}", file_kind, "artifact_kind")]
    tags.append(("case:associated" if case_uuid else "case:unassigned", "", "case_state"))
    tags.append(("processing:entity_extract", str(len(entities)), "entity_extract"))
    if analysis.get("image"):
        tags.append(("processing:ocr", "image", "processor"))
    if analysis.get("document"):
        tags.append(("processing:document_text", "document", "processor"))
    if analysis.get("ffprobe"):
        tags.append(("processing:ffprobe", "media", "processor"))
    for ent in entities:
        key = f"entity:{ent['entity_kind']}"
        if key in {"entity:name", "entity:alias", "entity:phone", "entity:ip", "entity:email", "entity:url", "entity:domain", "entity:address", "entity:date", "entity:identifier"}:
            tags.append((key, ent["normalized_value"], "entity"))
    for tag_key, value, source in tags:
        conn.execute(
            """
            INSERT INTO lucidota_investigation.artifact_tag(artifact_uuid, case_uuid, tag_key, value, confidence_bps, source, detail)
            VALUES (%s::uuid,%s::uuid,%s,%s,50,%s,%s::jsonb)
            ON CONFLICT(artifact_uuid, case_uuid, tag_key, value, source) DO NOTHING
            """,
            (artifact_uuid, case_uuid, tag_key, value[:240], source, jdump({})),
        )


def cmd_case_create(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        case = ensure_case(conn, args.case or args.title or "UNTITLED", args.title or args.case or "Untitled Case", args.summary or "")
        conn.commit()
    event_id = emit_event("case-create", case["case_uuid"], "case", "succeeded", case)
    return {"ok": True, "case": case, "graph_approval_mode": GRAPH_APPROVAL_MODE, "workflow_event": event_id}


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"artifact file not found: {path}")
    digest = sha256_file(path)
    mime, file_desc = detect_mime(path)
    file_kind = classify_kind(path, mime)
    exif = exiftool_json(path)
    evidence_date, evidence_date_source = choose_evidence_date(path, args.evidence_date or "", exif)
    title = title_from_metadata(path, args.title or "", exif)
    cas_uri, locked_path = store_locked_cas(path, digest, mime, "artifact_ingest")
    text, analysis = analyze_artifact(path, mime, file_kind, digest)
    entities = extract_entities(text, {**exif, "FileName": path.name, "Title": title})
    custody = {
        "event": "artifact_ingest",
        "sha256": digest,
        "source_path": str(path),
        "locked_path": rel(locked_path),
        "cas_uri": cas_uri,
        "operator_uuid": OPERATOR_UUID,
        "timestamp": now_utc(),
        "hash_algorithm": "sha256",
        "write_mode": "copy_then_chmod_0440",
    }
    case: dict[str, Any] | None = None
    evidence_id = f"UNASSIGNED-EVID-{digest[:12].upper()}"
    sidecar_base = OUTPUT_ROOT / "UNASSIGNED" / digest[:16]
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        artifact_graph = None
        existing = conn.execute("SELECT artifact_uuid::text, graph_item_uuid::text FROM lucidota_investigation.artifact WHERE sha256=%s", (digest,)).fetchone()
        if args.case:
            case = ensure_case(conn, args.case, args.case_title or args.case, "")
            linked = None
            if existing:
                linked = conn.execute(
                    "SELECT evidence_id FROM lucidota_investigation.case_artifact WHERE case_uuid=%s::uuid AND artifact_uuid=%s::uuid",
                    (case["case_uuid"], existing[0]),
                ).fetchone()
            evidence_id = str(linked[0]) if linked else next_evidence_id(conn, case["case_key"], case["case_uuid"])
            sidecar_base = case_folder(case["case_key"]) / "02_NORMALIZED" / evidence_id
            case_locked_path = lock_case_evidence_file(locked_path, digest, case["case_key"], evidence_id, path.name)
            custody["case_locked_path"] = rel(case_locked_path)
        if existing:
            artifact_uuid, artifact_graph = existing
        if not artifact_graph:
            artifact_graph = insert_graph_item(
                conn,
                "EVIDENCE",
                f"Artifact {title} [{digest[:12]}]",
                f"artifact://sha256/{digest}",
                {
                    "kind": "investigation_artifact",
                    "sha256": digest,
                    "cas_uri": cas_uri,
                    "file_kind": file_kind,
                    "mime": mime,
                    "title": title,
                    "evidence_note": "Artifact bytes hashed, locked in local CAS, and normalized by deterministic ingest.",
                },
                layer="digital_world",
                role="evidence_artifact",
            )
        if existing:
            conn.execute(
                """
                UPDATE lucidota_investigation.artifact SET
                  cas_uri=%s, locked_relative_path=%s, original_path=%s, original_name=%s, mime=%s,
                  file_kind=%s, size_bytes=%s, title=%s, evidence_date=%s::timestamptz,
                  evidence_date_source=%s, exif_json=%s::jsonb, metadata_json=%s::jsonb,
                  normalized_text=%s, analysis_json=%s::jsonb, graph_item_uuid=%s::uuid, updated_at=now()
                WHERE sha256=%s
                RETURNING artifact_uuid::text
                """,
                (cas_uri, rel(locked_path), str(path), path.name, mime, file_kind, path.stat().st_size, title, evidence_date, evidence_date_source, jdump(exif), jdump({"file_description": file_desc}), text, jdump(analysis), artifact_graph, digest),
            )
            artifact_uuid = existing[0]
        else:
            artifact_uuid = str(conn.execute(
                """
                INSERT INTO lucidota_investigation.artifact(
                  sha256, cas_uri, locked_relative_path, original_path, original_name, mime, file_kind,
                  size_bytes, title, evidence_date, evidence_date_source, exif_json, metadata_json,
                  normalized_text, analysis_json, graph_item_uuid
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s::uuid)
                RETURNING artifact_uuid::text
                """,
                (digest, cas_uri, rel(locked_path), str(path), path.name, mime, file_kind, path.stat().st_size, title, evidence_date, evidence_date_source, jdump(exif), jdump({"file_description": file_desc}), text, jdump(analysis), artifact_graph),
            ).fetchone()[0])
        conn.execute(
            """
            INSERT INTO lucidota_vault.cas_manifest(sha256, cas_uri, relative_path, size_bytes, mime, source, last_seen_at)
            VALUES (%s,%s,%s,%s,%s,'artifact_ingest',now())
            ON CONFLICT(sha256) DO UPDATE SET relative_path=EXCLUDED.relative_path, size_bytes=EXCLUDED.size_bytes, mime=EXCLUDED.mime, source=EXCLUDED.source, last_seen_at=now()
            """,
            (digest, cas_uri, rel(locked_path), path.stat().st_size, mime),
        )
        conn.execute(
            """
            INSERT INTO lucidota_vault.cas_ingest_journal(sha256, cas_uri, relative_path, size_bytes, source, stage, detail)
            VALUES (%s,%s,%s,%s,'artifact_ingest','db_committed',%s::jsonb)
            """,
            (digest, cas_uri, rel(locked_path), path.stat().st_size, jdump(custody)),
        )
        if case:
            eid_row = conn.execute(
                """
                INSERT INTO lucidota_investigation.case_artifact(case_uuid, artifact_uuid, evidence_id, artifact_title, evidence_date, evidence_date_source, sidecar_relative_dir, custody, status)
                VALUES (%s::uuid,%s::uuid,%s,%s,%s::timestamptz,%s,%s,%s::jsonb,'indexed')
                ON CONFLICT(case_uuid, artifact_uuid) DO UPDATE SET artifact_title=EXCLUDED.artifact_title, evidence_date=EXCLUDED.evidence_date, evidence_date_source=EXCLUDED.evidence_date_source, sidecar_relative_dir=EXCLUDED.sidecar_relative_dir, custody=EXCLUDED.custody, status='indexed', updated_at=now()
                RETURNING evidence_id
                """,
                (case["case_uuid"], artifact_uuid, evidence_id, title, evidence_date, evidence_date_source, rel(sidecar_base), jdump(custody)),
            ).fetchone()
            if eid_row:
                evidence_id = str(eid_row[0])
            insert_graph_edge(conn, case["graph_item_uuid"], artifact_graph, "CASE_HAS_ARTIFACT", {"evidence_id": evidence_id, "case_key": case["case_key"]}, evidence_uuid=artifact_graph)
        normalized = {
            "schema": "diogenes.artifact.normalized.v1",
            "artifact_uuid": artifact_uuid,
            "case": case,
            "evidence_id": evidence_id,
            "title": title,
            "sha256": digest,
            "cas_uri": cas_uri,
            "locked_relative_path": rel(locked_path),
            "original_path": str(path),
            "mime": mime,
            "file_kind": file_kind,
            "size_bytes": path.stat().st_size,
            "evidence_date": evidence_date,
            "evidence_date_source": evidence_date_source,
            "file_description": file_desc,
            "exif": exif,
            "analysis": analysis,
            "entities": entities,
            "custody": custody,
        }
        prefix = f"{evidence_id}.{digest[:12]}"
        sidecars = write_sidecars(sidecar_base, prefix, normalized, entities, text, custody)
        upsert_sidecars(conn, artifact_uuid, case["case_uuid"] if case else None, sidecars)
        insert_tags(conn, artifact_uuid, case["case_uuid"] if case else None, file_kind, entities, analysis)
        written_entities = []
        for ent in entities:
            graph_uuid = insert_graph_item(
                conn,
                ENTITY_TO_GO.get(ent["entity_kind"], "ENTITY"),
                f"{ent['entity_kind']}: {ent['value']}",
                f"entity://{ent['entity_kind']}/{hashlib.sha256(ent['normalized_value'].encode()).hexdigest()[:16]}",
                {"kind": "artifact_entity", "entity": ent, "artifact_sha256": digest, "evidence_note": "Entity extracted deterministically from normalized artifact text/metadata."},
                layer="map",
                role="artifact_entity",
            )
            existing_ent = conn.execute(
                """
                SELECT entity_uuid::text
                FROM lucidota_investigation.artifact_entity
                WHERE artifact_uuid=%s::uuid
                  AND coalesce(case_uuid, '00000000-0000-0000-0000-000000000000'::uuid)=coalesce(%s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
                  AND entity_kind=%s AND normalized_value=%s AND source=%s
                LIMIT 1
                """,
                (artifact_uuid, case["case_uuid"] if case else None, ent["entity_kind"], ent["normalized_value"], ent["source"]),
            ).fetchone()
            if existing_ent:
                entity_uuid = str(existing_ent[0])
                conn.execute(
                    """
                    UPDATE lucidota_investigation.artifact_entity
                    SET value=%s, confidence_bps=%s, context=%s, graph_item_uuid=%s::uuid, detail=%s::jsonb
                    WHERE entity_uuid=%s::uuid
                    """,
                    (ent["value"], ent["confidence_bps"], ent["context"], graph_uuid, jdump(ent.get("detail", {})), entity_uuid),
                )
            else:
                entity_uuid = str(conn.execute(
                    """
                    INSERT INTO lucidota_investigation.artifact_entity(
                      artifact_uuid, case_uuid, entity_kind, value, normalized_value, confidence_bps, source, context, graph_item_uuid, detail
                    ) VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s::uuid,%s::jsonb)
                    RETURNING entity_uuid::text
                    """,
                    (artifact_uuid, case["case_uuid"] if case else None, ent["entity_kind"], ent["value"], ent["normalized_value"], ent["confidence_bps"], ent["source"], ent["context"], graph_uuid, jdump(ent.get("detail", {}))),
                ).fetchone()[0])
            written_entities.append({**ent, "entity_uuid": entity_uuid, "graph_item_uuid": graph_uuid})
            insert_graph_edge(conn, artifact_graph, graph_uuid, "ARTIFACT_MENTIONS_ENTITY", {"entity_kind": ent["entity_kind"], "source": ent["source"]}, evidence_uuid=artifact_graph)
        if case:
            write_casefile(conn, case["case_uuid"])
        conn.commit()
    detail = {"artifact_uuid": artifact_uuid, "sha256": digest, "case": case, "evidence_id": evidence_id, "entity_count": len(entities), "sidecars": {k: rel(v) for k, v in sidecars.items()}, "graph_approval_mode": GRAPH_APPROVAL_MODE, "capability_modules": ["artifact-ingest", "metadata-extraction-medd", "json-sidecar-handling", "graph-network-data-processing"]}
    event_id = emit_event("artifact-ingest", artifact_uuid, "ingest", "succeeded", detail)
    return {"ok": True, **detail, "workflow_event": event_id}


def pivot_variants(kind: str, value: str, normalized: str) -> list[tuple[str, str, str, int]]:
    out: list[tuple[str, str, str, int]] = [(kind, value, normalized, 70)]
    if kind == "email" and "@" in normalized:
        local, domain = normalized.split("@", 1)
        out.append(("domain", domain, domain, 65))
        out.append(("identifier", local, local, 35))
    elif kind == "phone":
        digits = re.sub(r"\D+", "", normalized)
        if len(digits) >= 7:
            out.append(("phone", digits[-7:], digits[-7:], 45))
        if len(digits) == 11 and digits.startswith("1"):
            out.append(("phone", digits[1:], digits[1:], 55))
    elif kind == "url":
        host = urlparse(normalized).netloc.lower().removeprefix("www.")
        if host:
            out.append(("domain", host, host, 65))
    elif kind == "name":
        parts = normalized.split()
        for p in parts:
            if len(p) > 2:
                out.append(("name_part", p, p, 25))
        if len(parts) >= 2:
            out.append(("name_initial_last", f"{parts[0][0]} {parts[-1]}", f"{parts[0][0]} {parts[-1]}", 30))
    elif kind == "address":
        m = re.match(r"(\d+)\s+(.+)", normalized)
        if m:
            out.append(("address_number", m.group(1), m.group(1), 25))
            out.append(("address_street", m.group(2), m.group(2), 40))
    return out


def cmd_pivot(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        case_uuid = None
        case_key = ""
        seed_rows: list[tuple[str | None, str, str, str, str]] = []
        if args.case:
            key = normalize_case_key(args.case) if not args.case.upper().startswith("CASE-") else args.case.upper()
            row = conn.execute("SELECT case_uuid::text, case_key FROM lucidota_investigation.case_file WHERE case_key=%s", (key,)).fetchone()
            if not row:
                raise SystemExit(f"case not found: {key}")
            case_uuid, case_key = str(row[0]), str(row[1])
            seed_rows = conn.execute(
                """
                SELECT entity_uuid::text, entity_kind, value, normalized_value, source
                FROM lucidota_investigation.artifact_entity
                WHERE case_uuid=%s::uuid
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (case_uuid, args.limit * 5),
            ).fetchall()
        if args.seed:
            kind = args.kind if args.kind != "auto" else "other"
            seed_rows.append((None, kind, args.seed, normalize_entity(kind, args.seed), "operator_seed"))
        if not seed_rows:
            raise SystemExit("pivot needs --case with entities or --seed")
        job_uuid = str(conn.execute(
            """
            INSERT INTO lucidota_investigation.pivot_job(case_uuid, seed_kind, seed_value, status, depth, local_only, detail)
            VALUES (%s::uuid,%s,%s,'running',%s,true,%s::jsonb)
            RETURNING job_uuid::text
            """,
            (case_uuid, args.kind, args.seed or case_key or "case_entities", args.depth, jdump({"mode": "local_deterministic", "seed_count": len(seed_rows)})),
        ).fetchone()[0])
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for entity_uuid, kind, value, norm, source in seed_rows:
            for cand_kind, cand_value, cand_norm, base_score in pivot_variants(kind, value, norm):
                if (cand_kind, cand_norm) in seen or not cand_norm:
                    continue
                seen.add((cand_kind, cand_norm))
                like = f"%{cand_norm[:160]}%"
                entity_hits = int(conn.execute(
                    "SELECT count(*) FROM lucidota_investigation.artifact_entity WHERE normalized_value ILIKE %s OR value ILIKE %s",
                    (like, like),
                ).fetchone()[0])
                text_hits = int(conn.execute(
                    "SELECT count(*) FROM lucidota_investigation.artifact WHERE normalized_text ILIKE %s",
                    (like,),
                ).fetchone()[0])
                score = min(100, base_score + min(20, entity_hits * 3) + min(10, text_hits * 2))
                reason = f"variant from {kind}; entity_hits={entity_hits}; text_hits={text_hits}"
                row = conn.execute(
                    """
                    INSERT INTO lucidota_investigation.pivot_candidate(job_uuid, source_entity_uuid, candidate_kind, value, normalized_value, score, reason, status, detail)
                    VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s,%s,'candidate',%s::jsonb)
                    ON CONFLICT(job_uuid, candidate_kind, normalized_value) DO UPDATE SET score=GREATEST(lucidota_investigation.pivot_candidate.score, EXCLUDED.score), reason=EXCLUDED.reason
                    RETURNING candidate_uuid::text
                    """,
                    (job_uuid, entity_uuid, cand_kind, cand_value, cand_norm, score, reason, jdump({"source_kind": kind, "source_value": value, "source": source, "local_only": True})),
                ).fetchone()
                candidates.append({"candidate_uuid": str(row[0]), "kind": cand_kind, "value": cand_value, "normalized_value": cand_norm, "score": score, "reason": reason})
        candidates.sort(key=lambda c: (-int(c["score"]), c["kind"], c["normalized_value"]))
        candidates = candidates[: args.limit]
        conn.execute("UPDATE lucidota_investigation.pivot_job SET status='succeeded', finished_at=now(), detail=detail || %s::jsonb WHERE job_uuid=%s::uuid", (jdump({"candidate_count": len(candidates)}), job_uuid))
        pivot_path = None
        if case_uuid:
            pivot_dir = case_folder(case_key) / "04_PIVOTS"
            pivot_dir.mkdir(parents=True, exist_ok=True)
            pivot_path = pivot_dir / f"{job_uuid}.pivot_candidates.jsonl"
            pivot_path.write_text("".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in candidates), encoding="utf-8")
        conn.commit()
    detail = {"job_uuid": job_uuid, "case_key": case_key, "candidate_count": len(candidates), "candidates": candidates, "pivot_path": rel(pivot_path) if pivot_path else "", "local_only": True}
    event_id = emit_event("artifact-pivot", job_uuid, "pivot", "succeeded", detail)
    return {"ok": True, **detail, "workflow_event": event_id}


def cmd_capabilities(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        rows = conn.execute(
            """
            SELECT capability_group, capability_key, capability_name, lifecycle_status, run_state, workflow_name, command, detail
            FROM lucidota_investigation.capability_registry
            ORDER BY capability_group, capability_name
            """
        ).fetchall()
    return {"ok": True, "count": len(rows), "capabilities": [dict(zip(["group", "key", "name", "lifecycle", "run_state", "workflow", "command", "detail"], r)) for r in rows]}


def cmd_shape(args: argparse.Namespace) -> dict[str, Any]:
    key = normalize_case_key(args.case or args.title or "EXAMPLE", args.title or "Example")
    shape = ensure_case_shape(key) if args.create_dirs else {k.lower(): f"03_VAULT/cases/{key}/{k}" for k in CASE_DIRS}
    return {
        "ok": True,
        "case_key": key,
        "folder_shape": shape,
        "evidence_id_shape": f"{key}-EVID-000001",
        "sidecar_shape": f"{key}-EVID-000001.<sha12>.normalized.json|yaml|entities.jsonl|custody.jsonl|text.txt",
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-artifact-ingest")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("case-create")
    p.add_argument("--case", default="")
    p.add_argument("--title", default="")
    p.add_argument("--summary", default="")
    p.add_argument("--approval-mode", choices=["staged", "approved"], default=None)
    p.set_defaults(func=cmd_case_create)

    p = sub.add_parser("ingest")
    p.add_argument("path")
    p.add_argument("--case", default="")
    p.add_argument("--case-title", default="")
    p.add_argument("--title", default="")
    p.add_argument("--evidence-date", default="")
    p.add_argument("--approval-mode", choices=["staged", "approved"], default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("pivot")
    p.add_argument("--case", default="")
    p.add_argument("--seed", default="")
    p.add_argument("--kind", default="auto")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_pivot)

    p = sub.add_parser("capabilities")
    p.set_defaults(func=cmd_capabilities)

    p = sub.add_parser("case-shape")
    p.add_argument("--case", default="")
    p.add_argument("--title", default="Example Case")
    p.add_argument("--create-dirs", action="store_true")
    p.set_defaults(func=cmd_shape)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    set_graph_approval_mode(getattr(args, "approval_mode", None))
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
