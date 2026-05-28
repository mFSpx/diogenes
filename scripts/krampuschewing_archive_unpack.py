#!/usr/bin/env python3
"""Controlled KRAMPUSCHEWING archive unpack into staging only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing" / "archive_unpack"
DEFAULT_OUT_ROOT = ROOT / "09_STORAGE" / "krampuschewing_unpacked"
CHUNK = 1024 * 1024

ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}
TEXT_EXTS = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".py", ".rs", ".sql", ".sh", ".csv", ".log", ".html", ".xml"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp", ".svg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

CASE_RE = re.compile(r"case|RTB|BCFSA|tenant|landlord|property|complaint|investigation|forensic|timeline|exhibit|affidavit|lease|rickshaw|nordley|northernstrike|coop", re.I)
DEV_RE = re.compile(r"repo|src|scripts|cargo|pytest|bug|build|work order|phase|implementation|diff|patch|TODO|dev|rust|python|compile|test|luci|lucidota", re.I)
PROMPT_RE = re.compile(r"CURRENT MODE|HARD LAW|DO NOT|MUST|prompt|operator instruction|Claude|Codex|ChatGPT", re.I)
GRAPH_RE = re.compile(r"graph|node|edge|graph_item|graph_edge|graph_journal|promotion|staging", re.I)
PROOF_RE = re.compile(r"evidence|proof|custody|receipt|screenshot|exhibit|affidavit", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def utc_from_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def safe_slug(text: str) -> str:
    text = text.replace("\\", "/").strip("/")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "archive")[:160]


def archive_records(validation: Path, only: str | None = None) -> list[dict[str, Any]]:
    data = json.loads(validation.read_text(encoding="utf-8"))
    out = []
    for rec in data.get("records", []):
        if rec.get("large_class") != "LARGE_ARCHIVE":
            continue
        if only and rec.get("relative_path") != only:
            continue
        out.append(rec)
    priority = {
        "docs_Luci-010.zip": 0,
        "C_ARCHIVE.zip": 1,
        "mfspx_Pictures-009.zip": 2,
        "docs_NORDLEY_SQUEEZECOPY-001.zip": 3,
        "Screenshots.zip": 4,
        "docs_RICKSHAW_ROBBERY-002.zip": 5,
    }
    return sorted(out, key=lambda r: (priority.get(str(r.get("relative_path")), 100), str(r.get("relative_path"))))


def archive_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar"
    if name.endswith(".tar.bz2") or name.endswith(".tbz2"):
        return "tar"
    if name.endswith(".tar.xz") or name.endswith(".txz"):
        return "tar"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".7z"):
        return "7z"
    if name.endswith(".rar"):
        return "rar"
    return "unknown"


def tool_for(kind: str) -> str | None:
    if kind == "zip":
        return "python_zipfile"
    if kind == "tar":
        return "python_tarfile"
    if kind == "7z":
        return shutil.which("7z")
    if kind == "rar":
        return shutil.which("unrar")
    return None


def unsafe_member(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    p = PurePosixPath(normalized)
    if normalized.startswith("/"):
        return "absolute_path"
    if p.parts and re.match(r"^[A-Za-z]:", p.parts[0]):
        return "drive_root"
    if ".." in p.parts:
        return "path_traversal"
    if not normalized.strip("/"):
        return "empty_path"
    return None


def target_for(out_dir: Path, member_name: str) -> Path:
    clean = member_name.replace("\\", "/").lstrip("/")
    return out_dir / Path(*PurePosixPath(clean).parts)


def kind_guess(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in TEXT_EXTS:
        if ext == ".json":
            return "json"
        if ext == ".md":
            return "markdown"
        if ext == ".py":
            return "python"
        if ext == ".rs":
            return "rust"
        if ext == ".sql":
            return "sql"
        if ext == ".sh":
            return "shell"
        return "text"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz"}:
        return "archive"
    if ext == ".pdf":
        return "pdf"
    if ext in {".sqlite", ".sqlite3", ".db"}:
        return "database"
    return "unknown"


def lane_guess(path: str) -> str:
    if GRAPH_RE.search(path):
        return "GRAPH_STAGING"
    if PROMPT_RE.search(path):
        return "PROMPT_NOTE"
    if CASE_RE.search(path):
        return "CASE_WORK"
    if DEV_RE.search(path):
        return "DEV_WORK"
    if PROOF_RE.search(path):
        return "PROOF_CANDIDATE"
    k = kind_guess(path)
    if k in {"python", "rust", "sql", "shell"}:
        return "SOURCE_CODE"
    return "SAVED_FILE"


def sha256_stream(src, dst: Path | None, size: int | None, threshold: int) -> tuple[str | None, str, int]:
    h = hashlib.sha256()
    written = 0
    hash_enabled = size is not None and size <= threshold
    with (dst.open("xb") if dst else open(os.devnull, "wb")) as out:
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if hash_enabled:
                h.update(chunk)
            if dst:
                out.write(chunk)
    return (h.hexdigest() if hash_enabled else None, "computed" if hash_enabled else "skipped_large_member", written)


def ensure_enough_space(path: Path, needed: int) -> tuple[bool, int]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    # Keep 10 GiB and 10% of estimated extraction as safety margin.
    margin = max(10 * 1024**3, int(needed * 0.10))
    return usage.free >= needed + margin, usage.free


def list_zip(path: Path) -> tuple[list[zipfile.ZipInfo], list[str], bool]:
    blockers: list[str] = []
    encrypted = False
    try:
        with zipfile.ZipFile(path) as zf:
            # Listing only. Do not run ZipFile.testzip() here: it reads every
            # member and defeats the pre-extraction safety boundary.
            infos = zf.infolist()
            encrypted = any(bool(i.flag_bits & 0x1) for i in infos)
            return infos, blockers, encrypted
    except Exception as exc:
        return [], [f"zip_list_failed:{type(exc).__name__}:{exc}"], False


def list_tar(path: Path) -> tuple[list[tarfile.TarInfo], list[str], bool]:
    try:
        with tarfile.open(path) as tf:
            return tf.getmembers(), [], False
    except Exception as exc:
        return [], [f"tar_list_failed:{type(exc).__name__}:{exc}"], False


def write_receipt(slug: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"archive_unpack_{slug}_{stamp()}.json"
    payload["receipt_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    return p


def zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def unpack_zip(archive: Path, rec: dict[str, Any], out_dir: Path, manifest_fh, hash_threshold: int, mode: str) -> dict[str, Any]:
    infos, blockers, encrypted = list_zip(archive)
    members_listed = len(infos)
    safe_infos = []
    path_traversal = symlinks = 0
    for info in infos:
        reason = unsafe_member(info.filename)
        if reason:
            path_traversal += 1
            continue
        if zip_symlink(info):
            symlinks += 1
            continue
        if info.is_dir():
            continue
        safe_infos.append(info)
    estimated = sum(int(i.file_size or 0) for i in safe_infos)
    ok_space, free = ensure_enough_space(out_dir.parent, estimated)
    if not ok_space:
        blockers.append(f"insufficient_disk_space:needed={estimated}:free={free}")
        return {"members_listed": members_listed, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": 0, "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": 0, "encrypted_or_passworded": encrypted, "blockers": blockers, "verdict": "BLOCKED"}
    if mode == "dry_run":
        return {"members_listed": members_listed, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": sum(1 for i in safe_infos if int(i.file_size or 0) > hash_threshold), "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": 0, "encrypted_or_passworded": encrypted, "blockers": blockers, "verdict": "PASS" if not blockers else "PARTIAL_FAIL"}
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = bytes_out = hashed = hash_skipped = corrupt = 0
    with zipfile.ZipFile(archive) as zf:
        for info in safe_infos:
            target = target_for(out_dir, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                blockers.append(f"existing_extracted_file_skipped:{rel(target)}")
                continue
            try:
                with zf.open(info, "r") as src:
                    digest, hash_status, written = sha256_stream(src, target, int(info.file_size or 0), hash_threshold)
            except RuntimeError as exc:
                corrupt += 1
                blockers.append(f"member_extract_failed:{info.filename}:{exc}")
                continue
            except Exception as exc:
                corrupt += 1
                blockers.append(f"member_extract_failed:{info.filename}:{type(exc).__name__}:{exc}")
                continue
            extracted += 1
            bytes_out += written
            if digest:
                hashed += 1
            else:
                hash_skipped += 1
            mtime = datetime(*info.date_time, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z") if info.date_time else None
            row = {
                "schema": "lucidota.krampuschewing.archive_unpack_manifest.row.v1",
                "archive_path": rel(archive),
                "archive_size": rec.get("size_bytes"),
                "archive_mtime": rec.get("modified_time_utc"),
                "member_path": info.filename,
                "extracted_path": rel(target),
                "member_size": int(info.file_size or 0),
                "member_mtime": mtime,
                "sha256": digest,
                "hash_status": hash_status,
                "kind_guess": kind_guess(info.filename),
                "lane_guess": lane_guess(info.filename),
                "time_guess": mtime or rec.get("modified_time_utc"),
                "blockers": [],
            }
            manifest_fh.write(json.dumps(row, sort_keys=False, ensure_ascii=False) + "\n")
    verdict = "PASS" if not blockers and corrupt == 0 else "PARTIAL_FAIL"
    return {"members_listed": members_listed, "members_extracted": extracted, "bytes_extracted": bytes_out, "members_hashed": hashed, "hash_skipped_large_members": hash_skipped, "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": corrupt, "encrypted_or_passworded": encrypted, "blockers": blockers[:200], "verdict": verdict}


def unpack_tar(archive: Path, rec: dict[str, Any], out_dir: Path, manifest_fh, hash_threshold: int, mode: str) -> dict[str, Any]:
    members, blockers, encrypted = list_tar(archive)
    members_listed = len(members)
    safe = []
    path_traversal = symlinks = 0
    for m in members:
        reason = unsafe_member(m.name)
        if reason:
            path_traversal += 1
            continue
        if m.issym() or m.islnk():
            symlinks += 1
            continue
        if not m.isfile():
            continue
        safe.append(m)
    estimated = sum(int(m.size or 0) for m in safe)
    ok_space, free = ensure_enough_space(out_dir.parent, estimated)
    if not ok_space:
        blockers.append(f"insufficient_disk_space:needed={estimated}:free={free}")
        return {"members_listed": members_listed, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": 0, "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": 0, "encrypted_or_passworded": encrypted, "blockers": blockers, "verdict": "BLOCKED"}
    if mode == "dry_run":
        return {"members_listed": members_listed, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": sum(1 for m in safe if int(m.size or 0) > hash_threshold), "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": 0, "encrypted_or_passworded": encrypted, "blockers": blockers, "verdict": "PASS" if not blockers else "PARTIAL_FAIL"}
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = bytes_out = hashed = hash_skipped = corrupt = 0
    with tarfile.open(archive) as tf:
        for m in safe:
            target = target_for(out_dir, m.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                blockers.append(f"existing_extracted_file_skipped:{rel(target)}")
                continue
            src = tf.extractfile(m)
            if src is None:
                corrupt += 1
                blockers.append(f"member_extract_failed:{m.name}:no_fileobj")
                continue
            try:
                digest, hash_status, written = sha256_stream(src, target, int(m.size or 0), hash_threshold)
            except Exception as exc:
                corrupt += 1
                blockers.append(f"member_extract_failed:{m.name}:{type(exc).__name__}:{exc}")
                continue
            extracted += 1
            bytes_out += written
            if digest:
                hashed += 1
            else:
                hash_skipped += 1
            mtime = utc_from_ts(m.mtime)
            row = {"schema": "lucidota.krampuschewing.archive_unpack_manifest.row.v1", "archive_path": rel(archive), "archive_size": rec.get("size_bytes"), "archive_mtime": rec.get("modified_time_utc"), "member_path": m.name, "extracted_path": rel(target), "member_size": int(m.size or 0), "member_mtime": mtime, "sha256": digest, "hash_status": hash_status, "kind_guess": kind_guess(m.name), "lane_guess": lane_guess(m.name), "time_guess": mtime or rec.get("modified_time_utc"), "blockers": []}
            manifest_fh.write(json.dumps(row, sort_keys=False, ensure_ascii=False) + "\n")
    verdict = "PASS" if not blockers and corrupt == 0 else "PARTIAL_FAIL"
    return {"members_listed": members_listed, "members_extracted": extracted, "bytes_extracted": bytes_out, "members_hashed": hashed, "hash_skipped_large_members": hash_skipped, "path_traversal_blocked": path_traversal, "symlinks_blocked": symlinks, "corrupt_members": corrupt, "encrypted_or_passworded": encrypted, "blockers": blockers[:200], "verdict": verdict}


def unpack_one(rec: dict[str, Any], out_root: Path, manifest_fh, hash_threshold: int, mode: str) -> tuple[dict[str, Any], Path]:
    archive = Path(str(rec.get("path") or ""))
    slug = safe_slug(str(rec.get("relative_path") or archive.name))
    out_dir = out_root / f"{slug}_{rec.get('size_bytes')}"
    base = {"schema": "lucidota.krampuschewing.archive_unpack_receipt.v1", "generated_at_utc": now(), "archive_path": rel(archive), "archive_relative_path": rec.get("relative_path"), "archive_size_bytes": rec.get("size_bytes"), "archive_class": rec.get("large_class"), "out_dir": rel(out_dir), "mode": mode, "manifest_path": None, "files_moved": [], "files_deleted": [], "original_archive_modified": False, "canonical_graph_writes": False, "blockers": []}
    if not archive.exists() or not archive.is_file():
        base.update({"verdict": "FAIL", "members_listed": 0, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": 0, "path_traversal_blocked": 0, "symlinks_blocked": 0, "corrupt_members": 0, "encrypted_or_passworded": False, "blockers": ["archive_missing_or_not_file"]})
        return base, write_receipt(slug, base)
    kind = archive_kind(archive)
    tool = tool_for(kind)
    if tool is None:
        base.update({"verdict": "BLOCKED", "members_listed": 0, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": 0, "path_traversal_blocked": 0, "symlinks_blocked": 0, "corrupt_members": 0, "encrypted_or_passworded": False, "blockers": [f"unsupported_or_missing_tool:{kind}"]})
        return base, write_receipt(slug, base)
    if kind == "zip":
        stats = unpack_zip(archive, rec, out_dir, manifest_fh, hash_threshold, mode)
    elif kind == "tar":
        stats = unpack_tar(archive, rec, out_dir, manifest_fh, hash_threshold, mode)
    else:
        stats = {"verdict": "BLOCKED", "members_listed": 0, "members_extracted": 0, "bytes_extracted": 0, "members_hashed": 0, "hash_skipped_large_members": 0, "path_traversal_blocked": 0, "symlinks_blocked": 0, "corrupt_members": 0, "encrypted_or_passworded": False, "blockers": [f"tool_not_integrated:{kind}:{tool}"]}
    base.update(stats)
    return base, write_receipt(slug, base)


def main() -> int:
    ap = argparse.ArgumentParser(description="Controlled KRAMPUSCHEWING archive unpack into staging")
    ap.add_argument("--large-file-validation", required=True)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--max-archives", type=int)
    ap.add_argument("--only")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    ap.add_argument("--hash-threshold-bytes", type=int, default=268_435_456)
    ap.add_argument("--skip-active-runtime-db", action="store_true", default=True)
    args = ap.parse_args()
    validation = Path(args.large_file_validation)
    if not validation.is_absolute():
        validation = ROOT / validation
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    mode_s = "execute" if args.execute else "dry_run"
    records = archive_records(validation, args.only)
    if args.max_archives is not None:
        records = records[: args.max_archives]
    OUT.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = OUT / f"archive_unpack_manifest_{stamp()}.jsonl"
    receipts: list[str] = []
    totals = Counter()
    blockers: list[str] = []
    archive_verdicts: Counter[str] = Counter()
    with manifest.open("w", encoding="utf-8") as mf:
        for rec in records:
            payload, receipt_path = unpack_one(rec, out_root, mf, args.hash_threshold_bytes, mode_s)
            payload["manifest_path"] = rel(manifest)
            # update receipt with manifest path after path known
            receipt_path.write_text(json.dumps({**payload, "receipt_path": rel(receipt_path)}, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
            receipts.append(rel(receipt_path))
            archive_verdicts[str(payload.get("verdict"))] += 1
            totals["members_extracted"] += int(payload.get("members_extracted") or 0)
            totals["bytes_extracted"] += int(payload.get("bytes_extracted") or 0)
            totals["archives_unpacked"] += 1 if payload.get("members_extracted") else 0
            if payload.get("verdict") == "BLOCKED":
                totals["archives_blocked"] += 1
            if payload.get("verdict") == "PARTIAL_FAIL":
                totals["archives_partial"] += 1
            blockers.extend(str(x) for x in payload.get("blockers") or [])
    verdict = "PASS"
    if totals["archives_blocked"] or totals["archives_partial"] or blockers:
        verdict = "PARTIAL_FAIL"
    if records and totals["members_extracted"] == 0 and args.execute:
        verdict = "BLOCKED" if totals["archives_blocked"] else "FAIL"
    summary = {"schema": "lucidota.krampuschewing.archive_unpack_summary.v1", "generated_at_utc": now(), "mode": mode_s, "verdict": verdict, "archives_considered": len(records), "archives_unpacked": int(totals["archives_unpacked"]), "archives_blocked": int(totals["archives_blocked"]), "archives_partial": int(totals["archives_partial"]), "archive_verdicts": dict(sorted(archive_verdicts.items())), "members_extracted": int(totals["members_extracted"]), "bytes_extracted": int(totals["bytes_extracted"]), "output_root": rel(out_root), "manifest_path": rel(manifest), "archive_receipts": receipts, "canonical_graph_writes": False, "files_moved": [], "files_deleted": [], "original_archives_modified": False, "blockers": sorted(set(blockers))[:500]}
    summary_path = OUT / f"archive_unpack_summary_{stamp()}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print("ARCHIVE_UNPACK_SUMMARY=" + rel(summary_path))
    print("ARCHIVE_UNPACK_MANIFEST=" + rel(manifest))
    print("ARCHIVE_UNPACK=" + verdict)
    print("ARCHIVES_CONSIDERED=" + str(summary["archives_considered"]))
    print("ARCHIVES_UNPACKED=" + str(summary["archives_unpacked"]))
    print("ARCHIVES_BLOCKED=" + str(summary["archives_blocked"]))
    print("MEMBERS_EXTRACTED=" + str(summary["members_extracted"]))
    print("BYTES_EXTRACTED=" + str(summary["bytes_extracted"]))
    return 0 if verdict == "PASS" else 2 if verdict == "PARTIAL_FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
