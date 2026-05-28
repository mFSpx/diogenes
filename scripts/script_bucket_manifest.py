#!/usr/bin/env python3
"""Classify LUCIDOTA scripts without moving or deleting anything.

Buckets:
- KEEP_ACTIVE: entrypoint/check/workflow script referenced by tests, schemas, CI, or docs.
- KEEP_LIBRARY: imported helper/library script with no strong standalone-entrypoint signal.
- MERGE_DUPLICATE: duplicate basename/pattern where another copy is more canonical.
- QUARANTINE_LEGACY: legacy/archive path; keep for proof hoard but do not route as active.
- DELETE_CANDIDATE_NEEDS_HUMAN: generated/cache/broken-junk candidate only; never deleted here.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "script_buckets"
BUCKETS = [
    "KEEP_ACTIVE",
    "KEEP_LIBRARY",
    "MERGE_DUPLICATE",
    "QUARANTINE_LEGACY",
    "DELETE_CANDIDATE_NEEDS_HUMAN",
]
ACTIVE_NAME_RE = re.compile(
    r"(gate|smoke|check|cli|worker|watch|watcher|router|queue|pipeline|ingest|export|import|runner|orchestrator|service|status|ledger|manifest)",
    re.I,
)
REFERENCE_ROOTS = [
    ".github",
    "00_PROJECT_BRAIN",
    "06_SCHEMA",
    "tests",
    "scripts",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def iter_script_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_file() and path.suffix in {".py", ".sh"}:
            files.append(path)
    return files


def load_tickletrunk() -> dict[str, dict[str, Any]]:
    path = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    toolboxes = data.get("toolboxes") or {}
    groups = toolboxes.values() if isinstance(toolboxes, dict) else toolboxes
    for toolbox in groups:
        items = toolbox.get("items") if isinstance(toolbox, dict) else toolbox
        for item in items or []:
            rp = item.get("relative_path")
            if isinstance(rp, str):
                out[rp] = item
    return out


def reference_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    searchable: list[Path] = []
    for root in REFERENCE_ROOTS:
        p = ROOT / root
        if not p.exists():
            continue
        if p.is_file():
            searchable.append(p)
            continue
        for child in p.rglob("*"):
            if child.is_file() and child.suffix.lower() in {".py", ".sh", ".md", ".json", ".yml", ".yaml", ".sql", ".toml"}:
                if "__pycache__" not in child.parts:
                    searchable.append(child)
    scripts = [rel(p) for p in iter_script_files(ROOT / "scripts")]
    basenames = {Path(s).name: s for s in scripts}
    for path in searchable:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for script in scripts:
            if script in text:
                counts[script] += 1
        for name, script in basenames.items():
            if name in text:
                counts[script] += 1
    return counts


def has_main_guard(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "__main__" in text or path.suffix == ".sh"


def classify(path: Path, *, refs: Counter[str], duplicates: dict[str, list[Path]], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rpath = rel(path)
    name = path.name
    reasons: list[str] = []
    bucket = "KEEP_LIBRARY"
    if path.suffix not in {".py", ".sh"} or "__pycache__" in path.parts or name.endswith((".pyc", ".pyo")):
        bucket = "DELETE_CANDIDATE_NEEDS_HUMAN"
        reasons.append("generated_or_non_source_script_candidate")
    elif "legacy" in path.parts:
        bucket = "QUARANTINE_LEGACY"
        reasons.append("under_scripts_legacy")
    duplicate_peers = [p for p in duplicates[name] if p != path]
    non_legacy_peers = [p for p in duplicate_peers if "legacy" not in p.parts]
    if duplicate_peers and not non_legacy_peers and bucket not in {"DELETE_CANDIDATE_NEEDS_HUMAN", "QUARANTINE_LEGACY"}:
        reasons.append("legacy_duplicate_counterpart_quarantined:" + ",".join(rel(p) for p in duplicate_peers)[:240])
    if non_legacy_peers and bucket not in {"DELETE_CANDIDATE_NEEDS_HUMAN", "QUARANTINE_LEGACY"}:
        bucket = "MERGE_DUPLICATE"
        reasons.append("duplicate_active_basename_present:" + ",".join(rel(p) for p in non_legacy_peers)[:240])
    ref_count = int(refs.get(rpath, 0))
    if bucket not in {"DELETE_CANDIDATE_NEEDS_HUMAN", "QUARANTINE_LEGACY", "MERGE_DUPLICATE"}:
        if ref_count > 0 or ACTIVE_NAME_RE.search(name):
            bucket = "KEEP_ACTIVE"
            reasons.append("active_reference_or_entrypoint_name")
        elif not has_main_guard(path):
            bucket = "KEEP_LIBRARY"
            reasons.append("importable_helper_no_main_guard")
    if not reasons:
        reasons.append("default_library_keep")
    item = manifest.get(rpath) or {}
    return {
        "path": rpath,
        "bucket": bucket,
        "reasons": reasons,
        "reference_count": ref_count,
        "has_main_guard": has_main_guard(path),
        "manifest_status": item.get("status"),
        "manifest_role": item.get("proof_hoard_role"),
        "size_bytes": path.stat().st_size,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LUCIDOTA script bucket manifest",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "No files were moved, renamed, or deleted. DELETE_CANDIDATE_NEEDS_HUMAN is review-only.",
        "",
        "## Counts",
        "",
    ]
    for bucket in BUCKETS:
        lines.append(f"- {bucket}: {payload['counts'].get(bucket, 0)}")
    lines.extend(["", "## Buckets", ""])
    for bucket in BUCKETS:
        lines.extend([f"### {bucket}", ""])
        for item in payload["items"]:
            if item["bucket"] == bucket:
                reasons = "; ".join(item["reasons"])
                lines.append(f"- `{item['path']}` — refs={item['reference_count']} — {reasons}")
        lines.append("")
    return "\n".join(lines)


def build_manifest(scripts_dir: Path) -> dict[str, Any]:
    files = iter_script_files(scripts_dir)
    dups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        dups[path.name].append(path)
    manifest = load_tickletrunk()
    refs = reference_counts()
    items = [classify(path, refs=refs, duplicates=dups, manifest=manifest) for path in files]
    counts = Counter(item["bucket"] for item in items)
    return {
        "schema": "lucidota.script_bucket_manifest.v1",
        "generated_at": now(),
        "scripts_dir": rel(scripts_dir),
        "non_destructive": True,
        "counts": {bucket: int(counts.get(bucket, 0)) for bucket in BUCKETS},
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a non-destructive KEEP/MERGE/QUARANTINE script bucket manifest.")
    parser.add_argument("--scripts-dir", default=str(ROOT / "scripts"))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    scripts_dir = Path(args.scripts_dir)
    if not scripts_dir.is_absolute():
        scripts_dir = ROOT / scripts_dir
    payload = build_manifest(scripts_dir)
    if not args.no_write:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"script_bucket_manifest_{stamp()}"
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        payload["report_path"] = rel(json_path)
        payload["markdown_path"] = rel(md_path)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        print("REPORT_PATH=" + rel(json_path))
        print("MARKDOWN_PATH=" + rel(md_path))
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("SCRIPT_BUCKET_MANIFEST=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
