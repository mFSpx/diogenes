#!/usr/bin/env python3
"""Mirror LUCIDOTA wiring into a repo manifest and seed the script registry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[0]
for p in (str(SCRIPT_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spine_common import now, receipt, rel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
OUTPUT_ROOT = ROOT / "05_OUTPUTS" / "diogenes"
MAX_FILE_BYTES = 2_000_000
INCLUDE_EXTS = {".py", ".sh", ".sql", ".md", ".json", ".yaml", ".yml", ".toml", ".rs", ".txt", ".pyi"}
EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules", "target", "dist", "build", "coverage", "05_OUTPUTS", "04_RUNTIME",
}
EXCLUDE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz", ".xz", ".bz2",
    ".7z", ".tar", ".tgz", ".parquet", ".sqlite", ".db", ".bin", ".so", ".dll", ".dylib",
    ".pt", ".pth", ".ckpt", ".onnx", ".npy", ".npz", ".csv", ".tsv", ".avro", ".arrow",
}
CORPSE_HINTS = ("legacy", "corpse", "quarantine", "deleted", "obsolete")


@dataclass(frozen=True)
class MirrorFile:
    path: Path
    sha256: str
    kind: str
    status: str
    subsystem: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    tests: list[str]
    receipts: list[str]
    promotion_route: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded(path: Path) -> str | None:
    parts = set(path.parts)
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return f"dir:{part}"
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return f"suffix:{path.suffix.lower()}"
    try:
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            return f"size_gt:{MAX_FILE_BYTES}"
    except FileNotFoundError:
        return "missing"
    return None


def classify(path: Path) -> tuple[str, str]:
    lower = str(path).lower()
    if any(h in lower for h in CORPSE_HINTS):
        return "quarantine", "quarantine"
    if path.suffix.lower() == ".sql" or "/06_schema/" in lower:
        return "db-native", "active"
    if path.suffix.lower() in {".rs", ".toml"} or "/rust/" in lower:
        return "runtime-core", "active"
    if path.suffix.lower() in {".py", ".sh"}:
        return "experimental", "prototype"
    return "experimental", "prototype"


def infer_purpose(path: Path) -> str:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:8]
    except Exception:
        head = []
    for line in head:
        line = line.strip()
        if line.startswith('"""') and len(line) > 3:
            return line.strip('"').strip("'")
        if line.startswith("-- PURPOSE:"):
            return line.split(":", 1)[1].strip()
        if line.startswith("# ") and len(line) > 2:
            return line[2:].strip()
    return path.name.replace("_", " ")


def build_manifest(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[MirrorFile]]:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    registry: list[MirrorFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        reason = is_excluded(path)
        rel_path = path.relative_to(root)
        if reason:
            excluded.append({"path": str(rel_path), "reason": reason})
            continue
        if path.suffix.lower() in INCLUDE_EXTS:
            kind, status = classify(rel_path)
            record = {
                "path": str(rel_path),
                "sha256": sha256_file(path),
                "kind": kind,
                "status": status,
                "size_bytes": path.stat().st_size,
            }
            included.append(record)
            if str(rel_path).startswith(("scripts/", "ALGOS/", "06_SCHEMA/", "src/", "01_REPOS/PocketFlow/pocketflow/")):
                registry.append(
                    MirrorFile(
                        path=rel_path,
                        sha256=record["sha256"],
                        kind=kind,
                        status=status,
                        subsystem=str(rel_path).split("/", 1)[0],
                        purpose=infer_purpose(path),
                        inputs=["filesystem"],
                        outputs=["receipt", "registry"],
                        tests=[],
                        receipts=[],
                        promotion_route="tickletrunk" if kind == "experimental" else ("db" if kind == "db-native" else "runtime-core"),
                    )
                )
        else:
            excluded.append({"path": str(rel_path), "reason": f"ext:{path.suffix.lower() or 'none'}"})
    return included, excluded, registry


def sample_telemetry() -> dict[str, Any]:
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "mem_percent": mem.percent,
        "mem_available": mem.available,
        "swap_percent": psutil.swap_memory().percent,
        "disk_percent": disk.percent,
        "disk_free": disk.free,
        "loadavg": list(load),
    }


def mirror_kind(files: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"db-native": 0, "runtime-core": 0, "experimental": 0, "quarantine": 0}
    for row in files:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    return counts


def upsert_registry(conn: psycopg.Connection, rows: list[MirrorFile]) -> int:
    sql = """
    INSERT INTO lucidota_control.script_registry
    (script_path, sha256, purpose, inputs, outputs, subsystem, kind, status, tests, receipts, promotion_route, updated_at)
    VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb,%s::jsonb,%s,now())
    ON CONFLICT (script_path) DO UPDATE SET
        sha256=EXCLUDED.sha256,
        purpose=EXCLUDED.purpose,
        inputs=EXCLUDED.inputs,
        outputs=EXCLUDED.outputs,
        subsystem=EXCLUDED.subsystem,
        kind=EXCLUDED.kind,
        status=EXCLUDED.status,
        tests=EXCLUDED.tests,
        receipts=EXCLUDED.receipts,
        promotion_route=EXCLUDED.promotion_route,
        updated_at=now()
    """
    n = 0
    for row in rows:
        conn.execute(sql, (
            str(row.path), row.sha256, row.purpose, json.dumps(row.inputs), json.dumps(row.outputs),
            row.subsystem, row.kind, row.status, json.dumps(row.tests), json.dumps(row.receipts), row.promotion_route,
        ))
        n += 1
    return n


def write_snapshot(conn: psycopg.Connection, *, root: Path, manifest_path: Path, included: int, excluded: int, registry_rows: int, telemetry: dict[str, Any], dry_run: bool) -> str:
    if dry_run:
        return "dry-run"
    row = conn.execute(
        """
        INSERT INTO lucidota_control.project_mirror_snapshot
        (root_path, manifest_path, included_count, excluded_count, registry_rows, exclusion_rules, telemetry, note)
        VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
        RETURNING snapshot_uuid::text
        """,
        (
            str(root), rel(manifest_path), included, excluded, registry_rows,
            json.dumps([
                f"excluded_dir:{sorted(EXCLUDE_DIRS)}",
                f"excluded_suffix:{sorted(EXCLUDE_SUFFIXES)}",
                f"max_bytes:{MAX_FILE_BYTES}",
            ]),
            json.dumps(telemetry),
            "project wiring mirror; bulk data/logs/binaries excluded by design",
        ),
    ).fetchone()
    return str(row[0]) if row else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--db-url", default=DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", default=str(OUTPUT_ROOT / f"diogenes_mirror_{now().replace(':','').replace('-','')}.json"))
    ap.add_argument("--limit", type=int, default=0, help="reserved; 0 means scan everything")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    included, excluded, registry = build_manifest(root)
    telemetry = sample_telemetry()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / f"mirror_manifest_{now().replace(':','').replace('-','')}.json"
    payload = {
        "schema": "lucidota.diogenes_mirror.v1",
        "generated_at": now(),
        "root": str(root),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "kind_counts": mirror_kind(included),
        "registry_candidates": len(registry),
        "telemetry": telemetry,
        "included_sample": included[:50],
        "excluded_sample": excluded[:50],
        "manifest_path": rel(manifest_path),
        "dry_run": args.dry_run,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    db_result = {"registry_rows": 0, "snapshot_uuid": "dry-run"}
    if not args.dry_run:
        with psycopg.connect(args.db_url) as conn:
            registry_rows = upsert_registry(conn, registry)
            snapshot_uuid = write_snapshot(conn, root=root, manifest_path=manifest_path, included=len(included), excluded=len(excluded), registry_rows=registry_rows, telemetry=telemetry, dry_run=False)
            conn.execute(
                "INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail) VALUES (%s,%s,'mirror','succeeded','diogenes_mirror',%s::jsonb)",
                ( "diogenes-mirror", manifest_path.stem, json.dumps({"included_count": len(included), "excluded_count": len(excluded), "registry_rows": registry_rows, "snapshot_uuid": snapshot_uuid}) ),
            )
            conn.execute(
                "INSERT INTO lucidota_learning.bytewax_abductive_event(source, source_ref, text_surface, ontology_terms, epistemic_flag, injection_flag, compressed_activity, certainty_trace, payload) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb) ON CONFLICT (source, source_ref) DO UPDATE SET text_surface=EXCLUDED.text_surface, ontology_terms=EXCLUDED.ontology_terms, epistemic_flag=EXCLUDED.epistemic_flag, injection_flag=EXCLUDED.injection_flag, compressed_activity=EXCLUDED.compressed_activity, certainty_trace=EXCLUDED.certainty_trace, payload=EXCLUDED.payload",
                ("diogenes_mirror", manifest_path.stem, "Project mirror snapshot", json.dumps(["backup", "wiring", "registry"]), "FACT", False, json.dumps({"root": str(root)}), json.dumps({"included": len(included), "excluded": len(excluded)}), json.dumps(payload)),
            )
            conn.execute(
                "INSERT INTO lucidota_learning.river_run(status, events_seen, examples_trained, detail) VALUES ('succeeded', %s, %s, %s::jsonb)",
                (1, 1, json.dumps({"mode": "mirror", "snapshot_uuid": snapshot_uuid, "registry_rows": registry_rows, "telemetry": telemetry})),
            )
            conn.execute(
                """
                INSERT INTO lucidota_learning.river_score(source, phase, decision, examples, successes, failures, success_rate, river_prediction, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (source, phase, decision) DO UPDATE SET
                    examples=lucidota_learning.river_score.examples+EXCLUDED.examples,
                    successes=lucidota_learning.river_score.successes+EXCLUDED.successes,
                    failures=lucidota_learning.river_score.failures+EXCLUDED.failures,
                    success_rate=(lucidota_learning.river_score.successes+EXCLUDED.successes)::float/GREATEST(lucidota_learning.river_score.examples+EXCLUDED.examples,1),
                    river_prediction=EXCLUDED.river_prediction,
                    updated_at=now()
                """,
                ("diogenes_mirror", "mirror", "mirror_complete", 1, 1, 0, 1.0, 1.0),
            )
            conn.commit()
            db_result = {"registry_rows": registry_rows, "snapshot_uuid": snapshot_uuid}
    receipt_payload = {
        "schema": "lucidota.diogenes_mirror.receipt.v1",
        "generated_at": now(),
        "root": str(root),
        "db_url": args.db_url,
        "dry_run": args.dry_run,
        "manifest_path": rel(manifest_path),
        "included_count": len(included),
        "excluded_count": len(excluded),
        "registry_candidates": len(registry),
        "db_result": db_result,
        "kind_counts": mirror_kind(included),
        "telemetry": telemetry,
        "github_status": "unauthenticated",
        "excluded_sample": excluded[:20],
        "included_sample": included[:20],
    }
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt("diogenes_mirror", receipt_payload, root="05_OUTPUTS/diogenes")
    print(json.dumps(receipt_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
