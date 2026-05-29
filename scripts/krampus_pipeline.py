#!/usr/bin/env python3
"""One-pass KRAMPUSCHEWING lane planner and governed launcher.

This module reads the existing inventory JSONL once, classifies rows into the
four requested lanes, writes durable manifests/receipts, and can launch each
lane through the resource governor.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.spine_common import append_jsonl, receipt, rel, sha256_json
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spine_common import append_jsonl, receipt, rel, sha256_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "runtime"
DEFAULT_INVENTORY = ROOT / "05_OUTPUTS" / "krampus_inventory" / "krampus_queue_eligible.jsonl"
DEFAULT_CORPUS_MAP = ROOT / "05_OUTPUTS" / "goals" / "corpus_map_20260528T091724121300Z.json"
DEFAULT_SKIP_MANIFEST = OUT / "krampus_skip_manifest.jsonl"
TEXT_SUFFIXES = {".md", ".txt", ".log", ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".xml", ".html", ".htm", ".sql", ".ini", ".conf"}
OCR_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif", ".docx", ".pptx", ".xlsx", ".eml", ".odt", ".rtf"}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".tar", ".gz", ".bz2", ".xz", ".rar", ".iso", ".tgz"}
DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".duckdb", ".mdb", ".db-journal", ".ldb"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
HEAVY_SKIP_SUFFIXES = ARCHIVE_SUFFIXES | DATABASE_SUFFIXES | VIDEO_SUFFIXES
DEFAULT_TEXT_LIMIT = 64 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inventory_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def classify_row(row: dict[str, Any], *, text_limit: int, huge_threshold: int) -> tuple[str, str]:
    size = int(row.get("size_bytes") or 0)
    suffix = str(row.get("suffix") or "").lower()
    if size > huge_threshold or suffix in HEAVY_SKIP_SUFFIXES:
        return "lane4_skip", "heavy_archive_or_over_100mb"
    if suffix in TEXT_SUFFIXES and size <= text_limit:
        return "lane1_text", "deterministic_text"
    if suffix in OCR_SUFFIXES or suffix not in TEXT_SUFFIXES:
        return "lane2_groq", "groq_ocr_or_inference"
    return "lane2_groq", "groq_ocr_or_inference"


def build_pipeline_plan(inventory_jsonl: Path | str, *, corpus_map: dict[str, Any] | None = None) -> dict[str, Any]:
    inventory_path = Path(inventory_jsonl)
    if not inventory_path.is_absolute():
        inventory_path = ROOT / inventory_path
    corpus = corpus_map or load_json(DEFAULT_CORPUS_MAP)
    huge_threshold = int((corpus.get("recommendations") or {}).get("huge_file_threshold_bytes", 104857600))
    text_limit = DEFAULT_TEXT_LIMIT
    rows = load_inventory_rows(inventory_path)

    buckets: dict[str, list[dict[str, Any]]] = {"lane1_text": [], "lane2_groq": [], "lane4_skip": []}
    skips: list[dict[str, Any]] = []
    for row in rows:
        lane, reason = classify_row(row, text_limit=text_limit, huge_threshold=huge_threshold)
        item = dict(row)
        item["lane"] = lane
        item["reason"] = reason
        if lane == "lane4_skip":
            skips.append(
                {
                    "path": row.get("path"),
                    "size_bytes": row.get("size_bytes"),
                    "suffix": row.get("suffix"),
                    "reason": reason,
                    "sha256": row.get("sha256"),
                }
            )
        else:
            buckets[lane].append(item)

    lane1_count = len(buckets["lane1_text"])
    lane2_count = len(buckets["lane2_groq"])
    lane4_count = len(skips)
    plan = {
        "schema": "lucidota.krampus_pipeline.plan.v1",
        "generated_at": now(),
        "inventory_path": rel(inventory_path),
        "corpus_map_path": rel(DEFAULT_CORPUS_MAP),
        "text_limit_bytes": text_limit,
        "huge_threshold_bytes": huge_threshold,
        "lane_counts": {
            "lane1_text": lane1_count,
            "lane2_groq": lane2_count,
            "lane3_registration": lane1_count + lane2_count,
            "lane4_skips": lane4_count,
        },
        "lane_bytes": {
            "lane1_text": sum(int(r.get("size_bytes") or 0) for r in buckets["lane1_text"]),
            "lane2_groq": sum(int(r.get("size_bytes") or 0) for r in buckets["lane2_groq"]),
            "lane4_skips": sum(int(r.get("size_bytes") or 0) for r in skips),
        },
        "lane_rows": buckets,
        "skip_manifest_rows": skips,
        "skip_manifest_path": rel(DEFAULT_SKIP_MANIFEST),
    }
    plan["plan_sha256"] = sha256_json({"inventory_path": plan["inventory_path"], "lane_counts": plan["lane_counts"], "lane_bytes": plan["lane_bytes"], "huge_threshold_bytes": plan["huge_threshold_bytes"]})
    return plan


def summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": plan["schema"],
        "generated_at": plan["generated_at"],
        "inventory_path": plan["inventory_path"],
        "corpus_map_path": plan["corpus_map_path"],
        "text_limit_bytes": plan["text_limit_bytes"],
        "huge_threshold_bytes": plan["huge_threshold_bytes"],
        "lane_counts": plan["lane_counts"],
        "lane_bytes": plan["lane_bytes"],
        "skip_manifest_path": plan["skip_manifest_path"],
        "plan_sha256": plan["plan_sha256"],
        "skip_manifest_count": len(plan["skip_manifest_rows"]),
    }


def write_skip_manifest(rows: list[dict[str, Any]], path: Path = DEFAULT_SKIP_MANIFEST) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    for row in rows:
        append_jsonl(path, row)
    return path


def lane_script(lane: str) -> str:
    return f"scripts/krampus_pipeline.py --lane {lane}"


def build_lane_commands(plan: dict[str, Any], *, root: Path = ROOT) -> list[list[str]]:
    py = sys.executable
    me = str(Path(__file__).resolve())
    common = [
        py,
        str(root / "scripts" / "resource_governor.py"),
        "spawn",
        "--owner",
        "codex",
        "--requested-workers",
        "1",
        "--max-workers",
        "1",
        "--max-memory-mb",
        "1024",
        "--max-cpu-percent",
        "100",
        "--execute",
        "--wait",
        "--timeout",
        "300",
    ]
    commands = []
    for lane in ("lane1", "lane2", "lane3", "lane4"):
        purpose = f"krampuschewing_{lane}"
        commands.append(
            common
            + [
                "--purpose",
                purpose,
                "--",
                py,
                me,
                "--lane",
                lane,
                "--manifest",
                plan.get("skip_manifest_path") if lane == "lane4" else rel(root / "05_OUTPUTS" / "runtime" / f"krampus_{lane}.jsonl"),
                "--inventory-jsonl",
                plan["inventory_path"],
                "--corpus-map",
                plan["corpus_map_path"],
            ]
        )
    return commands


def lane_result(lane: str, manifest: Path, rows: list[dict[str, Any]], *, execute: bool) -> dict[str, Any]:
    receipt_payload = {
        "schema": "lucidota.krampus_pipeline.lane_result.v1",
        "generated_at": now(),
        "lane": lane,
        "execute_performed": execute,
        "manifest_path": rel(manifest),
        "row_count": len(rows),
    }
    receipt_payload["receipt_fingerprint"] = sha256_json({"lane": lane, "manifest_path": receipt_payload["manifest_path"], "row_count": len(rows)})
    return receipt_payload


def run_plan(args: argparse.Namespace) -> int:
    plan = build_pipeline_plan(args.inventory_jsonl, corpus_map=load_json(Path(args.corpus_map)))
    write_skip_manifest(plan["skip_manifest_rows"], Path(args.skip_manifest) if args.skip_manifest else DEFAULT_SKIP_MANIFEST)
    receipt("krampus_pipeline_plan", summarize_plan(plan), root="05_OUTPUTS/runtime")
    if args.json:
        print(json.dumps(summarize_plan(plan), sort_keys=True))
    print("KRAMPUS_PIPELINE=PLAN")
    return 0


def run_lane(args: argparse.Namespace) -> int:
    plan = build_pipeline_plan(args.inventory_jsonl, corpus_map=load_json(Path(args.corpus_map)))
    lane = args.lane
    if lane == "lane1":
        rows = plan["lane_rows"]["lane1_text"]
    elif lane == "lane2":
        rows = plan["lane_rows"]["lane2_groq"]
    elif lane == "lane3":
        rows = plan["lane_rows"]["lane1_text"] + plan["lane_rows"]["lane2_groq"]
    else:
        rows = plan["skip_manifest_rows"]
    manifest = Path(args.manifest) if args.manifest else ROOT / "05_OUTPUTS" / "runtime" / f"krampus_{lane}.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    payload = lane_result(lane, manifest, rows, execute=args.execute)
    receipt(f"krampus_pipeline_{lane}", payload, root="05_OUTPUTS/runtime")
    print(f"LANE={lane}")
    print(f"ROW_COUNT={len(rows)}")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    return 0


def run_all(args: argparse.Namespace) -> int:
    plan = build_pipeline_plan(args.inventory_jsonl, corpus_map=load_json(Path(args.corpus_map)))
    write_skip_manifest(plan["skip_manifest_rows"], Path(args.skip_manifest) if args.skip_manifest else DEFAULT_SKIP_MANIFEST)
    receipt("krampus_pipeline_plan", summarize_plan(plan), root="05_OUTPUTS/runtime")
    commands = build_lane_commands(plan, root=ROOT)
    result_rows: list[dict[str, Any]] = []
    if args.execute:
        procs = [subprocess.Popen(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for cmd in commands]
        for lane, proc in zip(("lane1", "lane2", "lane3", "lane4"), procs, strict=True):
            stdout, stderr = proc.communicate()
            result_rows.append({"lane": lane, "returncode": proc.returncode, "stdout_tail": stdout[-1200:], "stderr_tail": stderr[-1200:]})
    else:
        result_rows = [{"lane": lane, "returncode": None, "planned": True} for lane in ("lane1", "lane2", "lane3", "lane4")]
    final = {
        "schema": "lucidota.krampus_pipeline.run.v1",
        "generated_at": now(),
        "execute_performed": bool(args.execute),
        "plan_sha256": plan["plan_sha256"],
        "lane_counts": plan["lane_counts"],
        "skip_manifest_path": plan["skip_manifest_path"],
        "lane_runs": result_rows,
    }
    receipt("krampus_pipeline_run", final, root="05_OUTPUTS/runtime")
    if args.json:
        print(json.dumps(final, sort_keys=True))
    print("KRAMPUS_PIPELINE=PASS")
    return 0


def write_blocked_receipt(*, reason: str, args: argparse.Namespace) -> Path:
    payload = {
        "schema": "lucidota.krampus_pipeline.blocked_unauthorized.v1",
        "verdict": "BLOCKED",
        "reason": reason,
        "canonical_graph_writes": False,
        "db_writes_performed": False,
        "external_effects": False,
        "safe_next_action": "return_to_existing_inventory_and_absurd_flows_authority",
        "inventory_jsonl": args.inventory_jsonl,
        "corpus_map": args.corpus_map,
        "skip_manifest": args.skip_manifest or rel(DEFAULT_SKIP_MANIFEST),
        "allow_experimental_unauthorized_pipeline": bool(args.allow_experimental_unauthorized_pipeline),
    }
    return receipt("krampus_pipeline_blocked", payload, root="05_OUTPUTS/runtime")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-jsonl", default=str(DEFAULT_INVENTORY))
    ap.add_argument("--corpus-map", default=str(DEFAULT_CORPUS_MAP))
    ap.add_argument("--skip-manifest", default="")
    ap.add_argument("--lane", choices=["lane1", "lane2", "lane3", "lane4"], default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--allow-experimental-unauthorized-pipeline", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.allow_experimental_unauthorized_pipeline:
        write_blocked_receipt(reason="operator_did_not_authorize_four_lane_plan", args=args)
        print("KRAMPUS_PIPELINE=BLOCKED_UNAUTHORIZED_ARCHITECTURE")
        return 3
    if args.lane:
        return run_lane(args)
    if args.execute:
        return run_all(args)
    return run_plan(args)


if __name__ == "__main__":
    raise SystemExit(main())
