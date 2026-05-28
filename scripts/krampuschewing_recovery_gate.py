#!/usr/bin/env python3
"""Run non-mutating KRAMPUSCHEWING recovery engine outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "krampuschewing"
SERVICE = Path("/home/mfspx/.config/systemd/user/krampuschewing.service")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def run(cmd: list[str], timeout: int | None = None) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
    paths: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            if key.endswith("PATH") or key in {"REPORT_PATH", "INDEX_PATH", "SUMMARY_PATH"}:
                paths[key] = val.strip()
    return {
        "command": " ".join(cmd),
        "rc": proc.returncode,
        "paths": paths,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def count_lines(path: str | None) -> int:
    if not path:
        return 0
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def load_json_rel(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="KRAMPUSCHEWING recovery gate: spine, index, graph candidates, River candidates")
    ap.add_argument("--root", default="KRAMPUSCHEWING")
    ap.add_argument("--max-files", type=int, default=100_000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    receipts: list[str] = []
    blockers: list[str] = []

    spine = run([sys.executable, "scripts/krampuschewing_role_discovery.py", "--root", args.root], timeout=60)
    commands.append(spine)
    spine_path = spine["paths"].get("REPORT_PATH")
    if spine_path:
        receipts.append(spine_path)
    if spine["rc"] != 0:
        blockers.append("spine_discovery_failed")

    master = run([sys.executable, "scripts/krampuschewing_master_index.py", "--root", args.root, "--max-files", str(args.max_files)], timeout=None)
    commands.append(master)
    index_path = master["paths"].get("INDEX_PATH")
    summary_path = master["paths"].get("SUMMARY_PATH")
    if index_path:
        receipts.append(index_path)
    if summary_path:
        receipts.append(summary_path)
    if master["rc"] != 0:
        blockers.append("master_index_failed")

    graph = {"command": "SKIPPED graph stage", "rc": 99, "paths": {}, "stdout_tail": "", "stderr_tail": "missing index"}
    river = {"command": "SKIPPED river rows", "rc": 99, "paths": {}, "stdout_tail": "", "stderr_tail": "missing index"}
    if index_path:
        graph = run([sys.executable, "scripts/krampuschewing_graph_stage.py", "--index", index_path], timeout=None)
        commands.append(graph)
        graph_path = graph["paths"].get("GRAPH_CANDIDATES_PATH")
        if graph_path:
            receipts.append(graph_path)
        if graph["rc"] != 0:
            blockers.append("graph_stage_failed")

        river = run([sys.executable, "scripts/krampuschewing_river_rows.py", "--index", index_path], timeout=None)
        commands.append(river)
        river_path = river["paths"].get("RIVER_TRAINING_CANDIDATES_PATH")
        if river_path:
            receipts.append(river_path)
        if river["rc"] != 0:
            blockers.append("river_rows_failed")
    else:
        commands.extend([graph, river])
        blockers.append("index_path_missing")

    summary = load_json_rel(summary_path)
    graph_path = graph.get("paths", {}).get("GRAPH_CANDIDATES_PATH")
    river_path = river.get("paths", {}).get("RIVER_TRAINING_CANDIDATES_PATH")
    files_indexed = int(summary.get("files_indexed") or 0)
    files_seen = int(summary.get("files_seen") or 0)
    hash_skipped_large_files = int(summary.get("hash_skipped_large_files") or 0)
    if hash_skipped_large_files:
        blockers.append(f"master_index_large_file_sha256_skips:{hash_skipped_large_files}")
    verdict = "PASS" if not blockers and files_indexed > 0 and count_lines(graph_path) > 0 and count_lines(river_path) > 0 else "PARTIAL_FAIL"
    receipt = {
        "schema": "lucidota.krampuschewing.recovery_receipt.v1",
        "generated_at_utc": now(),
        "verdict": verdict,
        "root": str((ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root).resolve()),
        "appears_to_be": "legacy dev/case/file/proof/graph/River organizer",
        "files_seen": files_seen,
        "files_indexed": files_indexed,
        "bytes_total": summary.get("bytes_total", 0),
        "duplicates": summary.get("duplicates", 0),
        "hash_skipped_large_files": hash_skipped_large_files,
        "max_hash_bytes": summary.get("max_hash_bytes"),
        "by_lane_guess": summary.get("by_lane_guess", {}),
        "by_kind_guess": summary.get("by_kind_guess", {}),
        "master_index_path": index_path,
        "master_summary_path": summary_path,
        "spine_discovery_path": spine_path,
        "graph_candidates": count_lines(graph_path),
        "graph_candidates_path": graph_path,
        "river_training_candidates": count_lines(river_path),
        "river_training_candidates_path": river_path,
        "service_found": SERVICE.exists(),
        "service_run_or_restarted": False,
        "files_moved": 0,
        "files_deleted": 0,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "river_training_performed": False,
        "receipts_written": receipts,
        "commands_run": commands,
        "blockers": blockers,
        "next_smallest_safe_work": "Patch krampuschewing_watcher.sh / korpus_krampii.py into explicit safe mode: storage-only or graph-promotion-gated ingest, no move-to-DIGESTED until a chew receipt validates.",
    }
    out = OUT / f"krampuschewing_recovery_receipt_{stamp()}.json"
    receipt["report_path"] = rel(out)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("KRAMPUSCHEWING_RECOVERY_GATE=" + verdict)
    print("MASTER_INDEX_PATH=" + str(index_path or ""))
    print("MASTER_SUMMARY_PATH=" + str(summary_path or ""))
    print("GRAPH_CANDIDATES_PATH=" + str(graph_path or ""))
    print("RIVER_TRAINING_CANDIDATES_PATH=" + str(river_path or ""))
    return 0 if verdict == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
