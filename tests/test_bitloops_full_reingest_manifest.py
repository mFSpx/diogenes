#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_manifest_classifies_luci_krampus_pony_and_korpus_roots(tmp_path):
    from bitloops_full_reingest_manifest import build_manifest

    luci = tmp_path / "LUCIDOTA_CURRENT"
    krampus = tmp_path / "KRAMPUSCHEWING"
    pony = krampus / "Lucidota" / "Lucidota" / "PONYBOY"
    korpus = tmp_path / "03_VAULT" / "korpus_krampii" / "DIGESTED"
    for path, text in [
        (luci / "GOALS" / "note.md", "current luci goal"),
        (krampus / "case.txt", "krampus case"),
        (pony / "pony.txt", "pony evidence"),
        (korpus / "digest.md", "korpus digest"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    rows_path = tmp_path / "manifest.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = build_manifest(
        roots=[luci, krampus, korpus],
        output=rows_path,
        summary_output=summary_path,
        max_files=0,
    )

    rows = read_jsonl(rows_path)
    families = {row["source_family"] for row in rows}
    assert {"lucidota_current", "krampuschewing", "ponyboy", "korpus_krampii"} <= families
    assert all(row["schema"] == "lucidota.bitloops.full_reingest_manifest.row.v1" for row in rows)
    assert all(len(row["source_sha256"]) == 64 for row in rows)
    assert all(row["truth_status"] == "training_candidate_only" for row in rows)
    assert summary["status"] == "PASS"
    assert summary["files_indexed"] == 4
    assert summary["canonical_graph_writes_performed"] is False
    assert summary["purged_case_count"] == 0
    assert json.loads(summary_path.read_text(encoding="utf-8"))["manifest_path"] == str(rows_path)


def test_cli_manifest_feeds_bitloops_automation_loop(tmp_path):
    source = tmp_path / "KRAMPUSCHEWING" / "Lucidota" / "Lucidota" / "PONYBOY" / "signal.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("pony signal becomes recovered bitloops river lane", encoding="utf-8")
    rows_path = tmp_path / "full_reingest.jsonl"
    summary_path = tmp_path / "full_reingest_summary.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bitloops_full_reingest_manifest.py",
            "--root",
            str(tmp_path / "KRAMPUSCHEWING"),
            "--output",
            str(rows_path),
            "--summary-output",
            str(summary_path),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    rows = read_jsonl(rows_path)
    assert rows[0]["source_family"] == "ponyboy"

    loop = subprocess.run(
        [sys.executable, "scripts/bitloops_automation_loop.py", "--legacy-jsonl", str(rows_path), "--legacy-etl", "full_world_reingest", "--limit", "1", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert loop.returncode == 0, loop.stderr
    receipt_line = next(line for line in loop.stdout.splitlines() if line.startswith("REPORT_PATH="))
    report = json.loads((ROOT / receipt_line.split("=", 1)[1]).read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["accepted_case_ids"] == [rows[0]["row_id"]]
    assert report["river_training_lane_count"] == 4
    assert report["canonical_graph_writes_performed"] is False
