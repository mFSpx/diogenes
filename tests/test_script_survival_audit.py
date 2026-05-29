#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from script_survival_audit import KRAMPUS_ACTIONS, script_facts


def test_script_facts_extracts_python_contract():
    facts = script_facts(ROOT / "scripts/script_survival_audit.py")
    assert facts["path"] == "scripts/script_survival_audit.py"
    assert facts["hash_blake3_or_sha256"]
    assert "argparse" in facts["imports"]
    assert "main" in facts["functions"]
    assert "--verdict" in facts["argparse_flags"]


def test_active_audit_appends_only_audit_manifest(tmp_path):
    audit = tmp_path / "SCRIPT_AUDIT_MANIFEST.jsonl"
    corpse = tmp_path / "CORPSE_MANIFEST.jsonl"
    krampus = tmp_path / "KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/script_survival_audit.py"),
            "--path",
            "scripts/script_survival_audit.py",
            "--verdict",
            "ACTIVE_KEEP",
            "--role",
            "audit_manifest_writer",
            "--slop-score",
            "1",
            "--survival-score",
            "8",
            "--ncnn-alignment",
            "7 deterministic CLI, compact dataflow",
            "--flow-alignment",
            "8 append-only manifest writer",
            "--decision-reason",
            "needed to preserve script survival verdicts",
            "--audit-manifest",
            str(audit),
            "--corpse-manifest",
            str(corpse),
            "--krampus-queue",
            str(krampus),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "SCRIPT_SURVIVAL_AUDIT=PASS" in proc.stdout
    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["verdict"] == "ACTIVE_KEEP"
    assert not corpse.exists()
    assert not krampus.exists()


def test_corpse_audit_feeds_krampus_queue(tmp_path):
    audit = tmp_path / "SCRIPT_AUDIT_MANIFEST.jsonl"
    corpse = tmp_path / "CORPSE_MANIFEST.jsonl"
    krampus = tmp_path / "KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl"
    archive = tmp_path / "KRAMPUSCHEWING" / "Script_Corpses"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/script_survival_audit.py"),
            "--path",
            "scripts/legacy/dbos_chrono_worker.py",
            "--verdict",
            "LEGACY_CORPSE",
            "--role",
            "legacy_worker",
            "--slop-score",
            "10",
            "--survival-score",
            "2",
            "--ncnn-alignment",
            "2 superseded legacy path",
            "--flow-alignment",
            "2 outside active ABSURD flow",
            "--decision-reason",
            "superseded by ABSURD chrono worker",
            "--death-reason",
            "DBOS legacy worker superseded",
            "--superseded-by",
            "scripts/absurd_chrono_worker.py",
            "--risk-if-kept-active",
            "legacy event source could confuse active source-trust gates",
            "--historical-value",
            "preserves DBOS to ABSURD migration evidence",
            "--evidence-read",
            "scripts/legacy/DBOS_LEGACY_ARCHIVE_MANIFEST.json",
            "--audit-manifest",
            str(audit),
            "--corpse-manifest",
            str(corpse),
            "--krampus-queue",
            str(krampus),
            "--krampus-code-archive",
            str(archive),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    corpse_row = json.loads(corpse.read_text(encoding="utf-8").splitlines()[0])
    krampus_row = json.loads(krampus.read_text(encoding="utf-8").splitlines()[0])
    assert corpse_row["krampuschewing_ingest"] is True
    assert krampus_row["ingest_class"] == "SCRIPT_CORPSE"
    assert krampus_row["desired_actions"] == KRAMPUS_ACTIONS
    assert krampus_row["never_delete"] is True
    assert corpse_row["krampuschewing_archived_copy"]
    assert krampus_row["archived_copy"] == corpse_row["krampuschewing_archived_copy"]
    archived_path = Path(corpse_row["krampuschewing_archived_copy"])
    if not archived_path.is_absolute():
        archived_path = ROOT / archived_path
    assert archived_path.exists()
