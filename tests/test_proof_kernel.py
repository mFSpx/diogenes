from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/proof_kernel.py"


def run_pk(tmp_path: Path, args: list[str], expect: int = 0):
    storage = tmp_path / "storage"
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPT), "--storage-root", str(storage), "--output-dir", str(out), *args]
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, (p.returncode, p.stdout, p.stderr, cmd)
    return p, storage, out


def receipt_from(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            path = Path(line.split("=", 1)[1])
            return json.loads(path.read_text(encoding="utf-8"))
    raise AssertionError(stdout)


def test_file_ingest_and_immutable_stored_bytes(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("exact bytes\n", encoding="utf-8")
    p, storage, _ = run_pk(tmp_path, ["file-ingest", "--source", str(source)])
    r = receipt_from(p.stdout)
    proof = r["proof_object"]
    stored = ROOT / proof["stored_artifact_path"] if not Path(proof["stored_artifact_path"]).is_absolute() else Path(proof["stored_artifact_path"])
    assert proof["source_type"] == "file"
    assert stored.read_bytes() == source.read_bytes()
    assert not (stored.stat().st_mode & stat.S_IWUSR)
    assert r["canonical_graph_writes_performed"] is False


def test_text_ingest_preserves_exact_text_and_lookup(tmp_path):
    p, storage, _ = run_pk(tmp_path, ["text-ingest", "--text", "one\ntwo", "--operator-id", "op", "--session-id", "sess"])
    r = receipt_from(p.stdout)
    proof = r["proof_object"]
    stored = ROOT / proof["stored_artifact_path"] if not Path(proof["stored_artifact_path"]).is_absolute() else Path(proof["stored_artifact_path"])
    assert stored.read_text(encoding="utf-8") == "one\ntwo"
    lookup, _, _ = run_pk(tmp_path, ["lookup", "--proof-id", proof["proof_id"]])
    lr = receipt_from(lookup.stdout)
    assert lr["verdict"] == "PASS"
    assert lr["matches"][0]["proof_id"] == proof["proof_id"]


def test_command_output_ingest_without_rerun(tmp_path):
    p, _, _ = run_pk(tmp_path, [
        "command-output-ingest",
        "--command", "echo hello",
        "--cwd", str(ROOT),
        "--return-code", "0",
        "--stdout", "hello\n",
        "--stderr", "",
        "--redaction-status", "none",
    ])
    r = receipt_from(p.stdout)
    manifest = r["proof_object"]["command_output"]
    assert manifest["command"] == "echo hello"
    assert manifest["return_code"] == 0
    assert manifest["rerun_performed"] is False
    assert manifest["stdout_sha256"]


def test_diff_ingest_records_patch_and_changed_files(tmp_path):
    patch = "diff --git a/a.txt b/a.txt\n+hello\n"
    p, _, _ = run_pk(tmp_path, ["diff-ingest", "--patch-text", patch, "--changed-file", "a.txt", "--base-id", "base", "--head-id", "head"])
    r = receipt_from(p.stdout)
    diff = r["proof_object"]["code_diff"]
    assert diff["patch_text"] == patch
    assert diff["changed_files"] == ["a.txt"]
    assert diff["base_identifier"] == "base"
    assert diff["head_identifier"] == "head"


def test_duplicate_hash_handling(tmp_path):
    source = tmp_path / "dup.txt"
    source.write_text("duplicate", encoding="utf-8")
    p1, _, _ = run_pk(tmp_path, ["file-ingest", "--source", str(source)])
    p2, _, _ = run_pk(tmp_path, ["file-ingest", "--source", str(source)])
    r1 = receipt_from(p1.stdout)
    r2 = receipt_from(p2.stdout)
    assert r1["proof_object"]["sha256"] == r2["proof_object"]["sha256"]
    assert r2["index_inserted"] is False or r2["proof_object"]["storage_existed"] is True


def test_missing_source_failure(tmp_path):
    p, _, _ = run_pk(tmp_path, ["file-ingest", "--source", str(tmp_path / "missing.txt")], expect=4)
    r = receipt_from(p.stdout)
    assert r["verdict"] == "FAIL"
    assert r["blockers"]


def test_lookup_missing_fails(tmp_path):
    p, _, _ = run_pk(tmp_path, ["lookup", "--proof-id", "proof_deadbeefdeadbeefdeadbeefdeadbeef"], expect=4)
    r = receipt_from(p.stdout)
    assert "proof_not_found" in r["blockers"]
