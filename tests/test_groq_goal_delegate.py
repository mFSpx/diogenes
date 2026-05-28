#!/usr/bin/env python3
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def report(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("REPORT_PATH="):
            return json.loads((ROOT / line.split("=", 1)[1]).read_text())
    raise AssertionError(stdout)


def clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}


def test_groq_catalog_dry_run_is_redacted_and_local_first():
    proc = subprocess.run([sys.executable, "scripts/groq_model_catalog.py", "--json"], cwd=ROOT, text=True, capture_output=True, env=clean_env(), timeout=10)
    assert proc.returncode == 0, proc.stderr
    r = report(proc.stdout)
    assert r["execute_performed"] is False
    assert r["api_key_redacted"] is False
    assert r["recommendation"]["default_model"] is None
    assert "Core LUCIDOTA never requires cloud" in r["policy"]


def test_groq_delegate_dry_run_does_not_need_key_or_call_model():
    proc = subprocess.run([sys.executable, "scripts/groq_goal_delegate.py", "--task", "audit one file", "--json"], cwd=ROOT, text=True, capture_output=True, env=clean_env(), timeout=10)
    assert proc.returncode == 0, proc.stderr
    r = report(proc.stdout)
    assert r["execute_performed"] is False
    assert r["model_calls_performed"] is False
    assert r["api_key_redacted"] is False


def test_groq_delegate_prompt_includes_official_ontology():
    import scripts.groq_goal_delegate as delegate
    body = delegate.prompt("route this", [], "audit")
    assert "GO-25" in body
    assert "ENTITY" in body
    assert "EVIDENCE supports CLAIM" in body


def test_groq_delegate_execute_fails_closed_without_key():
    proc = subprocess.run([sys.executable, "scripts/groq_goal_delegate.py", "--task", "audit one file", "--execute", "--json"], cwd=ROOT, text=True, capture_output=True, env=clean_env(), timeout=10)
    assert proc.returncode == 4
    r = report(proc.stdout)
    assert r["blockers"] == ["missing_api_key_env:GROQ_API_KEY"]


def test_groq_delegate_execute_surfaces_subreceipt_blockers(monkeypatch, capsys):
    import scripts.groq_goal_delegate as delegate

    sub = ROOT / "05_OUTPUTS" / "model_invocations" / "fake_groq_401.json"
    sub.parent.mkdir(parents=True, exist_ok=True)
    sub.write_text(json.dumps({"blockers": ["groq_http_error:401"], "usage": None, "text": ""}))

    class P:
        returncode = 4
        stdout = "RECEIPT_PATH=05_OUTPUTS/model_invocations/fake_groq_401.json\n"
        stderr = ""

    monkeypatch.setenv("GROQ_API_KEY", "redacted-test-key")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: P())
    monkeypatch.setattr(sys, "argv", [sys.executable, "--task", "audit one file", "--execute", "--json"])

    assert delegate.main() == 4
    r = report(capsys.readouterr().out)
    assert "groq_http_error:401" in r["blockers"]
    assert r["prompt_path"].startswith("04_RUNTIME/goals/groq_delegate_")


def test_groq_delegate_is_allowlisted_for_absurd_external_command():
    import scripts.absurd_queue_spine as spine
    import scripts.absurd_consume_one as consume
    assert "scripts/groq_goal_delegate.py" in spine.ALLOWED_EXTERNAL_COMMANDS
    assert "scripts/groq_goal_delegate.py" in consume.ALLOWED_EXTERNAL_COMMANDS


def test_groq_helpers_stay_tiny():
    for rel in ["scripts/groq_model_catalog.py", "scripts/groq_goal_delegate.py"]:
        lines = Path(ROOT / rel).read_text().splitlines()
        code = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
        assert len(code) <= 100
