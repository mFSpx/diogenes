#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lucidota_usecase_proof as proof  # noqa: E402


def test_email_campaign_is_deterministic_draft_and_language_routed(tmp_path: Path) -> None:
    packet = proof.email_campaign("CASE-20260526-PROOF", tmp_path)
    text = Path(packet["path"]).read_text(encoding="utf-8")
    assert "DRAFT ONLY" in text
    assert "CASE-20260526-PROOF" in text
    assert packet["route"]["lane"] == "rete_regex"
    assert packet["model_calls_performed"] is False


def test_hypertimeline_fixture_has_two_timestamp_sources(tmp_path: Path) -> None:
    files = proof.write_hypertimeline_fixture(tmp_path)
    assert len(files) == 2
    rows = [json.loads(line) for p in files for line in Path(p).read_text().splitlines()]
    assert {row["source"] for row in rows} == {"codex", "claude"}
    assert sum(1 for row in rows if row["timestamp"]) >= 60


def test_proxy_mask_identity_proof_is_local_and_procedural(tmp_path: Path) -> None:
    packet = proof.proxy_mask_identity_proof("CASE-20260526-PROOF", tmp_path)
    assert packet["bridge"]["status"] == "ROUTED"
    assert packet["bridge"]["percyphon"]["zero_vram"] is True
    assert packet["bridge"]["percyphon"]["authority"] == "procedural_scaffold_candidate_not_truth"
    assert packet["anthropic_route"]["model"] == "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
    assert packet["anthropic_route"]["stream"] is False
    assert packet["model_calls_performed"] is False


def test_prove_writes_single_receipt_and_runs_bounded_commands(monkeypatch, tmp_path: Path) -> None:
    calls = []
    def fake_run(cmd, timeout):
        calls.append(cmd)
        return {"cmd": " ".join(cmd), "rc": 0, "stdout_tail": "REPORT_PATH=fake.json", "stderr_tail": "", "timeout_sec": timeout}
    monkeypatch.setattr(proof, "run_cmd", fake_run)
    result = proof.prove("CASE-20260526-PROOF", tmp_path, timeout=7)
    assert result["status"] == "PASSED"
    assert Path(result["report_path"]).exists()
    assert any("lucidota_acceptance.py" in c for cmd in calls for c in cmd)
    assert any("hypertimeline_engine.py" in c for cmd in calls for c in cmd)
    assert any("lucidota_ouroboros_loop.py" in c for cmd in calls for c in cmd)
    assert any("lucidota_bytewax_mini.py" in c for cmd in calls for c in cmd)
