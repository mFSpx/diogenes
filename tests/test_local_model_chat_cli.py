#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_local_llama_execute_receipt_measures_tokens(tmp_path, monkeypatch):
    import local_model_chat_cli as local_chat

    monkeypatch.setattr(local_chat, "OUT", tmp_path)
    monkeypatch.setattr(
        local_chat,
        "post_json",
        lambda url, payload, timeout: {
            "choices": [{"message": {"content": "ok from mamba"}}],
            "model": "Falcon3-Mamba-7B-Instruct-Q2_K.gguf",
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        },
    )

    receipt = local_chat.probe_lane(
        lane="mamba_gpu",
        prompt="prove tokens",
        system="be terse",
        max_tokens=8,
        temperature=0.0,
        timeout=1.0,
        execute=True,
        log_prompts=True,
    )

    assert receipt["status"] == "PASS"
    assert receipt["model_calls_performed"] is True
    assert receipt["real_inference_performed"] is True
    assert receipt["token_accounting"]["source"] == "provider_usage"
    assert receipt["token_accounting"]["total_tokens"] == 8
    assert receipt["request"]["prompt_text"] == "prove tokens"
    assert Path(ROOT / receipt["report_path"]).exists()


def test_local_needle_execute_receipt_estimates_tokens(tmp_path, monkeypatch):
    import local_model_chat_cli as local_chat

    monkeypatch.setattr(local_chat, "OUT", tmp_path)
    monkeypatch.setattr(local_chat, "post_json", lambda url, payload, timeout: {"ok": True, "model": "needle-26m", "output": " []"})

    receipt = local_chat.probe_lane(
        lane="needle_0",
        prompt="route one tool",
        system="",
        max_tokens=8,
        temperature=0.0,
        timeout=1.0,
        execute=True,
        log_prompts=True,
    )

    assert receipt["status"] == "PASS"
    assert receipt["endpoint"].endswith(":8090/generate")
    assert receipt["token_accounting"]["source"] == "char_estimate"
    assert receipt["token_accounting"]["total_tokens"] > 0
    assert receipt["text"] == " []"


def test_model_runner_cli_local_chat_dry_run_has_no_model_call():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "local-chat",
            "--lane",
            "deepseek",
            "--prompt",
            "dry run only",
            "--max-tokens",
            "4",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["mode"] == "dry_run"
    assert payload["provider"] == "local"
    assert payload["model_calls_performed"] is False
    assert payload["token_accounting"]["source"] == "dry_run_estimate"
