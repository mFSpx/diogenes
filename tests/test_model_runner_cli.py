#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_runner_stub import run_stub_model  # noqa: E402


def test_stub_runner_rejects_non_stub_backend_to_avoid_fake_inference_claim() -> None:
    result = run_stub_model(
        {"model_id": "realish.gguf", "backend": "llama.cpp", "requested_vram_mb": 512, "available_vram_mb": 4096},
        "hello",
    )
    assert result["status"] == "REJECTED"
    assert result["real_inference_performed"] is False
    assert result["validation"]["error"] == "STUB_RUNNER_REQUIRES_STUB_BACKEND"


def test_model_runner_cli_validate_and_stub() -> None:
    validate = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "validate",
            "--model-id",
            "fixture.gguf",
            "--backend",
            "STUB",
            "--requested-vram-mb",
            "512",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert validate.returncode == 0, validate.stderr
    assert json.loads(validate.stdout.splitlines()[0])["ok"] is True

    stub = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "stub",
            "--model-id",
            "fixture.gguf",
            "--backend",
            "STUB",
            "--requested-vram-mb",
            "512",
            "--prompt",
            "pytest prompt",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert stub.returncode == 0, stub.stderr
    payload = next(json.loads(line) for line in stub.stdout.splitlines() if line.startswith("{"))
    assert payload["status"] == "PASSED"
    assert payload["real_inference_performed"] is False
    assert "RECEIPT_PATH=" in stub.stdout


def test_cloud_chat_dry_runs_do_not_require_api_keys() -> None:
    groq = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "groq-chat",
            "--prompt",
            "ping",
            "--max-tokens",
            "8",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env={k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k},
    )
    assert groq.returncode == 0, groq.stderr
    groq_payload = next(json.loads(line) for line in groq.stdout.splitlines() if line.startswith("{"))
    assert groq_payload["mode"] == "dry_run"
    assert groq_payload["execute_performed"] is False
    assert groq_payload["status"] == "PASS"

    cohere = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "cohere-chat",
            "--prompt",
            "ping",
            "--max-tokens",
            "8",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env={k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k},
    )
    assert cohere.returncode == 0, cohere.stderr
    cohere_payload = next(json.loads(line) for line in cohere.stdout.splitlines() if line.startswith("{"))
    assert cohere_payload["mode"] == "dry_run"
    assert cohere_payload["execute_performed"] is False
    assert cohere_payload["status"] == "PASS"


def test_cloud_chat_receipts_expose_exact_model_request_text_by_default() -> None:
    groq = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "groq-chat",
            "--prompt",
            "operator payload visible",
            "--system",
            "caller system context visible",
            "--temperature",
            "0.1",
            "--max-tokens",
            "8",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env={k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k},
    )
    assert groq.returncode == 0, groq.stderr
    groq_payload = next(json.loads(line) for line in groq.stdout.splitlines() if line.startswith("{"))
    assert groq_payload["model"] == "llama-3.1-8b-instant"
    assert groq_payload["request"]["temperature"] == 0.1
    assert groq_payload["request"]["max_tokens"] == 8
    assert any(m.get("content_text") == "operator payload visible" for m in groq_payload["request"]["messages"])
    assert "wire_request" in groq_payload
    assert any(m.get("content") == "operator payload visible" for m in groq_payload["wire_request"]["messages"])

    cohere = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "cohere-chat",
            "--prompt",
            "operator payload visible",
            "--system",
            "caller system context visible",
            "--temperature",
            "0.1",
            "--max-tokens",
            "8",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env={k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k},
    )
    assert cohere.returncode == 0, cohere.stderr
    cohere_payload = next(json.loads(line) for line in cohere.stdout.splitlines() if line.startswith("{"))
    assert cohere_payload["model"] == "command-a-03-2025"
    assert any(m.get("content_text") == "operator payload visible" for m in cohere_payload["request"]["messages"])
    assert any(m.get("content") == "operator payload visible" for m in cohere_payload["wire_request"]["messages"])


def assert_generation_trace(payload: dict, target: str, model: str) -> None:
    trace = payload["generation_trace"]
    assert trace["target"] == target
    assert trace["model_name"] == model
    assert trace["payload_size_bytes"] > 0
    assert trace["payload_size_chars"] > 0
    assert trace["latency_ms"] >= 0
    assert "raw_output" in trace
    assert isinstance(trace["raw_output_chars"], int)


def test_model_chat_receipts_have_generation_routing_trace() -> None:
    clean_env = {k: v for k, v in os.environ.items() if "GROQ_API_KEY" not in k and "COHERE_API_KEY" not in k and "CO_API_KEY" not in k}
    groq = subprocess.run(
        [sys.executable, "scripts/model_runner_cli.py", "groq-chat", "--prompt", "trace ping", "--temperature", "0", "--max-tokens", "8", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env=clean_env,
    )
    assert groq.returncode == 0, groq.stderr
    groq_payload = next(json.loads(line) for line in groq.stdout.splitlines() if line.startswith("{"))
    assert_generation_trace(groq_payload, "groq", "llama-3.1-8b-instant")

    cohere = subprocess.run(
        [sys.executable, "scripts/model_runner_cli.py", "cohere-chat", "--prompt", "trace ping", "--temperature", "0", "--max-tokens", "8", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env=clean_env,
    )
    assert cohere.returncode == 0, cohere.stderr
    cohere_payload = next(json.loads(line) for line in cohere.stdout.splitlines() if line.startswith("{"))
    assert_generation_trace(cohere_payload, "cohere", "command-a-03-2025")

    local = subprocess.run(
        [sys.executable, "scripts/model_runner_cli.py", "local-chat", "--lane", "deepseek", "--prompt", "trace ping", "--temperature", "0", "--max-tokens", "8", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        env=clean_env,
    )
    assert local.returncode == 0, local.stderr
    local_payload = next(json.loads(line) for line in local.stdout.splitlines() if line.startswith("{"))
    assert_generation_trace(local_payload, "local", "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")


def test_response_extractors_fallback_to_reasoning_when_content_is_empty() -> None:
    from groq_chat_cli import groq_text
    from local_model_chat_cli import response_text

    groq_response = {"choices": [{"message": {"content": "", "reasoning": "raw groq reasoning"}}]}
    assert groq_text(groq_response) == "raw groq reasoning"

    local_spec = {"kind": "llama.cpp"}
    local_response = {"choices": [{"message": {"content": "", "reasoning_content": "raw local reasoning"}}]}
    assert response_text(local_spec, local_response) == "raw local reasoning"
