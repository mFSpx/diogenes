#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from model_runner_stub import run_stub_model  # noqa: E402
from gemini_chat_cli import execute_gemini_with_key_fallback  # noqa: E402
from model_runner_cli import run_bonsai_chain  # noqa: E402


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


def test_gemini_chat_dry_run_is_available_without_api_key() -> None:
    gemini = subprocess.run(
        [
            sys.executable,
            "scripts/model_runner_cli.py",
            "gemini-chat",
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
        env={k: v for k, v in os.environ.items() if "GEMINI_API_KEY" not in k and "GOOGLE_API_KEY" not in k},
    )
    assert gemini.returncode == 0, gemini.stderr
    gemini_payload = next(json.loads(line) for line in gemini.stdout.splitlines() if line.startswith("{"))
    assert gemini_payload["mode"] == "dry_run"
    assert gemini_payload["execute_performed"] is False
    assert gemini_payload["status"] == "PASS"
    assert gemini_payload["provider"] == "gemini"
    assert gemini_payload["model"] == "gemini-2.5-flash"


def test_gemini_chat_execute_falls_back_to_second_key_on_quota_error() -> None:
    calls: list[str] = []

    def fake_call(base_url, key, model, request_payload, timeout):
        calls.append(key)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url=base_url,
                code=429,
                msg="quota exhausted",
                hdrs=None,
                fp=None,
            )
        return {
            "responseId": "ok",
            "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}],
        }

    response, key_env_used, blockers, error_body, attempt_blockers = execute_gemini_with_key_fallback(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        model="gemini-2.5-flash",
        request_payload={"model": "gemini-2.5-flash", "contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
        timeout=5.0,
        key_candidates=[("GEMINI_API_KEY", "billing-key"), ("GOOGLE_API_KEY", "free-key")],
        call_fn=fake_call,
    )

    assert response["responseId"] == "ok"
    assert key_env_used == "GOOGLE_API_KEY"
    assert blockers == []
    assert error_body == ""
    assert attempt_blockers == ["gemini_http_error:429"]
    assert calls == ["billing-key", "free-key"]


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


def test_bonsai_chain_routes_through_bonsai_needles_bonsai_order() -> None:
    calls: list[tuple[str, bool, str]] = []

    def fake_probe_lane(*, lane: str, prompt: str, system: str = "", execute: bool = False, **_: object) -> dict:
        calls.append((lane, execute, prompt))
        return {
            "status": "PASS",
            "lane": lane,
            "execute_performed": execute,
            "report_path": f"05_OUTPUTS/model_invocations/fake_{lane}.json",
            "text": f"{lane}:{prompt}",
        }

    payload = run_bonsai_chain(
        prompt="route this",
        system="system seed",
        execute=False,
        lane_runner=fake_probe_lane,
    )

    assert payload["status"] == "PASS"
    assert payload["lane_sequence"] == ["bonsai_q1_0", "needle_0", "needle_1", "needle_2", "needle_3", "needle_4", "needle_5", "bonsai_q1_0"]
    assert [lane for lane, _, _ in calls] == payload["lane_sequence"]
    assert payload["needle_stage_count"] == 6
    assert payload["final_lane"] == "bonsai_q1_0"
    assert payload["merge_strategy"] == "deterministic_needle_digest"
