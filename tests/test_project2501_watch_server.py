from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_collect_state_exposes_batteries_logs_and_route_map() -> None:
    from project2501_watch_server import collect_state

    state = collect_state(ROOT)
    assert state["schema"] == "lucidota.project2501.watch_state.v1"
    assert state["batteries"]
    ids = {b["id"] for b in state["batteries"]}
    assert {"deterministic_algos", "codex", "groq", "cohere", "local_models"} <= ids
    assert state["route_map"]["nodes"]
    assert state["route_map"]["edges"]
    assert state["logs"]


def test_render_html_has_eventsource_batteries_and_subway_map() -> None:
    from project2501_watch_server import render_html

    html = render_html()
    assert "EventSource('/events')" in html
    assert "battery-grid" in html
    assert "subway-map" in html
    assert "task-stream" in html


def test_sse_event_formats_strict_json_data_line() -> None:
    from project2501_watch_server import sse_event

    payload = {"schema": "x", "value": 1}
    event = sse_event(payload)
    assert event.startswith("event: state\n")
    data_line = next(line for line in event.splitlines() if line.startswith("data: "))
    assert json.loads(data_line.removeprefix("data: ")) == payload
    assert event.endswith("\n\n")


def test_cli_once_writes_receipt_and_prints_url() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/project2501_watch_server.py", "--once", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PROJECT2501_WATCH_SERVER=PASS" in proc.stdout
    report_path = next(line.split("=", 1)[1] for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert receipt["url"].startswith("http://127.0.0.1:")
    assert receipt["once"] is True
    assert receipt["state"]["route_map"]["edges"]


def test_model_batteries_are_receipt_backed_not_static_workshare(tmp_path) -> None:
    from project2501_watch_server import collect_state

    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text('- Current step: 1/4\n- Status: in_progress\n', encoding='utf-8')
    inv = tmp_path / "05_OUTPUTS" / "model_invocations"
    inv.mkdir(parents=True)
    (inv / "groq_chat_dry_run_20260101T000000Z.json").write_text(json.dumps({
        "schema": "lucidota.model_invocation.groq_chat.v1",
        "status": "PASS",
        "mode": "dry_run",
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model": "fixture-groq-model",
        "request": {"temperature": 0.1, "max_tokens": 8, "messages": [{"role": "system", "content_chars": 10}, {"role": "user", "content_chars": 4}]},
        "text": "",
        "usage": None,
    }), encoding='utf-8')

    state = collect_state(tmp_path)
    batteries = {row["id"]: row for row in state["batteries"]}
    assert batteries["groq"]["pct"] != 25
    assert "25% of LLM residual" not in batteries["groq"]["detail"]
    assert "fixture-groq-model" in batteries["groq"]["detail"]
    assert "groq_chat_dry_run_20260101T000000Z.json" in batteries["groq"]["detail"]
    assert state["model_traces"]["providers"]["groq"]["model"] == "fixture-groq-model"


def test_collect_state_includes_hardware_telemetry() -> None:
    from project2501_watch_server import collect_state

    state = collect_state(ROOT)
    assert "hardware" in state
    assert "memory" in state["hardware"]
    assert "cpu" in state["hardware"]
    battery_ids = {row["id"] for row in state["batteries"]}
    assert {"cpu_load", "ram", "thermal", "gpu"} <= battery_ids


def test_render_html_exposes_hardware_and_model_trace_panels() -> None:
    from project2501_watch_server import render_html

    html = render_html()
    assert "Hardware telemetry" in html
    assert "Postgres live" in html
    assert "Model trace" in html
    assert "LLXPRT Groq orchestrator" in html
    assert "model-trace" in html
    assert "llxprt" in html
    assert "hardware" in html


def test_collect_state_includes_postgres_live_snapshot(monkeypatch) -> None:
    from project2501_watch_server import collect_state

    monkeypatch.setenv("LUCIDOTA_DISABLE_PG_AUTO_PROBE", "1")
    state = collect_state(ROOT)
    assert "postgres" in state
    assert state["postgres"]["schema"] == "lucidota.project2501.postgres_snapshot.v1"
    assert "status" in state["postgres"]
    assert "postgres" in {row["id"] for row in state["batteries"]}


def test_collect_state_includes_llxprt_groq_orchestrator_receipt(tmp_path) -> None:
    from project2501_watch_server import collect_state

    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text('- Current step: 1/4\n', encoding='utf-8')
    out = tmp_path / "05_OUTPUTS" / "llxprt_project2501"
    out.mkdir(parents=True)
    (out / "llxprt_project2501_20260101T000000Z.json").write_text(json.dumps({
        "schema": "lucidota.llxprt_project2501.receipt.v1",
        "status": "PASS",
        "profile": {"name": "lucidota-groq-orchestrator", "exists": True},
        "provider_alias": {"name": "lucidota-groq", "exists": True},
        "groq": {"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1/", "api_key_env_present": True},
        "llxprt": {"binary": "/usr/bin/llxprt", "version": {"stdout": "0.9.3"}},
    }), encoding='utf-8')

    state = collect_state(tmp_path)
    assert state["llxprt"]["profile"] == "lucidota-groq-orchestrator"
    assert state["llxprt"]["model"] == "openai/gpt-oss-120b"
    batteries = {row["id"]: row for row in state["batteries"]}
    assert batteries["llxprt_groq"]["pct"] == 100
    assert "openai/gpt-oss-120b" in batteries["llxprt_groq"]["detail"]


def test_collect_state_exposes_recent_model_generation_ledger(tmp_path) -> None:
    from project2501_watch_server import collect_state

    (tmp_path / "GOALS").mkdir()
    (tmp_path / "GOALS" / "CURRENT_HANDOFF.md").write_text('- Current step: 1/4\n', encoding='utf-8')
    inv = tmp_path / "05_OUTPUTS" / "model_invocations"
    inv.mkdir(parents=True)
    (inv / "groq_chat_execute_20260101T000000Z.json").write_text(json.dumps({
        "schema": "lucidota.model_invocation.groq_chat.v1",
        "status": "PASS",
        "mode": "execute",
        "provider": "groq",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model": "fixture-groq-model",
        "generation_trace": {
            "schema": "lucidota.model_generation_trace.v1",
            "target": "groq",
            "model_name": "fixture-groq-model",
            "payload_size_bytes": 123,
            "payload_size_chars": 123,
            "latency_ms": 45.6,
            "raw_output": "fixture raw output",
            "raw_output_chars": 18,
            "raw_response_present": True,
            "execute_performed": True
        },
        "report_path": "05_OUTPUTS/model_invocations/groq_chat_execute_20260101T000000Z.json"
    }), encoding='utf-8')

    state = collect_state(tmp_path)
    assert state["model_generations"]["schema"] == "lucidota.project2501.model_generation_ledger.v1"
    row = state["model_generations"]["recent"][0]
    assert row["target"] == "groq"
    assert row["model_name"] == "fixture-groq-model"
    assert row["payload_size_bytes"] == 123
    assert row["latency_ms"] == 45.6
    assert row["raw_output"] == "fixture raw output"
    assert "groq_chat_execute_20260101T000000Z.json" in row["receipt_path"]


def test_render_html_exposes_model_generation_ledger_panel() -> None:
    from project2501_watch_server import render_html

    html = render_html()
    assert "Generation ledger" in html
    assert "model-generations" in html


def test_postgres_dsn_targets_control_state_database_by_default(monkeypatch) -> None:
    import project2501_watch_server as watch

    for key in ("LUCIDOTA_CONTROL_DATABASE_URL", "ABSURD_SYSTEM_DATABASE_URL", "DATABASE_URL", "LUCIDOTA_GO_STORAGE_DSN", "PG_DSN"):
        monkeypatch.delenv(key, raising=False)
    assert watch.postgres_dsn() == "postgresql:///lucidota_state"

    monkeypatch.setenv("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///absurd_state")
    assert watch.postgres_dsn() == "postgresql:///absurd_state"


def test_collect_state_exposes_pg_staged_model_generation_events(monkeypatch) -> None:
    import project2501_watch_server as watch

    fake_pg = {
        "schema": "lucidota.project2501.postgres_snapshot.v1",
        "status": "ok",
        "tables": [],
        "model_generation_events": {
            "schema": "lucidota.project2501.pg_model_generation_events.v1",
            "recent": [
                {
                    "target": "groq",
                    "model_name": "openai/gpt-oss-120b",
                    "payload_size_bytes": 321,
                    "latency_ms": 12.5,
                    "raw_output": "OK",
                    "receipt_path": "05_OUTPUTS/model_invocations/x.json",
                    "execute_performed": True,
                }
            ],
        },
    }
    monkeypatch.setattr(watch, "postgres_snapshot_async", lambda ttl=2.0: fake_pg)

    state = watch.collect_state(ROOT)

    assert state["pg_model_generation_events"]["schema"] == "lucidota.project2501.pg_model_generation_events.v1"
    row = state["pg_model_generation_events"]["recent"][0]
    assert row["target"] == "groq"
    assert row["model_name"] == "openai/gpt-oss-120b"
    assert row["payload_size_bytes"] == 321
    assert row["raw_output"] == "OK"


def test_render_html_exposes_pg_model_generation_events_panel() -> None:
    from project2501_watch_server import render_html

    html = render_html()
    assert "PG generation events" in html
    assert "pg-model-generations" in html
