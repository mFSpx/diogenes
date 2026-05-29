#!/usr/bin/env python3
from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import absurd_river_worker as worker


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def commit(self) -> None:
        return None


def install_offline_db(monkeypatch) -> None:
    monkeypatch.setattr(worker.psycopg, "connect", lambda *_args, **_kwargs: FakeConnection())
    monkeypatch.setattr(worker, "ensure_state_schema", lambda _conn: None)
    monkeypatch.setattr(worker, "count_tables", lambda _conn, tables: {table: 0 for table in tables})


def make_args(**overrides):
    defaults = {
        "state_database_url": "postgresql:///unused_state",
        "storage_database_url": "postgresql:///unused_storage",
        "queue": worker.DEFAULT_QUEUE,
        "job_kind": worker.DEFAULT_JOB_KIND,
        "labels": ["Operator"],
        "gliner_model": None,
        "gliner_threshold": 0.35,
        "allow_remote_model": False,
        "component_limit": 1,
        "min_chars": 1,
        "python": sys.executable,
        "bytewax_limit": 1,
        "river_limit": 1,
        "claim_packet_limit": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_audit_blocks_literal_gliner_fallback_when_gliner_import_is_missing(monkeypatch):
    install_offline_db(monkeypatch)
    monkeypatch.setattr(
        worker,
        "load_korpus_components",
        lambda _args, _payload=None: [
            {
                "content": "Operator evidence mentions the Operator label exactly.",
                "file_uuid": "00000000-0000-0000-0000-000000000001",
                "component_uuid": "00000000-0000-0000-0000-000000000002",
                "component_index": 0,
                "component_sha256": "sha-component",
                "file_sha256": "sha-file",
                "source_path": "offline-fixture.md",
            }
        ],
    )

    real_import = builtins.__import__

    def import_without_gliner(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gliner" or name.startswith("gliner."):
            raise ImportError("offline test: GLiNER intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_gliner)

    captured: dict[str, dict] = {}

    def capture_report(_action, report):
        captured["report"] = report
        return ROOT / "unused-report.json"

    monkeypatch.setattr(worker, "write_report", capture_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "absurd_river_worker.py",
            "--action",
            "audit",
            "--dry-run",
            "--job-kind",
            worker.DEFAULT_JOB_KIND,
            "--labels",
            "Operator",
            "--component-limit",
            "1",
            "--min-chars",
            "1",
        ],
    )

    exit_code = worker.main()

    report = captured["report"]
    backend = report["action_result"]["health"]["backend_detail"].get("backend")
    blockers = report["blockers"]
    assert exit_code == 1, (
        "deterministic-core audit must hard-fail when GLiNER falls back instead "
        f"of using a real model; backend={backend!r}, blockers={blockers!r}"
    )
    assert blockers, f"literal fallback backend {backend!r} cannot be accepted with no blocker"


def test_river_bytewax_tick_blocks_when_root_scripts_are_missing(monkeypatch, tmp_path):
    install_offline_db(monkeypatch)
    monkeypatch.setattr(
        worker,
        "dependency_probe",
        lambda py: {
            "python": py,
            "returncode": 0,
            "modules": {"psycopg": "ok", "river": "ok", "bytewax": "ok", "gliner": "ok"},
            "stderr_tail": "",
        },
    )
    monkeypatch.setattr(worker, "RIVER_SCRIPT", tmp_path / "scripts" / "lucidota_river_reflex.py")
    monkeypatch.setattr(worker, "BYTEWAX_SCRIPT", tmp_path / "scripts" / "lucidota_bytewax_mini.py")
    monkeypatch.setattr(worker, "STREAM_WORKER", tmp_path / "scripts" / "lucidota_stream_river_worker.sh")

    def fail_if_tick_runs(*_args, **_kwargs):
        raise AssertionError("missing root scripts must block before tick commands run")

    monkeypatch.setattr(worker, "run_cmd", fail_if_tick_runs)

    health, blockers = worker.river_bytewax_health(
        make_args(job_kind=worker.LEGACY_JOB_KIND),
        payload={},
        run_tick=True,
    )

    assert set(blockers) >= {
        "river_reflex_script_missing",
        "bytewax_mini_script_missing",
        "stream_worker_script_missing",
    }
    assert health["command_results"] == {}
