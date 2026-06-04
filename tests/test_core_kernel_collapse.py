from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import indy_reads


def test_luci_runtime_standalone_loop_is_absent() -> None:
    assert not (ROOT / "scripts" / "luci_runtime.py").exists()


def test_runtime_spine_targets_canonical_8_only() -> None:
    text = (ROOT / "06_SCHEMA" / "20260603_runtime_spine.sql").read_text(encoding="utf-8")
    forbidden = [
        "CREATE SCHEMA runtime",
        "CREATE SCHEMA io",
        "runtime.daemon_state",
        "runtime.heartbeat",
        "io.legacy_atomized",
        "runtime.work_order",
        "runtime.enqueue",
    ]
    for needle in forbidden:
        assert needle not in text
    for needle in [
        "ironclaw.daemon_registry",
        "ironclaw.daemon_heartbeats",
        "lucidota_control.legacy_atomized_evidence",
        "lucidota_ontology.canonical_frameworks",
    ]:
        assert needle in text


def test_absurd_spine_registers_atomize_jobs() -> None:
    text = (ROOT / "scripts" / "absurd_queue_spine.py").read_text(encoding="utf-8")
    assert "intake.atomize_json" in text
    assert "intake.atomize_csv" in text
    assert "from indy_ops import handle_atomize_csv_file, handle_atomize_json_file" in text


def test_indy_reads_ledgers_and_dual_state_gate_present() -> None:
    text = (ROOT / "scripts" / "indy_reads.py").read_text(encoding="utf-8")
    for needle in [
        "ironclaw.daemon_heartbeats",
        "ironclaw.indy_read_judgments",
        "/tmp/lucidota_ego.sock",
        "Collaborative companion mode",
        "Autonomous slot claimed",
    ]:
        assert needle in text


def test_indy_reads_autonomous_slow_lane_tick_records_attention(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(indy_reads, "transport_socket_active", lambda path=indy_reads.TRANSPORT_SOCKET: False)
    monkeypatch.setattr(
        indy_reads,
        "tune_ingestion_batch_size",
        lambda st, book, page, parser, socket_active, score=None: {
            "batch_size": 7,
            "river_probability": 0.25,
            "features": {"socket_active": int(socket_active), "score": score},
        },
    )
    monkeypatch.setattr(indy_reads, "record_daemon_heartbeat", lambda **kwargs: calls.append(("heartbeat", kwargs)))
    monkeypatch.setattr(indy_reads, "save_state", lambda st: calls.append(("save_state", dict(st))))

    state = {}
    tick = indy_reads.run_autonomous_slow_lane_tick(state, None, None, None)

    assert tick["socket_active"] is False
    assert state["slow_lane"]["last_autonomous_reason"] == "terminal_timeout"
    assert calls and calls[0][0] == "heartbeat"
