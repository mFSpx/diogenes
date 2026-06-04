from __future__ import annotations

from scripts import indy_daemon, mamba_db_watch


def test_respond_once_command_targets_indy_reads_chat() -> None:
    cmd = indy_daemon.respond_once_command()
    assert cmd[1].endswith("scripts/indy_reads.py")
    assert cmd[-3:] == ["chat", "--respond-once", "--json"]


def test_run_once_invokes_respond_once_when_queue_has_rows(monkeypatch) -> None:
    monkeypatch.setattr(indy_daemon.indy_runtime_broker, "registry_snapshot", lambda **kwargs: {"schema": "lucidota.indy_runtime_broker.snapshot.v1"})
    monkeypatch.setattr(mamba_db_watch, "poll_once", lambda **kwargs: {"row_count": 1, "event_ids": ["event-1"]})

    calls = {}

    class FakeProc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    def fake_run(cmd, cwd=None, capture_output=None, text=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        return FakeProc()

    monkeypatch.setattr(indy_daemon.subprocess, "run", fake_run)
    result = indy_daemon.run_once()
    assert result["responded"] is True
    assert calls["cmd"][-3:] == ["chat", "--respond-once", "--json"]
