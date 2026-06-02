import subprocess
from pathlib import Path
from types import SimpleNamespace

import scripts.root_rotor_postgrest_control as control


def write_conf(path: Path, *, host="127.0.0.1", server_port="3000", admin_port="3001") -> None:
    path.write_text(
        "\n".join(
            [
                'db-uri = "postgresql:///lucidota_state"',
                'db-schemas = "lucidota_canon"',
                'db-anon-role = "mfspx"',
                f'server-host = "{host}"',
                f'server-port = "{server_port}"',
                f'admin-server-port = "{admin_port}"',
                'openapi-mode = "ignore-privileges"',
                'log-level = "info"',
            ]
        ),
        encoding="utf-8",
    )


def test_loads_root_rotor_postgrest_config(tmp_path: Path) -> None:
    conf = tmp_path / "root_rotor_postgrest.conf"
    write_conf(conf)

    parsed = control.load_config(conf)

    assert parsed["server-host"] == "127.0.0.1"
    assert parsed["server-port"] == "3000"
    assert parsed["admin-server-port"] == "3001"


def test_status_reports_running_process(tmp_path: Path) -> None:
    conf = tmp_path / "root_rotor_postgrest.conf"
    pid_file = tmp_path / "04_RUNTIME" / "root_rotor_postgrest.pid"
    log_file = tmp_path / "04_RUNTIME" / "root_rotor_postgrest.log"
    write_conf(conf)

    proc = subprocess.Popen(["sleep", "5"])
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(str(proc.pid), encoding="utf-8")

    status = control.build_status(
        conf_path=conf,
        pid_path=pid_file,
        log_path=log_file,
        request_get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("not used")),
    )

    assert status["action"] == "status"
    assert status["pid"] == proc.pid
    assert status["pid_alive"] is True
    assert status["api"] == "http://127.0.0.1:3000"
    assert status["admin"] == "http://127.0.0.1:3001"

    proc.terminate()
    proc.wait(timeout=2)


def test_readiness_waits_for_admin_and_api(monkeypatch) -> None:
    conf = control.DEFAULT_CONFIG
    calls: list[str] = []

    def fake_get(url, timeout=0.25):
        calls.append(url)
        class Resp:
            status_code = 200

            def raise_for_status(self):
                pass

        if "ready" in url or "api_bible_manuals?limit=1" in url:
            return Resp()
        raise RuntimeError("unexpected")

    ready = control.wait_for_readiness(
        conf_path=conf,
        timeout_seconds=0.05,
        poll_seconds=0.01,
        request_get=fake_get,
    )

    assert ready["ready"] is True
    assert calls[0].endswith("/ready")
    assert calls[1].endswith("/api_bible_manuals?limit=1")


def test_start_writes_pid_when_command_invoked(tmp_path, monkeypatch) -> None:
    conf = tmp_path / "root_rotor_postgrest.conf"
    pid_path = tmp_path / "04_RUNTIME/root_rotor_postgrest.pid"
    log_path = tmp_path / "04_RUNTIME/root_rotor_postgrest.log"
    write_conf(conf)

    fake_proc = SimpleNamespace(pid=98765)
    start_calls: list[tuple] = []

    def fake_popen(cmd, cwd=None, stdin=None, stdout=None, stderr=None, start_new_session=None):
        start_calls.append((tuple(cmd), cwd, start_new_session))
        return fake_proc

    monkeypatch.setattr(control, "subprocess", SimpleNamespace(Popen=fake_popen))
    monkeypatch.setattr(control.shutil, "which", lambda _: str(tmp_path / "postgrest"))

    result = control.start_postgrest(
        conf_path=conf,
        pid_path=pid_path,
        log_path=log_path,
        wait_for_ready=False,
    )

    assert pid_path.read_text(encoding="utf-8") == "98765"
    assert result["pid_started"] == 98765
    assert start_calls and start_calls[0][0][0].endswith("postgrest")
    assert start_calls[0][2] is True


def test_stop_terminates_pid_and_removes_pidfile(tmp_path) -> None:
    conf = tmp_path / "root_rotor_postgrest.conf"
    pid_path = tmp_path / "04_RUNTIME/root_rotor_postgrest.pid"
    log_path = tmp_path / "04_RUNTIME/root_rotor_postgrest.log"
    write_conf(conf)

    proc = subprocess.Popen(["sleep", "60"])
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    result = control.stop_postgrest(pid_path=pid_path, grace_seconds=0.01)

    assert result["signal_sent"] is True
    assert result["pid"] == proc.pid
    assert not pid_path.exists()
    assert proc.poll() is not None


def test_main_supports_status_json_output(tmp_path) -> None:
    conf = tmp_path / "root_rotor_postgrest.conf"
    write_conf(conf)

    status = control.build_status(conf_path=conf)
    assert isinstance(control.json.dumps(status), str)
    assert status["action"] == "status"
