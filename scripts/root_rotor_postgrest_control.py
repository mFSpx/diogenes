#!/usr/bin/env python3
"""Tiny deterministic Root-Rotor PostgREST control helper."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "GOALS" / "root_rotor_postgrest.conf"
DEFAULT_PID_FILE = ROOT / "04_RUNTIME" / "root_rotor_postgrest.pid"
DEFAULT_LOG_FILE = ROOT / "04_RUNTIME" / "root_rotor_postgrest.log"
REQUEST_TIMEOUT = 0.5


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(conf_path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    conf: dict[str, str] = {}
    for raw in conf_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        conf[key] = value.strip().strip('"').strip("'")
    return conf


def build_urls(conf: dict[str, str]) -> tuple[str, str]:
    host = conf.get("server-host", "127.0.0.1").strip("/")
    api_port = conf.get("server-port", "3000")
    admin_port = conf.get("admin-server-port", "3001")
    return (
        f"http://{host}:{api_port}",
        f"http://{host}:{admin_port}",
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def probe(url: str, *, timeout: float = REQUEST_TIMEOUT, request_get: Callable[..., Any] = requests.get) -> tuple[bool, str | None]:
    try:
        response = request_get(url, timeout=timeout)
        if response.status_code == 200:
            return True, None
        return False, f"{response.status_code}"
    except Exception as exc:  # pragma: no cover - deterministic on failure path
        return False, type(exc).__name__


def wait_for_readiness(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.25,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    conf = load_config(conf_path)
    api_base, admin_base = build_urls(conf)
    admin_url = f"{admin_base}/ready"
    api_url = f"{api_base}/api_bible_manuals?limit=1"

    deadline = time.monotonic() + timeout_seconds
    last_admin: tuple[bool, str | None] = (False, "not_checked")
    last_api: tuple[bool, str | None] = (False, "not_checked")

    while True:
        admin_ok, admin_err = probe(admin_url, request_get=request_get)
        api_ok, api_err = probe(api_url, request_get=request_get)
        last_admin = (admin_ok, admin_err)
        last_api = (api_ok, api_err)
        if admin_ok and api_ok:
            return {
                "schema": "lucidota.root_rotor.postgrest.readiness.v1",
                "generated_at": now(),
                "ready": True,
                "admin_ready": True,
                "api_ready": True,
                "admin_url": admin_base,
                "api_url": api_base,
                "admin_last_error": None,
                "api_last_error": None,
            }
        if time.monotonic() >= deadline:
            return {
                "schema": "lucidota.root_rotor.postgrest.readiness.v1",
                "generated_at": now(),
                "ready": False,
                "admin_ready": admin_ok,
                "api_ready": api_ok,
                "admin_url": admin_base,
                "api_url": api_base,
                "admin_last_error": last_admin[1],
                "api_last_error": last_api[1],
            }
        time.sleep(max(0.0, poll_seconds))


def build_status(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    check_readiness: bool = False,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    conf = load_config(conf_path)
    api_url, admin_url = build_urls(conf)
    pid = read_pid(pid_path)
    status: dict[str, Any] = {
        "schema": "lucidota.root_rotor.postgrest.status.v1",
        "generated_at": now(),
        "action": "status",
        "config": rel(conf_path),
        "pid_file": rel(pid_path),
        "log_file": rel(log_path),
        "pid": pid,
        "pid_alive": is_alive(pid),
        "api": api_url,
        "admin": admin_url,
        "postgrest_available": shutil.which("postgrest") is not None,
    }
    if check_readiness:
        status.update(wait_for_readiness(conf_path=conf_path, request_get=request_get))
        status["action"] = "status_readiness"
    return status


def start_postgrest(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    wait_for_ready: bool = False,
    readiness_timeout: float = 30.0,
    readiness_poll: float = 0.25,
    request_get: Callable[..., Any] = requests.get,
    ) -> dict[str, Any]:
    binary = shutil.which("postgrest")
    if not binary:
        result = {
            "schema": "lucidota.root_rotor.postgrest.control.v1",
            "generated_at": now(),
            "action": "start",
            "command": "postgrest",
            "error": "postgrest_binary_missing",
        }
        result.update(build_status(conf_path=conf_path, pid_path=pid_path, log_path=log_path))
        return result

    existing_pid = read_pid(pid_path)
    if is_alive(existing_pid):
        result = build_status(conf_path=conf_path, pid_path=pid_path, log_path=log_path)
        result.update({
            "schema": "lucidota.root_rotor.postgrest.control.v1",
            "generated_at": now(),
            "action": "start",
            "already_running": True,
        })
        if wait_for_ready:
            result["readiness"] = wait_for_readiness(
                conf_path=conf_path,
                timeout_seconds=readiness_timeout,
                poll_seconds=readiness_poll,
                request_get=request_get,
            )
        return result

    pid_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdin = subprocess.DEVNULL if hasattr(subprocess, "DEVNULL") else open(os.devnull, "rb")
    proc = subprocess.Popen(
        [binary, str(conf_path)],
        cwd=ROOT,
        stdin=stdin,
        stdout=log_path.open("ab"),
        stderr=getattr(subprocess, "STDOUT", None),
        start_new_session=True,
    )
    pid_path.write_text(str(proc.pid), encoding="utf-8")

    result: dict[str, Any] = {
        "schema": "lucidota.root_rotor.postgrest.control.v1",
        "generated_at": now(),
        "action": "start",
        "command": "postgrest",
        "command_arg": str(conf_path),
        "pid_started": proc.pid,
        "log_file": rel(log_path),
    }
    if wait_for_ready:
        result["readiness"] = wait_for_readiness(
            conf_path=conf_path,
            timeout_seconds=readiness_timeout,
            poll_seconds=readiness_poll,
            request_get=request_get,
        )
    return result


def start(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    wait_for_ready: bool = False,
    readiness_timeout: float = 30.0,
    readiness_poll: float = 0.25,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    return start_postgrest(
        conf_path=conf_path,
        pid_path=pid_path,
        log_path=log_path,
        wait_for_ready=wait_for_ready,
        readiness_timeout=readiness_timeout,
        readiness_poll=readiness_poll,
        request_get=request_get,
    )


def stop_postgrest(
    *,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    grace_seconds: float = 0.25,
) -> dict[str, Any]:
    pid = read_pid(pid_path)
    alive = is_alive(pid)
    result: dict[str, Any] = {
        "schema": "lucidota.root_rotor.postgrest.control.v1",
        "generated_at": now(),
        "action": "stop",
        "pid_file": rel(pid_path),
        "log_file": rel(log_path),
        "pid": pid,
        "pid_alive": alive,
        "signal_sent": False,
        "sigkill_sent": False,
    }
    if not pid or not alive:
        if pid_path.exists():
            pid_path.unlink()
        return result

    os.kill(pid, signal.SIGTERM)
    result["signal_sent"] = True
    deadline = time.monotonic() + grace_seconds
    while is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if is_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
            result["sigkill_sent"] = True
        except OSError:
            pass
    if pid_path.exists():
        pid_path.unlink()
    return result


def stop(
    *,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    grace_seconds: float = 0.25,
) -> dict[str, Any]:
    return stop_postgrest(pid_path=pid_path, log_path=log_path, grace_seconds=grace_seconds)


def status(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    pid_path: Path = DEFAULT_PID_FILE,
    log_path: Path = DEFAULT_LOG_FILE,
    check_readiness: bool = False,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    return build_status(
        conf_path=conf_path,
        pid_path=pid_path,
        log_path=log_path,
        check_readiness=check_readiness,
        request_get=request_get,
    )


def readiness(
    *,
    conf_path: Path = DEFAULT_CONFIG,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.25,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    return wait_for_readiness(
        conf_path=conf_path,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        request_get=request_get,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Root-Rotor PostgREST control wrapper")
    parser.add_argument("command", choices=("start", "stop", "status", "readiness"))
    parser.add_argument("--conf", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pid", default=str(DEFAULT_PID_FILE))
    parser.add_argument("--log", default=str(DEFAULT_LOG_FILE))
    parser.add_argument("--wait", type=float, default=30.0)
    parser.add_argument("--poll", type=float, default=0.25)
    parser.add_argument("--check-readiness", action="store_true")
    args = parser.parse_args()

    conf_path = Path(args.conf)
    pid_path = Path(args.pid)
    log_path = Path(args.log)

    if args.command == "start":
        report = start_postgrest(
            conf_path=conf_path,
            pid_path=pid_path,
            log_path=log_path,
            wait_for_ready=args.check_readiness,
            readiness_timeout=args.wait,
            readiness_poll=args.poll,
        )
    elif args.command == "stop":
        report = stop_postgrest(pid_path=pid_path, log_path=log_path)
    elif args.command == "readiness":
        report = wait_for_readiness(conf_path=conf_path, timeout_seconds=args.wait, poll_seconds=args.poll)
    else:
        report = build_status(
            conf_path=conf_path,
            pid_path=pid_path,
            log_path=log_path,
            check_readiness=args.check_readiness,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
