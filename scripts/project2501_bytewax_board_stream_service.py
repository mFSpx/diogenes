#!/usr/bin/env python3
"""Install/verify durable user service for the Project2501 Bytewax board stream."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "project2501_board_stream"
UNIT_NAME = "project2501-bytewax-board-stream.service"
DEFAULT_UNIT_DIR = Path.home() / ".config" / "systemd" / "user"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def render_unit(*, interval: int = 5, limit: int = 100) -> str:
    interval = max(1, int(interval))
    limit = max(1, min(1000, int(limit)))
    return f"""[Unit]
Description=LUCIDOTA Project2501 Bytewax board stream live cursor
After=default.target

[Service]
Type=simple
WorkingDirectory={ROOT}
Environment=PROJECT2501_BOARD_STREAM_INTERVAL={interval}
ExecStart=/usr/bin/bash -lc 'while true; do {sys.executable} scripts/project2501_bytewax_board_stream.py once --execute --live-cursor --limit {limit} --json; sleep "${{PROJECT2501_BOARD_STREAM_INTERVAL:-{interval}}}"; done'
Restart=always
RestartSec=5s

[Install]
WantedBy=default.target
"""


def run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_bytewax_board_stream_service_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def write_unit_file(*, unit_dir: Path = DEFAULT_UNIT_DIR, execute: bool = False, interval: int = 5, limit: int = 100) -> dict[str, Any]:
    unit_dir = Path(unit_dir)
    unit_path = unit_dir / UNIT_NAME
    unit_text = render_unit(interval=interval, limit=limit)
    payload = {
        "schema": "lucidota.project2501.bytewax_board_stream.service.v1",
        "unit_name": UNIT_NAME,
        "unit_path": str(unit_path),
        "unit_sha256": __import__("hashlib").sha256(unit_text.encode()).hexdigest(),
        "execute_performed": bool(execute),
        "unit_written": False,
        "systemctl_commands": [],
        "canonical_graph_writes_performed": False,
        "blockers": [],
        "status": "PASS",
    }
    if execute:
        try:
            unit_dir.mkdir(parents=True, exist_ok=True)
            unit_path.write_text(unit_text, encoding="utf-8")
            payload["unit_written"] = True
        except Exception as exc:
            payload["blockers"].append(f"write_unit_failed:{type(exc).__name__}:{exc}")
            payload["status"] = "FAIL"
    return payload


def install(args: argparse.Namespace) -> int:
    payload = write_unit_file(unit_dir=Path(args.unit_dir), execute=bool(args.execute), interval=args.interval, limit=args.limit)
    if args.execute and payload["status"] == "PASS" and args.systemctl:
        command_plan = [
            (["systemctl", "--user", "stop", UNIT_NAME], False),
            (["systemctl", "--user", "daemon-reload"], True),
            (["systemctl", "--user", "enable", UNIT_NAME], True),
            (["systemctl", "--user", "restart", UNIT_NAME], True),
            (["systemctl", "--user", "is-enabled", UNIT_NAME], True),
            (["systemctl", "--user", "is-active", UNIT_NAME], True),
        ]
        for cmd, required in command_plan:
            res = run(cmd)
            payload["systemctl_commands"].append(res)
            if required and res["returncode"] != 0:
                payload["blockers"].append(f"systemctl_failed:{' '.join(cmd)}")
                payload["status"] = "FAIL"
        if not any(r["cmd"][-2:] == ["is-active", UNIT_NAME] and r["returncode"] == 0 and "active" in r["stdout"] for r in payload["systemctl_commands"]):
            payload["blockers"].append("service_not_active")
            payload["status"] = "FAIL"
    write_report(payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    print("PROJECT2501_BYTEWAX_BOARD_STREAM_SERVICE=" + payload["status"])
    return 0 if payload["status"] == "PASS" else 4


def status(args: argparse.Namespace) -> int:
    cmds = [["systemctl", "--user", "is-enabled", UNIT_NAME], ["systemctl", "--user", "is-active", UNIT_NAME], ["systemctl", "--user", "status", UNIT_NAME, "--no-pager", "--lines=12"]]
    results = [run(cmd) for cmd in cmds]
    active = results[1]["returncode"] == 0 and "active" in results[1]["stdout"]
    payload = {
        "schema": "lucidota.project2501.bytewax_board_stream.service_status.v1",
        "unit_name": UNIT_NAME,
        "active": active,
        "results": results,
        "execute_performed": False,
        "canonical_graph_writes_performed": False,
        "status": "PASS" if active else "FAIL",
        "blockers": [] if active else ["service_not_active"],
    }
    write_report(payload)
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    print("PROJECT2501_BYTEWAX_BOARD_STREAM_SERVICE_STATUS=" + payload["status"])
    return 0 if active else 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Install/status durable Project2501 Bytewax board stream user service.")
    sub = p.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("install")
    ins.add_argument("--execute", action="store_true")
    ins.add_argument("--systemctl", action="store_true")
    ins.add_argument("--unit-dir", default=str(DEFAULT_UNIT_DIR))
    ins.add_argument("--interval", type=int, default=5)
    ins.add_argument("--limit", type=int, default=100)
    ins.add_argument("--json", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "install":
        return install(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
