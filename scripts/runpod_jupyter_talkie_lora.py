#!/usr/bin/env python3
"""RunPod Talkie/LoRA launcher via Jupyter API (non-SSH).

It can launch a bounded readiness check or a bounded LoRA smoke command and
optionally read back the last remote status receipt path.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode, urlsplit, urlunsplit, parse_qs

import requests

try:
    import websockets
except Exception:  # pragma: no cover
    websockets = None  # type: ignore

SCHEMA = "lucidota.runpod_jupyter_talkie_lora.v1"
SCHEMA_REMOTE = "lucidota.runpod_jupyter_talkie_lora.remote_status.v1"
DEFAULT_RECEIPT = Path("05_OUTPUTS/runpod/runpod_jupyter_talkie_lora_launcher.json")
DEFAULT_SOURCE_RECEIPT = Path("05_OUTPUTS/runpod/talkie_book_lora/remote_talkie_source_custody.json")
DEFAULT_MOE_READINESS_MANIFEST = Path("04_RUNTIME/TALKIE_MOE/talkie_moe_readiness_manifest.json")
DEFAULT_SMOKE_START_RECEIPT = Path("05_OUTPUTS/runpod/talkie_book_lora/talkie_load_smoke_start.json")
DEFAULT_READINESS_SCRIPT = "/workspace/talkie_forge/talkie_load_smoke.py"
DEFAULT_TRAIN_SCRIPT = "/workspace/talkie_book_lora/talkie_book_lora_runpod_pack/scripts/runpod_book_reader_lora_train.py"
DEFAULT_REMOTE_STATUS_PATH = {
    "readiness": "/workspace/talkie_forge/receipts/talkie_jupyter_readiness_receipt.json",
    "trainer_smoke": "/workspace/talkie_forge/receipts/talkie_jupyter_train_smoke_receipt.json",
    "custom": "/workspace/talkie_forge/receipts/talkie_jupyter_custom_receipt.json",
}
DEFAULT_TIMEOUT = 120
DEFAULT_WEB_TIMEOUT = 20
DEFAULT_MAX_OUT_CHARS = 6000


class JupyterProtocolError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    status: str
    exit_code: int | None
    output: dict[str, Any]
    kernel_id: str | None = None


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def redaction(text: str, token: str | None) -> str:
    if not token:
        return text
    return text.replace(token, "<REDACTED>")


def redacted_url(url: str | None) -> str | None:
    if not url:
        return None
    p = urlsplit(url)
    q = parse_qs(p.query, keep_blank_values=True)
    q.pop("token", None)
    query = urlencode({k: v[0] for k, v in q.items()}, doseq=True)
    return urlunsplit((p.scheme, p.netloc, p.path, query, p.fragment))


def _api_headers(token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def quoted_remote_path(remote_path: str) -> str:
    return quote(remote_path.lstrip("/"), safe="/")


def build_contents_url(base_url: str, remote_path: str, *, download: bool = False) -> str:
    out = f"{base_url.rstrip('/')}/api/contents/{quoted_remote_path(remote_path)}"
    if download:
        out += "?download=1"
    return out


def build_kernels_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/kernels"


def build_channels_url(base_url: str, kernel_id: str, token: str | None) -> str:
    if base_url.startswith("https://"):
        scheme = "wss://"
        netloc = base_url[len("https://"):]
    elif base_url.startswith("http://"):
        scheme = "ws://"
        netloc = base_url[len("http://"):]
    else:
        raise ValueError("jupyter_url must start with http:// or https://")

    base = f"{scheme}{netloc.rstrip('/')}/api/kernels/{kernel_id}/channels"
    if token:
        return f"{base}?token={quote(token)}"
    return base


def read_json_receipt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_context(
    *,
    source_receipt: Path = DEFAULT_SOURCE_RECEIPT,
    readiness_manifest: Path = DEFAULT_MOE_READINESS_MANIFEST,
    load_smoke_start: Path = DEFAULT_SMOKE_START_RECEIPT,
) -> dict[str, Any]:
    custody = read_json_receipt(source_receipt)
    readiness = read_json_receipt(readiness_manifest)
    smoke_start = read_json_receipt(load_smoke_start)
    return {
        "base_model_id": custody.get("model_id", "talkie-lm/talkie-1930-13b-it"),
        "checkpoint_path": custody.get("path", ""),
        "selected_file": custody.get("selected_file", ""),
        "source_receipt": str(source_receipt),
        "readiness_script": smoke_start.get("script", readiness.get("runpod_paths", {}).get("receipts", DEFAULT_READINESS_SCRIPT)),
        "train_script": DEFAULT_TRAIN_SCRIPT,
        "readiness_receipt": smoke_start.get("log", "/workspace/talkie_forge/receipts/talkie_load_smoke.json"),
    }


def build_command(
    *,
    mode: str,
    context: dict[str, Any],
    override_command: str | None,
    max_train_steps: int,
) -> str:
    if override_command:
        return override_command

    if mode == "readiness":
        return f"python3 {shlex.quote(context.get('readiness_script', DEFAULT_READINESS_SCRIPT))}"

    if mode == "trainer_smoke":
        train_script = context.get("train_script", DEFAULT_TRAIN_SCRIPT)
        env = {
            "TARGET": "talkie",
            "BASE_MODEL": context.get("base_model_id", "talkie-lm/talkie-1930-13b-it"),
            "MAX_TRAINING_STEPS": str(max_train_steps),
            "EPOCHS": "0.01",
            "BATCH_SIZE": "1",
            "GRAD_ACCUM": "1",
        }
        prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items() if v)
        return f"{prefix} python3 {shlex.quote(train_script)}"

    return override_command or ""


def build_runner_code(command_to_run: str, status_path: str, max_output_chars: int) -> str:
    payload = """
import json
import subprocess
import time
import traceback

command = COMMAND
status_path = STATUS_PATH
start = time.time()
try:
    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        text=True,
        capture_output=True,
        check=False,
    )
    report = {
        "schema": REMOTE_SCHEMA,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": int(result.returncode),
        "stdout": (result.stdout or "")[ -MAX_OUT :],
        "stderr": (result.stderr or "")[ -MAX_OUT :],
        "command": command,
        "started_at": start,
        "finished_at": time.time(),
        "duration_s": round(time.time() - start, 4),
    }
except Exception as exc:
    report = {
        "schema": REMOTE_SCHEMA,
        "status": "FAIL",
        "returncode": 1,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "command": command,
        "started_at": start,
        "finished_at": time.time(),
        "duration_s": round(time.time() - start, 4),
    }

with open(status_path, "w", encoding="utf-8") as fp:
    fp.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
print(json.dumps(report))
""".strip()
    return payload.replace("COMMAND", json.dumps(command_to_run)).replace("STATUS_PATH", json.dumps(status_path)).replace("MAX_OUT", str(max_output_chars)).replace("REMOTE_SCHEMA", json.dumps(SCHEMA_REMOTE))


def create_kernel(jupyter_url: str, token: str | None, *, request_post: Callable[..., Any] = requests.post) -> str:
    response = request_post(build_kernels_url(jupyter_url), headers=_api_headers(token), json={"name": "python3"}, timeout=DEFAULT_WEB_TIMEOUT)
    if response.status_code != 201:
        raise JupyterProtocolError(f"kernel_create_failed:{response.status_code}")
    data = response.json()
    kernel_id = data.get("id")
    if not isinstance(kernel_id, str):
        raise JupyterProtocolError("kernel_create_missing_id")
    return kernel_id


def delete_kernel(jupyter_url: str, token: str | None, kernel_id: str, *, request_delete: Callable[..., Any] = requests.delete) -> None:
    request_delete(f"{build_kernels_url(jupyter_url)}/{kernel_id}", headers=_api_headers(token), timeout=DEFAULT_WEB_TIMEOUT)


async def _execute_over_websocket(
    jupyter_url: str,
    token: str | None,
    kernel_id: str,
    code: str,
    *,
    timeout: int,
) -> ExecutionResult:
    if websockets is None:
        raise JupyterProtocolError("websockets_not_available")

    msg_id = uuid.uuid4().hex
    session_id = uuid.uuid4().hex
    ws_url = build_channels_url(jupyter_url, kernel_id, token)
    request = {
        "header": {
            "msg_id": msg_id,
            "msg_type": "execute_request",
            "session": session_id,
            "username": "lucidota",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    error: str | None = None
    exit_code: int | None = None

    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps(request))
        deadline = time.time() + timeout
        while True:
            timeout_remaining = max(1.0, deadline - time.time())
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_remaining)
            msg = json.loads(raw)
            header = msg.get("header") or {}
            parent = (msg.get("parent_header") or {}).get("msg_id")
            if parent != msg_id:
                continue

            msg_type = header.get("msg_type")
            content = msg.get("content") or {}
            if msg_type == "stream":
                text = content.get("text") or ""
                if content.get("name") == "stderr":
                    stderr_parts.append(str(text))
                else:
                    stdout_parts.append(str(text))
            elif msg_type == "error":
                tb = content.get("traceback") or []
                error = "; ".join(tb) if isinstance(tb, list) else str(tb)
                exit_code = 1
            elif msg_type == "status" and content.get("execution_state") == "idle" and exit_code is not None:
                break
            elif msg_type == "execute_reply":
                if content.get("status") == "ok":
                    exit_code = 0
                else:
                    exit_code = 1
                    error = error or str(content.get("evalue") or content.get("ename") or "execution_error")
                break

            if time.time() >= deadline:
                return ExecutionResult(
                    status="TIMEOUT",
                    exit_code=None,
                    output={"stdout": "".join(stdout_parts), "stderr": "".join(stderr_parts), "error": "execution_timeout"},
                    kernel_id=kernel_id,
                )

    return ExecutionResult(
        status="PASS" if exit_code == 0 else "FAIL",
        exit_code=exit_code,
        output={"stdout": "".join(stdout_parts)[-DEFAULT_MAX_OUT_CHARS:], "stderr": "".join(stderr_parts)[-DEFAULT_MAX_OUT_CHARS:], "error": error},
        kernel_id=kernel_id,
    )


def run_remote(
    jupyter_url: str,
    jupyter_token: str | None,
    command: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_output_chars: int = DEFAULT_MAX_OUT_CHARS,
    remote_status_path: str = "/tmp/talkie_jupyter_launcher_status.json",
    request_post: Callable[..., Any] = requests.post,
    request_delete: Callable[..., Any] = requests.delete,
    request_get: Callable[..., Any] = requests.get,
) -> ExecutionResult:
    del request_get  # currently unused; reserved for future async status prefetching.

    kernel_id = create_kernel(jupyter_url, jupyter_token, request_post=request_post)
    try:
        payload = build_runner_code(command, status_path=remote_status_path, max_output_chars=max_output_chars)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                _execute_over_websocket(
                    jupyter_url,
                    jupyter_token,
                    kernel_id,
                    payload,
                    timeout=timeout,
                )
            )
        finally:
            loop.close()
        return result
    finally:
        try:
            delete_kernel(jupyter_url, jupyter_token, kernel_id, request_delete=request_delete)
        except Exception:
            pass


def fetch_remote_receipt(jupyter_url: str, token: str | None, remote_path: str, *, request_get: Callable[..., Any] = requests.get) -> dict[str, Any]:
    response = request_get(build_contents_url(jupyter_url, remote_path, download=True), headers=_api_headers(token), timeout=DEFAULT_WEB_TIMEOUT)
    if response.status_code != 200:
        return {"status": "UNREADABLE", "error": f"fetch_failed_{response.status_code}"}

    try:
        payload = response.json()
    except Exception:
        return {"status": "FAILED_PARSE", "error": "response_is_not_json"}

    if isinstance(payload, dict) and payload.get("type") == "file":
        raw = payload.get("content", "")
        try:
            text = base64.b64decode(raw).decode("utf-8")
            return json.loads(text)
        except Exception:
            return {"status": "FAILED_PARSE", "error": "base64_or_json_decode_failed"}

    if isinstance(payload, dict):
        return payload
    return {"status": "UNSUPPORTED_FORMAT"}


def build_receipt(
    *,
    status: str,
    mode: str,
    command: str,
    jupyter_url: str | None,
    jupyter_token: str | None,
    context: dict[str, Any],
    requested_receipt: str,
    remote_receipt: str,
    exit_code: int | None = None,
    dry_run: bool = False,
    timeout_seconds: int | None = None,
    status_output: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "mode": mode,
        "dry_run": dry_run,
        "generated_at": now_z(),
        "jupyter_url": redacted_url(jupyter_url),
        "command": redaction(command, jupyter_token),
        "command_redacted": bool(jupyter_token),
        "exit_code": exit_code,
        "requested_receipt": requested_receipt,
        "remote_receipt": remote_receipt,
        "timeout_seconds": timeout_seconds,
        "context": {
            "source_receipt": context.get("source_receipt"),
            "base_model_id": context.get("base_model_id"),
            "checkpoint_path": context.get("checkpoint_path"),
            "selected_file": context.get("selected_file"),
        },
        "status_output": status_output,
        "error": error,
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "dolphin_touched": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RunPod Talkie/LoRA non-SSH launcher")
    ap.add_argument("--jupyter-url", help="RunPod/Jupyter base URL (or env JUPYTER_URL / RUNPOD_JUPYTER_URL)")
    ap.add_argument("--jupyter-token", help="RunPod/Jupyter API token (or env JUPYTER_TOKEN / RUNPOD_JUPYTER_TOKEN)")
    ap.add_argument("--mode", choices=["readiness", "trainer_smoke", "custom"], default="readiness")
    ap.add_argument("--command", help="Custom command for --mode custom")
    ap.add_argument("--max-train-steps", type=int, default=1)
    ap.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    ap.add_argument("--remote-status-receipt", default=None, help="Remote Jupyter receipt path")
    ap.add_argument("--source-receipt", type=Path, default=DEFAULT_SOURCE_RECEIPT)
    ap.add_argument("--moe-readiness-manifest", type=Path, default=DEFAULT_MOE_READINESS_MANIFEST)
    ap.add_argument("--load-smoke-start", type=Path, default=DEFAULT_SMOKE_START_RECEIPT)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="Fetch existing remote status receipt")
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.jupyter_url:
        args.jupyter_url = os.environ.get("JUPYTER_URL") or os.environ.get("RUNPOD_JUPYTER_URL")
    if not args.jupyter_token:
        args.jupyter_token = os.environ.get("JUPYTER_TOKEN") or os.environ.get("RUNPOD_JUPYTER_TOKEN")

    remote_status_path = args.remote_status_receipt or DEFAULT_REMOTE_STATUS_PATH[args.mode]
    context = build_context(
        source_receipt=args.source_receipt,
        readiness_manifest=args.moe_readiness_manifest,
        load_smoke_start=args.load_smoke_start,
    )
    command = build_command(mode=args.mode, context=context, override_command=args.command, max_train_steps=args.max_train_steps)

    if args.status:
        if args.no_network or not args.jupyter_url:
            payload = build_receipt(
                status="BLOCKED_NO_NETWORK" if args.no_network else "BLOCKED_MISSING_JUPYTER_URL",
                mode=args.mode,
                command=command,
                jupyter_url=args.jupyter_url,
                jupyter_token=args.jupyter_token,
                context=context,
                requested_receipt=str(args.receipt),
                remote_receipt=remote_status_path,
                error="status requested but unavailable",
                status_output={"runnable_command": f"JUPYTER_URL=<JUPYTER_URL> python3 scripts/runpod_jupyter_talkie_lora.py --status --mode {args.mode}"},
            )
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if args.json:
                print(json.dumps(payload, sort_keys=True))
            return 1

        status_output = fetch_remote_receipt(args.jupyter_url, args.jupyter_token, remote_status_path)
        payload = build_receipt(
            status="PASS" if status_output.get("status") == "PASS" else "FAIL",
            mode=args.mode,
            command=command,
            jupyter_url=args.jupyter_url,
            jupyter_token=args.jupyter_token,
            context=context,
            requested_receipt=str(args.receipt),
            remote_receipt=remote_status_path,
            status_output=status_output,
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        return 0 if status_output.get("status") == "PASS" else 1

    if not args.jupyter_url:
        payload = build_receipt(
            status="BLOCKED_MISSING_JUPYTER_URL",
            mode=args.mode,
            command=command,
            jupyter_url=args.jupyter_url,
            jupyter_token=args.jupyter_token,
            context=context,
            requested_receipt=str(args.receipt),
            remote_receipt=remote_status_path,
            error="missing jupyter URL",
            status_output={"runnable_command": f"JUPYTER_URL=<JUPYTER_URL> python3 scripts/runpod_jupyter_talkie_lora.py --mode {args.mode} --command {shlex.quote(command)}"},
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        return 1

    if args.dry_run or args.no_network:
        if args.no_network:
            status = "BLOCKED_NO_NETWORK"
            error = "--no-network set"
        else:
            status = "DRY_RUN"
            error = None

        payload = build_receipt(
            status=status,
            mode=args.mode,
            command=command,
            jupyter_url=args.jupyter_url,
            jupyter_token=args.jupyter_token,
            context=context,
            requested_receipt=str(args.receipt),
            remote_receipt=remote_status_path,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout,
            status_output={"runnable_command": f"python3 scripts/runpod_jupyter_talkie_lora.py --mode {args.mode} --jupyter-url {shlex.quote(args.jupyter_url)}"},
            error=error,
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        return 0 if status == "DRY_RUN" else 1

    result = run_remote(
        jupyter_url=args.jupyter_url,
        jupyter_token=args.jupyter_token,
        command=command,
        timeout=args.timeout,
        remote_status_path=remote_status_path,
    )

    status = "PASS" if result.exit_code == 0 else "FAIL"
    payload = build_receipt(
        status=status,
        mode=args.mode,
        command=command,
        jupyter_url=args.jupyter_url,
        jupyter_token=args.jupyter_token,
        context=context,
        exit_code=result.exit_code,
        requested_receipt=str(args.receipt),
        remote_receipt=remote_status_path,
        timeout_seconds=args.timeout,
        status_output=result.output,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
