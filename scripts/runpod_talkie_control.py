#!/usr/bin/env python3
"""RunPod Talkie control shim.

Idempotent local controller for the Talkie weight path:
- plan: show exact SSH/SCP/remote commands
- probe: test auth and poll remote custody receipt when possible
- run: if SSH works, upload pack and start/poll remote Talkie bootstrap

No graph/canon writes. Dolphin/Mixtral are not touched.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "213.192.6.98"
DEFAULT_PORT = 40100
DEFAULT_USER = "root"
DEFAULT_KEY = Path.home() / ".ssh/id_ed25519"
DEFAULT_PACK = ROOT / "05_OUTPUTS/runpod/talkie_book_lora/talkie_book_lora_runpod_pack.tar.gz"
DEFAULT_RECEIPT = ROOT / "05_OUTPUTS/runpod/talkie_book_lora/runpod_talkie_control_latest.json"
DEFAULT_AUTH_CHANGED_RECEIPT = ROOT / "05_OUTPUTS/runpod/talkie_book_lora/public_key_auth_material_changed.json"
MODEL_ID = "talkie-lm/talkie-1930-13b-it"
FILENAME = "rl-refined.pt"
REMOTE_BASE = "/workspace/talkie_book_lora"
REMOTE_FORGE = "/workspace/talkie_forge"


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(s: str | Path) -> str:
    return shlex.quote(str(s))


def ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={args.timeout}",
        "-p", str(args.port),
        "-i", str(args.key),
        f"{args.user}@{args.host}",
    ]


def scp_base(args: argparse.Namespace) -> list[str]:
    return [
        "scp",
        "-P", str(args.port),
        "-i", str(args.key),
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={args.timeout}",
    ]


def plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    pack = Path(args.pack)
    pubkey_path = Path(args.key).with_suffix(Path(args.key).suffix + ".pub") if Path(args.key).suffix else Path(str(args.key) + ".pub")
    public_key = pubkey_path.read_text(encoding="utf-8").strip() if pubkey_path.exists() else ""
    remote_pack = f"{REMOTE_BASE}/{pack.name}"
    upload_cmd = " ".join(map(q, scp_base(args) + [str(pack), f"{args.user}@{args.host}:{remote_pack}"]))
    ssh_prefix = " ".join(map(q, ssh_base(args)))
    unpack_cmd = f"mkdir -p {q(REMOTE_BASE)} && cd {q(REMOTE_BASE)} && tar -xzf {q(remote_pack)}"
    start_download = (
        f"export MODEL_ID={q(MODEL_ID)} FILENAME={q(FILENAME)} WORKDIR={q(REMOTE_FORGE)}; "
        f"cd {q(REMOTE_BASE)}/talkie_book_lora_runpod_pack && bash scripts/runpod_talkie_forge_bootstrap.sh"
    )
    poll_receipt = f"tail -n 40 {q(REMOTE_FORGE)}/receipts/talkie_download.log 2>/dev/null || true; cat {q(REMOTE_FORGE)}/receipts/talkie_source_custody.json 2>/dev/null || true"
    return {
        "schema": "lucidota.runpod.talkie_control.plan.v1",
        "generated_at": now_z(),
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "key": str(args.key),
        "public_key": public_key,
        "pack": rel(pack) if pack.is_absolute() and pack.exists() else str(pack),
        "pack_sha256": sha_file(pack) if pack.exists() else "",
        "model_id": MODEL_ID,
        "filename": FILENAME,
        "remote_base": REMOTE_BASE,
        "remote_forge": REMOTE_FORGE,
        "local_commands": {
            "test_ssh": f"{ssh_prefix} {q('echo DIRECT_TCP_OK; hostname')}",
            "upload_pack": upload_cmd,
        },
        "remote_commands": {
            "unpack_pack": unpack_cmd,
            "start_download": start_download,
            "poll_receipt": poll_receipt,
        },
        "dolphin_touched": False,
    }


def write_receipt(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prior_public_key_auth_block(receipt_path: Path) -> bool:
    if not receipt_path.exists():
        return False
    try:
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    status = str(prior.get("status") or "")
    stderr = str(prior.get("ssh_stderr_tail") or "")
    if status in {"WAITING_FOR_PUBLIC_KEY_AUTH", "PUBLIC_KEY_AUTH_REQUIRED", "WAITING_FOR_AUTHORIZED_KEY"}:
        return True
    return "Permission denied" in stderr and ("publickey" in stderr or "Offering public key" in stderr)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def auth_material_changed_after_refusal(auth_receipt_path: Path, refusal_receipt_path: Path) -> bool:
    if not auth_receipt_path.exists() or not refusal_receipt_path.exists():
        return False
    try:
        auth = json.loads(auth_receipt_path.read_text(encoding="utf-8"))
        refusal = json.loads(refusal_receipt_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if auth.get("status") not in {"PUBLIC_KEY_SET", "AUTHORIZED_KEYS_CHANGED", "AUTH_MATERIAL_CHANGED"}:
        return False
    changed_raw = auth.get("changed_at") or auth.get("generated_at")
    refused_raw = refusal.get("refused_at") or refusal.get("generated_at")
    if not changed_raw or not refused_raw:
        return False
    changed_at = _parse_ts(changed_raw)
    refused_at = _parse_ts(refused_raw)
    if changed_at and refused_at:
        return changed_at > refused_at
    return str(changed_raw) > str(refused_raw)


def public_key_auth_payload(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lucidota.runpod.talkie_control.probe_receipt.v1",
        "generated_at": now_z(),
        "status": "WAITING_FOR_PUBLIC_KEY_AUTH",
        "host": args.host,
        "port": args.port,
        "pod_id": "bh5zizy4coj8a0",
        "proxy_ssh": "bh5zizy4coj8a0-64412392@ssh.runpod.io",
        "direct_tcp": f"{args.user}@{args.host}:{args.port}",
        "model_id": MODEL_ID,
        "filename": FILENAME,
        "pack": plan["pack"],
        "pack_sha256": plan["pack_sha256"],
        "public_key": plan["public_key"],
        "ssh_attempted": False,
        "scp_attempted": False,
        "network_attempted": False,
        "reason": "remote refused correct offered key; PUBLIC_KEY/authorized_keys must change before retry",
        "next_required_action": "Set PUBLIC_KEY env on runpod/pytorch pod or paste key into /root/.ssh/authorized_keys via web terminal, then restart/redeploy if env changed.",
        "next_allowed_retry": "force_after_auth_change_or_auth_material_changed_receipt",
        "retry_policy": "forbidden_until_public_key_or_authorized_keys_changed",
        "required_fix": {
            "env_var": "PUBLIC_KEY",
            "value": plan["public_key"],
            "secret": False,
            "restart_required": True,
            "alternative": "paste 05_OUTPUTS/runpod/talkie_book_lora/RUNPOD_WEB_TERMINAL_PASTE.sh into RunPod web terminal to append authorized_keys",
        },
        "secret_env_examples": {
            "HF_TOKEN": "{{ RUNPOD_SECRET_hf_token }}",
            "JUPYTER_PASSWORD": "{{ RUNPOD_SECRET_jupyter_password }}",
        },
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "dolphin_touched": False,
    }


def run_cmd(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def probe(args: argparse.Namespace) -> dict[str, Any]:
    plan = plan_payload(args)
    receipt_path = Path(args.receipt)
    no_network = bool(getattr(args, "no_network", False))
    force = bool(getattr(args, "force_after_auth_change", False))
    auth_changed_receipt = Path(getattr(args, "auth_material_changed_receipt", DEFAULT_AUTH_CHANGED_RECEIPT))
    if (
        not no_network
        and not force
        and prior_public_key_auth_block(receipt_path)
        and not auth_material_changed_after_refusal(auth_changed_receipt, receipt_path)
    ):
        payload = public_key_auth_payload(args, plan)
        write_receipt(payload, receipt_path)
        return payload
    payload: dict[str, Any] = {
        "schema": "lucidota.runpod.talkie_control.probe_receipt.v1",
        "generated_at": now_z(),
        "status": "NOT_PROBED_NO_NETWORK" if no_network else "UNKNOWN",
        "host": args.host,
        "port": args.port,
        "model_id": MODEL_ID,
        "filename": FILENAME,
        "pack": plan["pack"],
        "pack_sha256": plan["pack_sha256"],
        "public_key": plan["public_key"],
        "db_writes_performed": False,
        "graph_writes_performed": False,
        "dolphin_touched": False,
    }
    if not no_network:
        cmd = ssh_base(args) + ["echo DIRECT_TCP_OK; hostname; test -f /workspace/talkie_forge/receipts/talkie_source_custody.json && cat /workspace/talkie_forge/receipts/talkie_source_custody.json || true"]
        cp = run_cmd(cmd, timeout=args.timeout + 5)
        payload.update({
            "ssh_returncode": cp.returncode,
            "ssh_stdout_tail": cp.stdout[-4000:],
            "ssh_stderr_tail": cp.stderr[-2000:],
        })
        if cp.returncode == 0:
            payload["status"] = "SSH_OK"
            if "talkie_source_custody" in cp.stdout or '"schema"' in cp.stdout:
                payload["remote_custody_seen"] = True
        else:
            payload["status"] = "WAITING_FOR_PUBLIC_KEY_AUTH" if "Permission denied" in cp.stderr else "SSH_FAILED"
            if payload["status"] == "WAITING_FOR_PUBLIC_KEY_AUTH":
                payload.update(public_key_auth_payload(args, plan))
                payload["network_attempted"] = True
                payload["ssh_attempted"] = True
                payload["scp_attempted"] = False
                payload["ssh_returncode"] = cp.returncode
                payload["ssh_stdout_tail"] = cp.stdout[-4000:]
                payload["ssh_stderr_tail"] = cp.stderr[-2000:]
    write_receipt(payload, Path(args.receipt))
    return payload


def execute_run(args: argparse.Namespace) -> dict[str, Any]:
    first = probe(args)
    if first.get("status") != "SSH_OK":
        return first
    pack = Path(args.pack)
    remote_pack = f"{args.user}@{args.host}:{REMOTE_BASE}/{pack.name}"
    mkdir = ssh_base(args) + [f"mkdir -p {q(REMOTE_BASE)}"]
    cp = run_cmd(mkdir, timeout=args.timeout + 5)
    if cp.returncode != 0:
        first.update({"status": "REMOTE_MKDIR_FAILED", "stderr": cp.stderr[-2000:]})
        write_receipt(first, Path(args.receipt))
        return first
    up = run_cmd(scp_base(args) + [str(pack), remote_pack], timeout=max(60, args.timeout + 30))
    if up.returncode != 0:
        first.update({"status": "SCP_FAILED", "stderr": up.stderr[-2000:]})
        write_receipt(first, Path(args.receipt))
        return first
    plan = plan_payload(args)
    remote = ssh_base(args) + [plan["remote_commands"]["unpack_pack"] + "; " + plan["remote_commands"]["start_download"]]
    start = run_cmd(remote, timeout=max(120, args.timeout + 90))
    first.update({
        "status": "REMOTE_START_ATTEMPTED" if start.returncode == 0 else "REMOTE_START_FAILED",
        "scp_uploaded": True,
        "remote_start_returncode": start.returncode,
        "remote_start_stdout_tail": start.stdout[-4000:],
        "remote_start_stderr_tail": start.stderr[-4000:],
    })
    write_receipt(first, Path(args.receipt))
    return first


def emit(payload: dict[str, Any], json_out: bool) -> None:
    if json_out:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="runpod-talkie-control")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--user", default=DEFAULT_USER)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--force-after-auth-change", action="store_true", help="allow one network probe after operator confirms PUBLIC_KEY/authorized_keys changed")
    ap.add_argument("--auth-material-changed-receipt", default=str(DEFAULT_AUTH_CHANGED_RECEIPT), help="local receipt proving PUBLIC_KEY/authorized_keys changed after refusal")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["plan", "probe", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--json", action="store_true")
        if name == "probe":
            p.add_argument("--no-network", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "plan":
        payload = plan_payload(args)
    elif args.cmd == "probe":
        payload = probe(args)
    else:
        payload = execute_run(args)
    emit(payload, bool(getattr(args, "json", False)))
    return 0 if payload.get("status") not in {"SCP_FAILED", "REMOTE_START_FAILED", "REMOTE_MKDIR_FAILED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
