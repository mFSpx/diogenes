#!/usr/bin/env python3
"""INDY_READs queued communications controller.

No network sends happen here. Email/Signal messages are queued as receipts and
require a later explicit operator-approved sender.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def queue_email(manifest: dict[str, Any], outbox: Path, subject: str, body: str) -> dict[str, Any]:
    email = manifest["email"]
    recipients = list(email["recipients"])
    subject = subject[: int(email.get("subject_max_chars", 120))]
    body = body[: int(email.get("body_max_chars", 4000))]
    record = {
        "schema": "lucidota.indy_reads.queued_email.v1",
        "queued_at": now_z(),
        "persona": manifest.get("persona", "INDY_READs"),
        "recipients": recipients,
        "subject": subject,
        "body": body,
        "body_sha256": sha256(body.encode()).hexdigest(),
        "send_requires_operator_approval": True,
        "status": "QUEUED_NOT_SENT",
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "status": "QUEUED_NOT_SENT",
        "outbox": str(outbox),
        "recipients": recipients,
        "subject": subject,
        "body_sha256": record["body_sha256"],
    }


def status(manifest: dict[str, Any]) -> dict[str, Any]:
    signal_tool = shutil.which("signal-cli")
    return {
        "schema": "lucidota.indy_reads.comms_status.v1",
        "persona": manifest.get("persona", "INDY_READs"),
        "email_mode": manifest["email"]["mode"],
        "email_recipients": manifest["email"]["recipients"],
        "signal_mode": manifest["signal"]["mode"],
        "signal_cli_present": bool(signal_tool),
        "signal_cli_path": signal_tool or "",
        "direct_chat_preferred": manifest["direct_chat"]["preferred"],
        "socket_path": manifest["direct_chat"]["socket_path"],
        "loopback_port": manifest["direct_chat"]["loopback_port"],
        "send_requires_operator_approval": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="indy-reads-comms")
    ap.add_argument("--manifest", default="04_RUNTIME/indy_reads_startup_comms_manifest.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue-email")
    q.add_argument("--manifest", dest="manifest_after", default=None)
    q.add_argument("--outbox", default=None)
    q.add_argument("--subject", required=True)
    q.add_argument("--body", required=True)
    st = sub.add_parser("status")
    st.add_argument("--manifest", dest="manifest_after", default=None)
    args = ap.parse_args()
    manifest_path = getattr(args, "manifest_after", None) or args.manifest
    manifest = load_manifest(Path(manifest_path))
    if args.cmd == "queue-email":
        outbox = Path(args.outbox or manifest["email"]["outbox"])
        print(json.dumps(queue_email(manifest, outbox, args.subject, args.body), sort_keys=True))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(manifest), sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
